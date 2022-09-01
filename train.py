
import datetime, os
import tensorflow as tf

#dataset_pattern = "../Datasets/safebooru_r63_256/train/female/*.png"
dataset_pattern = "D:/Felix/Downloads/danbooru2020/256px/*/*"
example_image_path = "../Datasets/safebooru_r63_256/test/female/"\
"00af2f4796bcf58f445ab78e4f8a42f4931c28eec024de0e79872fa019575c5f.png"

#dataset_pattern = example_image_path

size = 256
pixel_size = 64
max_size = 512
block_depth = 1
octaves = 7 # bottleneck = 2x2

batch_size = 8
steps = 100

min_noise = 0.5

residual = False
concat = True

mixed_precision = True

preferred_type = tf.float16 if mixed_precision else tf.float32

gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

if mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

min_noise = tf.cast(min_noise, preferred_type)

#optimizer = tf.keras.optimizers.SGD(0.01, 0.9, True)
optimizer = tf.keras.optimizers.Adam(1e-5)

if mixed_precision:
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

def alpha_dash(t):
    t /= steps
    return 1 - 3 * t**2 + 2 * t**3

def alpha(t):
    return alpha_dash(t) / alpha_dash(t - 1)

class Residual(tf.keras.layers.Layer):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def build(self, input_shape):
        #self.module.build(input_shape)
        if residual:
            self.dense = tf.keras.layers.Dense(input_shape[-1], use_bias=False)
            pass

    def call(self, input):
        if residual:
            return input + self.dense(self.module(input))
        elif concat:
            return tf.concat(
                [tf.cast(self.module(input), input.dtype), input], -1
            )
        else:
            return self.module(input)

class Block(tf.keras.layers.Layer):
    def __init__(self, filters, dilation = 1):
        super().__init__()

        self.filters = filters
        self.dilation = dilation

    def build(self, input_shape):
        self.module = tf.keras.Sequential([
            tf.keras.Sequential([
                tf.keras.layers.Dense(self.filters),
                tf.keras.layers.Conv2D(
                    self.filters, 3, 1, 'same', activation='relu', 
                    dilation_rate=(self.dilation, self.dilation)
                ),
            ]) for i in range(block_depth)
        ])
        self.module.build(input_shape)

    def call(self, input):
        return self.module(input)

class UpShuffle(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, input):
        return tf.keras.layers.UpSampling2D(interpolation='bilinear')(
            input# * 0.5
        )

class DownShuffle(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, input):
        return tf.keras.layers.AveragePooling2D()(input)# * 2.0

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.rgb_dense = tf.keras.layers.Dense(pixel_size, use_bias=False)
        #self.scale_dense = tf.keras.layers.Dense(pixel_size)
        self.output_dense = tf.keras.layers.Dense(pixel_size)

    def build(self, input_shape):
        rgb, scale = input_shape
        self.rgb_dense.build(rgb)
        #self.scale_dense.build(scale)
        self.output_dense.build([pixel_size])

    def call(self, input):
        rgb, scale = input
        return self.output_dense(tf.nn.relu(
            self.rgb_dense(rgb)# + 
            #self.scale_dense(scale)
        ))

@tf.function
def identity(y_true, y_pred):
    return tf.reduce_mean(y_pred)

class Denoiser(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.middle = Block(min(pixel_size * 2**octaves, max_size))
        for i in reversed(range(octaves)):
            self.middle = Residual(
                tf.keras.Sequential([
                    DownShuffle(),
                    Block(min(pixel_size * 2**i, max_size)),
                    self.middle, 
                    Block(min(pixel_size * 2**i, max_size)),
                    UpShuffle(),
                ])
            )
        self.middle = tf.keras.Sequential([
            Block(pixel_size),
            self.middle,
            Block(pixel_size),
            tf.keras.layers.Dense(pixel_size, activation='relu'),
            tf.keras.layers.Dense(3),
        ])

    def call(self, input):
        x_theta = self.middle(self.encoder(input))
        return x_theta

class Trainer(tf.keras.Model):
    def __init__(self, denoiser):
        super().__init__()

        self.denoiser = denoiser

    def call(self, input):
        t = tf.random.uniform(
            tf.shape(input)[:-3], maxval=steps, dtype=input.dtype
        )[..., None, None, None]
        epsilon = tf.random.normal(tf.shape(input), dtype=input.dtype)

        scale = alpha_dash(t)
        #scale = min_noise
        
        # exponential increase in SNR
        #scale = tf.sqrt(2**(octaves*scale)/(2**(octaves*scale) + 2**octaves)) 

        noised = (
            input * tf.sqrt(scale) + 
            epsilon * tf.sqrt(1 - scale)
        )

        epsilon_theta = self.denoiser((noised, 0))

        return tf.math.squared_difference(
            tf.cast(epsilon, tf.float32), 
            tf.cast(epsilon_theta, tf.float32)
        )

denoiser = Denoiser()
trainer = Trainer(denoiser)

@tf.function
def decode_file(file, crop=False):
    image = tf.image.decode_jpeg(file)[:, :, :3]
    if crop:
        image = tf.image.random_crop(image, [size, size, 3])
    image = tf.broadcast_to(image, [size, size, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, preferred_type) / 128 - 1
    return image, image

@tf.function
def load_file(file, crop=False):
    return decode_file(tf.io.read_file(file), crop)

classes = [
    dataset_pattern,
]

datasets = []

example_image = load_file(example_image_path, True)
example = tf.random.normal((1, 2, size, size, 3), dtype=preferred_type)

for folder in classes:
    datasets += [
        tf.data.Dataset.list_files(folder)
        .map(tf.io.read_file, num_parallel_calls=tf.data.AUTOTUNE)
        .cache("cache") # disk cache
        .map(decode_file, num_parallel_calls=tf.data.AUTOTUNE)
        .cache() # memory cache
        .shuffle(1000).repeat()
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ]

def log_sample(epochs, logs):
    with summary_writer.as_default():
        noised = (
            example_image[0][None] * tf.sqrt(1 - min_noise) + 
            example[0, :1, ...] * tf.sqrt(min_noise)
        )
        identity = (
            noised - denoiser((noised, 0)) * tf.sqrt(min_noise)
        ) / tf.sqrt(1 - min_noise)
        tf.summary.image('identity', identity * 0.5 + 0.5, epochs)
        tf.summary.scalar(
            'example loss', 
            tf.reduce_mean(tf.square(example_image[0][None] - identity)), 
            epochs
        )
        del identity

        fake = example[0, ...]

        for t in range(steps, 1, -1): # lowest t = 2
            epsilon_theta = tf.clip_by_value(denoiser((fake, 0)), -4, 4)

            x_theta = (
                (fake - (1 - alpha(t))**0.5 * epsilon_theta) / 
                alpha(t)**0.5
            )

            fake = (
                alpha(t - 1)**0.5 * x_theta + 
                (1 - alpha(t - 1))**0.5 * epsilon_theta
            )

            if t == steps - 1:
                tf.summary.image('step_1', x_theta * 0.5 + 0.5, epochs, 4)
            if t == steps // 4:
                tf.summary.image('step_0.25', x_theta * 0.5 + 0.5, epochs, 4)
            if t == 2 * steps // 4:
                tf.summary.image('step_0.5', x_theta * 0.5 + 0.5, epochs, 4)
            if t == 3 * steps // 4:
                tf.summary.image('step_0.75', x_theta * 0.5 + 0.5, epochs, 4)
        tf.summary.image('fake', x_theta * 0.5 + 0.5, epochs, 4)

if __name__ == "__main__":
    name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(os.path.join("logs", name))

    dataset_example = next(iter(datasets[0]))[0]
    loss = identity(
        dataset_example, trainer(dataset_example)
    )
    del loss, dataset_example

    trainer.compile(
        optimizer, 
        identity
    )

    trainer.fit(
        datasets[0], steps_per_epoch=1000, epochs=1000,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_epoch_begin=log_sample
            )
        ]
    )