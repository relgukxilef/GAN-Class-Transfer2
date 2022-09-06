
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
block_depth = 2
octaves = 7 # bottleneck = 2x2

batch_size = 8
steps = 100

min_noise = 0.5

residual = False
concat = True

predict_x = False # as opposed to epsilon

mixed_precision = True

preferred_type = tf.float16 if mixed_precision else tf.float32

gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

if mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

#optimizer = tf.keras.optimizers.SGD(0.01, 0.9, True)
optimizer = tf.keras.optimizers.Adam(2e-5)

if mixed_precision:
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

def alpha_dash(t):
    from math import pi
    t /= steps
    #return (256*256)**(-1*t)
    return (tf.cos(pi * t) + 1) / 2
    return (1 - t)**4 + 1e-5

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
                tf.keras.layers.Dense(self.filters, activation='relu'),
                tf.keras.layers.Conv2D(
                    self.filters, 3, 1, 'same', #use_bias=False,
                    dilation_rate=(self.dilation, self.dilation)
                ),
                #tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ]) for i in range(block_depth)
        ])
        self.module.build(input_shape)

    def call(self, input):
        return self.module(input)

class UpShuffle(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.convolution = tf.keras.layers.Conv2DTranspose(
            pixel_size, 4, 2, 'same', activation='relu'
        )

    def call(self, input):
        return self.convolution(input)

class DownShuffle(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.convolution = tf.keras.layers.Conv2D(
            pixel_size, 4, 2, 'same', activation='relu'
        )

    def call(self, input):
        return self.convolution(input)

@tf.function
def identity(y_true, y_pred):
    return tf.reduce_mean(y_pred)

class Denoiser(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.encoder = tf.keras.layers.Dense(pixel_size, activation='relu')
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

    def call(self, x):
        t = tf.random.uniform(
            tf.shape(x)[:-3], maxval=steps, dtype=x.dtype
        )[..., None, None, None]
        epsilon = tf.random.normal(tf.shape(x), dtype=x.dtype)

        scale = alpha_dash(t)

        noised = (
            x * tf.sqrt(scale) + 
            epsilon * tf.sqrt(1 - scale)
        )

        prediction = self.denoiser(noised)

        return tf.math.squared_difference(
            tf.cast(x if predict_x else epsilon, tf.float32), 
            tf.cast(prediction, tf.float32)
        )

denoiser = Denoiser()
trainer = Trainer(denoiser)

@tf.function
def decode_file(file, crop=True):
    image = tf.image.decode_jpeg(file, 3)
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

example_image = tf.cast(load_file(example_image_path, True), tf.float32)
example = tf.random.normal((1, 2, size, size, 3), dtype=tf.float32)

for folder in classes:
    datasets += [
        tf.data.Dataset.list_files(folder)
        .map(tf.io.read_file, num_parallel_calls=tf.data.AUTOTUNE)
        #.cache("cache") # disk cache
        .shuffle(1000).repeat()
        .map(decode_file, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ]

def log_sample(epochs, logs):
    with summary_writer.as_default():
        noised = (
            example_image[0][None] * tf.sqrt(1 - min_noise) + 
            example[0, :1, ...] * tf.sqrt(min_noise)
        )
        prediction = tf.cast(
            denoiser(tf.cast(noised, preferred_type)), tf.float32
        )
        if not predict_x:
            denoised = (
                noised - prediction * tf.sqrt(min_noise)
            ) / tf.sqrt(1 - min_noise)
        else:
            denoised = prediction
        tf.summary.image('identity', denoised * 0.5 + 0.5, epochs)
        tf.summary.scalar(
            'example loss', 
            tf.reduce_mean(tf.square(example_image[0][None] - denoised)), 
            epochs
        )
        del denoised

        fake = example[0, ...]

        for t_value in reversed(range(steps)):
            t = t_value
            prediction = tf.cast(
                denoiser(tf.cast(fake, preferred_type)), tf.float32
            )

            if predict_x:
                x_theta = prediction
                epsilon_theta = (
                    fake - alpha_dash(t)**0.5 * x_theta
                ) / (1 - alpha_dash(t))**0.5
                
            else:
                epsilon_theta = prediction
                x_theta = (
                    fake - (1 - alpha_dash(t))**0.5 * epsilon_theta
                ) / alpha_dash(t)**0.5

            if t > 0:
                fake = (
                    alpha_dash(t - 1)**0.5 * x_theta + 
                    (1 - alpha_dash(t - 1))**0.5 * epsilon_theta
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
    day = datetime.datetime.now().strftime("%Y%m%d")
    time = datetime.datetime.now().strftime("%H%M%S")
    summary_writer = tf.summary.create_file_writer(
        os.path.join("logs", day, time)
    )

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