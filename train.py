
import datetime, os
import tensorflow as tf

dataset_pattern = "../Datasets/safebooru_r63_256/train/female/*.png"
example_image_path = "../Datasets/safebooru_r63_256/test/female/"\
"00af2f4796bcf58f445ab78e4f8a42f4931c28eec024de0e79872fa019575c5f.png"

#dataset_pattern = example_image_path

size = 256
pixel_size = 32
block_depth = 2
octaves = 6

batch_size = 2
steps = 200

residual = True
concat = False

mixed_precision = False # learning slows down to 0 with mixed precision

prefered_type = tf.float16 if mixed_precision else tf.float32

def log_image_schedule(r):
    # returns log of square of scale of image at step t/T
    return tf.math.log(0.0001**r)

gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

if mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

optimizer = tf.keras.optimizers.SGD(0.00001, 0.9, True)
#optimizer = tf.keras.optimizers.Adam(0.00001)

if mixed_precision:
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

class Residual(tf.keras.layers.Layer):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def build(self, input_shape):
        #self.module.build(input_shape)
        if residual or concat:
            self.dense = tf.keras.layers.Dense(input_shape[-1], use_bias=False)

    def call(self, input):
        if residual:
            return input + self.dense(self.module(input))
        elif concat:
            return self.dense(tf.concat(
                [tf.cast(self.module(input), input.dtype), input], -1
            ))
        else:
            return self.module(input)

class Block(tf.keras.layers.Layer):
    def __init__(self, filters, dilation = 1):
        super().__init__()

        self.filters = filters
        self.dilation = dilation

    def build(self, input_shape):
        self.module = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.filters, 3, 1, 'same', activation='relu', 
                dilation_rate=(self.dilation, self.dilation)
            ) for i in range(block_depth)
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
        self.middle = Block(pixel_size * 2**octaves)
        for i in reversed(range(octaves)):
            self.middle = Residual(
                tf.keras.Sequential([
                    DownShuffle(),
                    Block(pixel_size * 2**i),
                    self.middle, 
                    Block(pixel_size * 2**i),
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
        scale = tf.random.uniform(
            tf.shape(input)[:-3], dtype=input.dtype
        )[..., None, None, None]
        epsilon = tf.random.normal(tf.shape(input), dtype=input.dtype)

        noised = (
            input * scale + 
            epsilon * (1 - scale)
        )

        x_theta = self.denoiser((noised, 0))
        return tf.math.squared_difference(input, x_theta)

denoiser = Denoiser()
trainer = Trainer(denoiser)

def load_file(file, crop=True):
    image = tf.image.decode_jpeg(tf.io.read_file(file))[:, :, :3]
    if crop:
        image = tf.image.random_crop(image, [size, size, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, prefered_type) / 128 - 1
    return image, image

classes = [
    dataset_pattern,
]

datasets = []

example_image = load_file(example_image_path)
example = tf.random.normal((1, 4, size, size, 3), dtype=prefered_type)

for folder in classes:
    dataset = tf.data.Dataset.list_files(folder)
    dataset = dataset.shuffle(1000).repeat()
    dataset = dataset.map(load_file).batch(batch_size).prefetch(8)
    datasets += [dataset]

def log_sample(epochs, logs):
    with summary_writer.as_default():
        noised = (
            example_image[0][None] * 0.5 + example[0, :1, ...] * 0.5
        )
        identity = denoiser((
            noised, 
            0
        ))
        tf.summary.image('identity', identity * 0.5 + 0.5, epochs)
        del identity

        fake = example[0, ...]
        for t in reversed(range(1, steps + 1)):
            x_theta = denoiser((fake, 0))

            fake = (
                fake * 0.98 + 
                x_theta * 0.02
            )

            if t == 1:
                tf.summary.image('step_0', x_theta * 0.5 + 0.5, epochs, 4)
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