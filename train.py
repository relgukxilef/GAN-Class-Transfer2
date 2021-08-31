
import datetime, os
import tensorflow as tf
import tensorflow_probability as tfp

size = 256
pixel_size = 128

batch_size = 1

gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

policy = tf.keras.mixed_precision.Policy('mixed_float16')
#tf.keras.mixed_precision.experimental.set_policy(policy)

optimizer = tf.keras.optimizers.Adam()

#optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

class Residual(tf.keras.layers.Layer):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def build(self, input_shape):
        self.module.build(input_shape)

    def call(self, input):
        #return self.module(input)
        return input + self.module(input)

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
            ),
            tf.keras.layers.Conv2D(
                input_shape[-1], 3, 1, 'same', #kernel_initializer='zeros',
                dilation_rate=(self.dilation, self.dilation)
            )
        ])
        self.module.build(input_shape)

    def call(self, input):
        return self.module(input)

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.rgb_dense = tf.keras.layers.Dense(pixel_size, use_bias=False)
        self.scale_dense = tf.keras.layers.Dense(pixel_size)
        self.output_dense = tf.keras.layers.Dense(pixel_size)

    def build(self, input_shape):
        rgb, scale = input_shape
        self.rgb_dense.build(rgb)
        self.scale_dense.build(scale)
        self.output_dense.build([pixel_size])

    def call(self, input):
        rgb, scale = input
        return self.output_dense(tf.nn.relu(
            self.rgb_dense(rgb) + 
            self.scale_dense(scale)
        ))

class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, input):
        rgb, scale = tf.split(input, [3, 1], -1)
        scale = tf.reduce_mean(scale, axis=[-1, -2, -3], keepdims=True)
        return rgb, scale

@tf.function
def identity(y_true, y_pred):
    return tf.reduce_mean(y_pred)

class Denoiser(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.middle = tf.keras.Sequential([
            Residual(Block(pixel_size, 64)),
            Residual(Block(pixel_size, 32)),
            Residual(Block(pixel_size, 16)),
            Residual(Block(pixel_size, 8)),
            Residual(Block(pixel_size, 4)),
            Residual(Block(pixel_size, 2)),
            Residual(Block(pixel_size)), 
            tf.keras.layers.Dense(pixel_size, activation='relu'),
            tf.keras.layers.Dense(4),
        ])
        self.decoder = Decoder()

    def call(self, input):
        return self.decoder(self.middle(self.encoder(input)))

class Trainer(tf.keras.Model):
    def __init__(self, denoiser):
        super().__init__()

        self.denoiser = denoiser

    def call(self, input):
        log_scale = tf.math.log(tf.random.uniform(
            tf.shape(input)[:-3], 
            1.0/256, 1.0
        ))[..., None, None, None]
        scale = tf.exp(log_scale)
        epsilon = tf.random.normal(tf.shape(input))

        noised = (
            input * tf.sqrt(1 - tf.square(scale)) + 
            epsilon * scale
        )

        rgb, log_scale = self.denoiser((noised, log_scale))
        return -tfp.distributions.Normal(
            rgb, tf.exp(log_scale)
        ).log_prob(input)

denoiser = Denoiser()
trainer = Trainer(denoiser)

def load_file(file, crop=True):
    image = tf.image.decode_jpeg(tf.io.read_file(file))[:, :, :3]
    if crop:
        image = tf.image.random_crop(image, [size, size, 3])
    image = tf.cast(image, tf.float32) / 128 - 1
    return image, image

classes = [
    "../Datasets/safebooru_r63_256/train/male/*",
    "../Datasets/safebooru_r63_256/train/female/*"
]

datasets = []

example_image = load_file(
    "../Datasets/safebooru_r63_256/train/male/" +
    "00fdb833c64d7824edaa555277e494331b3882891f9422c6bca07611a5193b5f.png"
)
example = tf.random.normal((4, size, size, 3))

for folder in classes:
    dataset = tf.data.Dataset.list_files(folder)
    dataset = dataset.shuffle(1000).repeat()
    dataset = dataset.map(load_file).batch(batch_size).prefetch(8)
    datasets += [dataset]

@tf.function
def log_sample(epochs, logs):
    with summary_writer.as_default():
        identity, _ = denoiser((
            example_image[0][None], 
            tf.math.log(1.0 / 256)[None, None, None, None]
        ))
        tf.summary.image('identity', identity * 0.5 + 0.5, epochs)
        del identity

        sample = example
        log_scale = tf.math.log(1.0)[None, None, None, None]
        for i in range(20):
            fake, log_scale = denoiser((sample, log_scale))
            scale = tf.exp(log_scale)
            epsilon = tf.random.normal(tf.shape(fake))

            sample = (
                fake * tf.sqrt(1 - tf.square(scale)) + 
                epsilon * scale
            )

            if i == 0:
                tf.summary.image('first_step', fake * 0.5 + 0.5, epochs, 4)
        tf.summary.image('fake', fake * 0.5 + 0.5, epochs, 4)
        tf.summary.image('sample', sample * 0.5 + 0.5, epochs, 4)
        tf.summary.histogram('log_scale', log_scale, epochs)

if __name__ == "__main__":
    name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(os.path.join("logs", name))

    dataset_example = next(iter(datasets[1]))[0]
    loss = identity(
        dataset_example, trainer(dataset_example)
    )
    del loss, dataset_example

    trainer.compile(
        #tf.keras.optimizers.SGD(1e-5), 
        tf.keras.optimizers.Adam(1e-6), 
        identity
    )

    trainer.fit(
        datasets[1], steps_per_epoch=1000, epochs=100,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_epoch_begin=log_sample
            )
        ]
    )