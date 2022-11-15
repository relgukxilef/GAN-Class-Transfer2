
import datetime, os
import tensorflow as tf

dataset_pattern = "../Datasets/safebooru_r63_256/train/female/*.png"
#dataset_pattern = "D:/Felix/Downloads/danbooru2020/256px/*/*"
#dataset_pattern = \
#"D:/Felix/Documents/Python/PyTorch Projects/Datasets/saki_castle/256px/*.jpg"

example_image_path = "../Datasets/safebooru_r63_256/test/female/"\
"00af2f4796bcf58f445ab78e4f8a42f4931c28eec024de0e79872fa019575c5f.png"
#example_image_path = "D:/Felix/Documents/Python/PyTorch Projects/Datasets/"\
#"saki_castle/256px/1039 saki_castle.jpg"

dataset_pattern = example_image_path

size = 256
pixel_size = 128 * 1
max_size = 512 * 1
block_depth = 1
octaves = 7 # bottleneck = 2x2

batch_size = 8
steps = 50

residual = False
concat = True

predict_x = False # as opposed to epsilon
predict_scaled_epsilon = False
prediction_weighting = False
ordinary_differential_equation = False

mixed_precision = False

preferred_type = tf.float16 if mixed_precision else tf.float32

gpu = tf.config.list_physical_devices('GPU')[0]
#tf.config.experimental.set_memory_growth(gpu, True)

if mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

def sign_gradient(gradient):
    return [(tf.sign(g), v) for g, v in gradient]

#optimizer = tf.keras.optimizers.SGD(0.25, 0.5, True)
optimizer = tf.keras.optimizers.SGD(
    tf.keras.optimizers.schedules.InverseTimeDecay(2.0, 10_000, 1)
)
#optimizer = tf.keras.optimizers.SGD(
#    0.0001,#, 0.9, True, 
#    gradient_transformers=[sign_gradient]
#)
#optimizer = tf.keras.optimizers.Adam(1e-4)
#optimizer = tf.keras.optimizers.RMSprop(
#    tf.keras.optimizers.schedules.InverseTimeDecay(1e-5, 10_000, 1)
#)

regularizer = tf.keras.regularizers.l2(1e-6)

if mixed_precision:
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

def alpha_dash(t):
    from math import pi
    t /= (steps + 1)
    #return 1 - 2**(t - 1)
    #return (2**8 - 2**8**t) / (256 * 2**8**t - 2**8**t + 2**8)
    #return (256*256)**(-1*t)
    #return tf.cos(pi / 2 * t)**2
    #return (1 - t)**4
    return (1 - t)**2

test_step = 25

class Residual(tf.keras.layers.Layer):
    def __init__(self, module, highway = lambda x: x):
        super().__init__()

        self.module = module
        self.highway = highway

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
                [
                    tf.cast(self.module(input), input.dtype), 
                    self.highway(input)
                ], -1
            )
        else:
            return self.module(input)

class Block(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()

        self.filters = filters

    def build(self, input_shape):
        self.module = tf.keras.Sequential([
            tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    self.filters, 3, 1, 'same', 
                    kernel_initializer='he_uniform', activation='relu', 
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer
                ),
            ]) for i in range(block_depth)
        ])
        self.module.build(input_shape)

    def call(self, input):
        return self.module(input)

class UpShuffle(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.convolution = tf.keras.layers.Conv2DTranspose(
            filters, 4, 2, 'same', kernel_initializer='he_uniform', 
            activation='relu', 
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer
        )

    def call(self, input):
        return self.convolution(input)

class DownShuffle(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.convolution = tf.keras.layers.Conv2D(
            filters, 4, 2, 'same', kernel_initializer='he_uniform', 
            activation='relu', 
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer
        )

    def call(self, input):
        return self.convolution(input)

@tf.function
def identity(y_true, y_pred):
    return tf.reduce_mean(y_pred)

class Denoiser(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.middle = Block(min(pixel_size * 2**octaves, max_size))
        for i in reversed(range(octaves)):
            filters = min(pixel_size * 2**i, max_size)
            self.middle = Residual(
                tf.keras.Sequential([
                    DownShuffle(filters),
                    #Block(filters),
                    self.middle, 
                    #Block(filters),
                    UpShuffle(min(pixel_size * 2**i // 2, max_size)),
                ])
            )
        self.middle = tf.keras.Sequential([
            #Block(pixel_size),
            self.middle,
            #Block(pixel_size),
            #tf.keras.layers.Dense(
            #    pixel_size, kernel_initializer='he_uniform', activation='relu'
            #),
            tf.keras.layers.Dense(
                3 * steps, #kernel_initializer='zeros', 
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer
            ),
            tf.keras.layers.Reshape((size, size, steps, 3)),
        ])

    def call(self, input):
        x, t = input
        t = t[..., 0]
        t = tf.broadcast_to(t, tf.shape(x)[:-1])
        prediction = self.middle(x)
        x_theta = tf.gather(
            # [batch, width, height, steps, channels]
            prediction, t - 1, batch_dims=3
        )
        return x_theta

class Trainer(tf.keras.Model):
    def __init__(self, denoiser):
        super().__init__()

        self.denoiser = denoiser

    def call(self, x):
        t_int = tf.random.uniform(
            tf.shape(x)[:-3], minval=1, maxval=steps + 1, dtype=tf.int32
        )[..., None, None, None]
        epsilon = tf.random.normal(tf.shape(x), dtype=x.dtype)

        t = tf.cast(t_int, x.dtype)

        noised = (
            x * alpha_dash(t)**0.5 + 
            epsilon * (1 - alpha_dash(t))**0.5
        )

        prediction = self.denoiser((noised, t_int))

        if ordinary_differential_equation:
            target = (
                x * alpha_dash(t - 1)**0.5 + 
                epsilon * (1 - alpha_dash(t - 1))**0.5
            )
        elif predict_x:
            target = x
        else:
            target = epsilon
            if predict_scaled_epsilon:
                target *= (1 - alpha_dash(t))**0.5

            if prediction_weighting:
                target *= (1 - alpha_dash(t))**0.5
                prediction *= (1 - alpha_dash(t))**0.5

        def dct2d(x):
            frequency_weights = 1.0 / tf.range(1, size + 1, dtype=x.dtype)
            x = tf.transpose(x, [0, 3, 1, 2])
            x = tf.signal.dct(x, norm='ortho') * frequency_weights
            x = tf.transpose(x, [0, 1, 3, 2])
            x = tf.signal.dct(x, norm='ortho') * frequency_weights
            return tf.transpose(x, [0, 2, 3, 1])

        target = tf.cast(target, tf.float32)
        prediction = tf.cast(prediction, tf.float32)

        #return dct2d(target - prediction)**2

        #tf.abs crashes
        #return tf.reduce_mean(
        #    tf.maximum(target - prediction, prediction - target)
        #)

        return tf.reduce_mean(tf.math.squared_difference(target, prediction))

        return (
            tf.reduce_mean(tf.math.squared_difference(target, prediction)) +
            tf.reduce_mean(tf.math.squared_difference(
                tf.nn.avg_pool2d(target, 16, 16, 'SAME'), 
                tf.nn.avg_pool2d(prediction, 16, 16, 'SAME')
            ))
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

bits_per_pixel = 3
dictionary = tf.random.normal(
    (size, size, 2**bits_per_pixel, 3), dtype=tf.float32
)

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
        image_factor = alpha_dash(test_step)
        if ordinary_differential_equation:
            # need adjascent alpha_dash values
            image_factor = alpha_dash(steps / 2)**0.5
        noised = (
            example_image[0][None] * image_factor**0.5 + 
            example[0, :1, ...] * (1 - image_factor)**0.5
        )
        prediction = tf.cast(
            denoiser((tf.cast(noised, preferred_type), 
            tf.constant([test_step]))), 
            tf.float32
        )
        if ordinary_differential_equation:
            denoised = (
                prediction * (1 - alpha_dash(steps / 2))**0.5 -
                noised * (1 - alpha_dash(steps / 2 - 1))**0.5
            ) / (
                alpha_dash(steps / 2 - 1)**0.5 * 
                (1 - alpha_dash(steps / 2))**0.5 -
                alpha_dash(steps / 2)**0.5 *
                (1 - alpha_dash(steps / 2 - 1))**0.5
            )
        elif predict_x:
            denoised = prediction
        else:
            if not predict_scaled_epsilon:
                prediction = prediction * (1 - image_factor)**0.5
            denoised = (
                noised - prediction
            ) / image_factor**0.5
        tf.summary.image('denoised', denoised * 0.5 + 0.5, epochs)
        tf.summary.scalar(
            'example loss', 
            tf.reduce_mean((example_image[0][None] - denoised)**2)**0.5, 
            epochs
        )
        del denoised

        # forward diffusion
        if True:
            x_theta = example_image[0][None]
            epsilon_theta = x_theta # might be close enough

            for t_value in reversed(range(steps, 0, -1)): # smallest t = 1
                t = t_value

                fake = (
                    alpha_dash(t)**0.5 * x_theta + 
                    (1 - alpha_dash(t))**0.5 * epsilon_theta
                )

                prediction = tf.cast(
                    denoiser((tf.cast(fake, preferred_type), tf.constant([t]))), 
                    tf.float32
                )

                if ordinary_differential_equation:
                    x_theta = (
                        prediction * (1 - alpha_dash(t))**0.5 -
                        fake * (1 - alpha_dash(t - 1))**0.5
                    ) / (
                        alpha_dash(t - 1)**0.5 * 
                        (1 - alpha_dash(t))**0.5 -
                        alpha_dash(t)**0.5 *
                        (1 - alpha_dash(t - 1))**0.5
                    )
                    fake = 2 * fake - prediction # TODO
                else:
                    if predict_x:
                        x_theta = prediction
                        epsilon_theta = (
                            fake - alpha_dash(t)**0.5 * x_theta
                        ) / (1 - alpha_dash(t))**0.5
                        
                    else:
                        if predict_scaled_epsilon:
                            epsilon_theta = (
                                prediction / (1 - alpha_dash(t))**0.5
                            )
                            scaled_epsilon = prediction
                        else:
                            epsilon_theta = prediction
                            scaled_epsilon = (
                                prediction * (1 - alpha_dash(t))**0.5
                            )
                        x_theta = (
                            fake - scaled_epsilon
                        ) / alpha_dash(t)**0.5

        # backward diffusion
        # TODO: apply transformations to epsilon_theta

        pixelated = tf.keras.layers.UpSampling2D(4, interpolation='nearest')(
            tf.nn.avg_pool2d(epsilon_theta, 4, 4, 'SAME')
        )

        shifted = tf.roll(tf.roll(epsilon_theta, 1, 1), 1, 2)

        quantisation_error = tf.reduce_sum(tf.math.squared_difference(
            epsilon_theta[..., None, :], dictionary
        ), -1)
        quantised = tf.gather(
            dictionary[None, ...], tf.argmin(quantisation_error, -1), 
            batch_dims=3
        )

        fake = tf.concat([epsilon_theta, pixelated, shifted, quantised], 0)

        fake = tf.concat([example[0, ...], fake], 0)

        x_theta = fake
        epsilon_theta = fake

        for t_value in range(steps, 0, -1): # smallest t = 1
            t = t_value

            fake = (
                alpha_dash(t)**0.5 * x_theta + 
                (1 - alpha_dash(t))**0.5 * epsilon_theta
            )

            prediction = tf.cast(
                denoiser((tf.cast(fake, preferred_type), tf.constant([t]))), 
                tf.float32
            )

            if ordinary_differential_equation:
                x_theta = (
                    prediction * (1 - alpha_dash(t))**0.5 -
                    fake * (1 - alpha_dash(t - 1))**0.5
                ) / (
                    alpha_dash(t - 1)**0.5 * 
                    (1 - alpha_dash(t))**0.5 -
                    alpha_dash(t)**0.5 *
                    (1 - alpha_dash(t - 1))**0.5
                )
                fake = prediction
            else:
                if predict_x:
                    x_theta = prediction
                    epsilon_theta = (
                        fake - alpha_dash(t)**0.5 * x_theta
                    ) / (1 - alpha_dash(t))**0.5
                    
                else:
                    if predict_scaled_epsilon:
                        epsilon_theta = prediction / (1 - alpha_dash(t))**0.5
                        scaled_epsilon = prediction
                    else:
                        epsilon_theta = prediction
                        scaled_epsilon = prediction * (1 - alpha_dash(t))**0.5
                    x_theta = (
                        fake - scaled_epsilon
                    ) / alpha_dash(t)**0.5

            # epsilon_theta' = 
            #     f(a_t**0.5 * x_theta + (1 - a_t)**0.5 * epsilon_theta)
            # x_theta' = 
            #     x_theta + (epsilon_theta - epsilon_theta') * 
            #     (1 - a_t)**0.5 / a_t**0.5
            # last term is very large for t close to 1

            if t == steps:
                tf.summary.image('step_1', x_theta * 0.5 + 0.5, epochs, 10)
            if t == steps // 4:
                tf.summary.image('step_0.25', x_theta * 0.5 + 0.5, epochs, 10)
            if t == 2 * steps // 4:
                tf.summary.image('step_0.5', x_theta * 0.5 + 0.5, epochs, 10)
            if t == 3 * steps // 4:
                tf.summary.image('step_0.75', x_theta * 0.5 + 0.5, epochs, 10)
        tf.summary.image('fake', x_theta * 0.5 + 0.5, epochs, 10)

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
            ),
        ]
    )
