
import datetime, os
import tensorflow as tf
import tqdm

size = 256
pixel_size = 128
code_size = 128

batch_size = 1

gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

policy = tf.keras.mixed_precision.Policy('mixed_float16')
#tf.keras.mixed_precision.experimental.set_policy(policy)

#optimizer = tf.keras.optimizers.Adam()
optimizer_g = tf.keras.optimizers.Adam(1e-5, 0, 0.99)
optimizer_d = tf.keras.optimizers.Adam(1e-5, 0, 0.99)

#optimizer_g, optimizer_d = [
#    tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
#    for optimizer in [optimizer_g, optimizer_d]
#]

class PairwiseSum(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, input):
        a, b = tf.split(input, 2, 0)
        return a + b

class Residual(tf.keras.layers.Layer):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def build(self, input_shape):
        self.module.build(input_shape)

    def call(self, input):
        return self.module(input)
        return input + self.module(input)

class Block(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()

        self.filters = filters

    def build(self, input_shape):
        self.module = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.filters, 3, 1, 'same', activation='relu'
            ),
            tf.keras.layers.Conv2D(
                input_shape[-1], 3, 1, 'same'#, kernel_initializer='zeros'
            )
        ])
        self.module.build(input_shape)

    def call(self, input):
        return self.module(input)

class UpShuffle(tf.keras.layers.Layer):
    def __init__(self, size):
        super().__init__()

        self.size = size

    def call(self, input):
        return tf.keras.layers.UpSampling2D()(input)

class DownShuffle(tf.keras.layers.Layer):
    def __init__(self, size):
        super().__init__()

        self.size = size

    def call(self, input):
        return tf.keras.layers.AveragePooling2D()(input)

generator = tf.keras.Sequential([
    tf.keras.layers.Dense(code_size, activation='relu'),
    tf.keras.layers.Dense(code_size, activation='relu'),
    tf.keras.layers.Dense(8 * 8 * 256),
    tf.keras.layers.Reshape((8, 8, 256)),
    # 8x8
    Residual(Block(pixel_size)), UpShuffle(2),
    Residual(Block(pixel_size)), UpShuffle(2),
    Residual(Block(pixel_size)), UpShuffle(2),
    Residual(Block(pixel_size)), UpShuffle(2),
    Residual(Block(pixel_size)), UpShuffle(2),
    Residual(Block(pixel_size)), 
    # 256x256
    tf.keras.layers.Dense(pixel_size, activation='relu'),
    tf.keras.layers.Dense(3),
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Dense(pixel_size, activation='relu'),
    tf.keras.layers.Dense(256),
    # 256x256
    Residual(Block(pixel_size)), DownShuffle(2),
    Residual(Block(pixel_size)), DownShuffle(2),
    Residual(Block(pixel_size)), DownShuffle(2),
    Residual(Block(pixel_size)), DownShuffle(2),
    Residual(Block(pixel_size)), DownShuffle(2),
    Residual(Block(pixel_size)), 
    # 8x8
    tf.keras.layers.Reshape((-1,)),
    tf.keras.layers.Dense(code_size, activation='relu'),
    #PairwiseSum(),
    tf.keras.layers.Dense(code_size, activation='relu'),
    tf.keras.layers.Dense(code_size, activation='relu'),
    tf.keras.layers.Dense(1, kernel_initializer='zeros'),
])

@tf.function
def train_step(source_images, target_images, step):
    target_images = tf.image.random_flip_left_right(target_images)

    with tf.GradientTape(persistent=True) as tape:
        generated_images = generator(
            tf.random.normal((batch_size, code_size))
        )

        real_output = discriminator(target_images)
        fake_output = discriminator(generated_images)

        #generator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        #    tf.ones_like(fake_output), fake_output
        #)
        generator_loss = tf.keras.losses.MeanSquaredError()(0, fake_output)
        #generator_loss = tf.reduce_mean(
        #    -0.5 * fake_output + tf.math.softplus(fake_output)
        #)

        #discriminator_loss = 0.5 * sum([
        #    tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        #        tf.ones_like(real_output), real_output
        #    ),
        #    tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        #        tf.zeros_like(fake_output), fake_output
        #    )
        #])
        discriminator_loss = 0.5 * (
            tf.keras.losses.MeanSquaredError()(1, real_output) +
            tf.keras.losses.MeanSquaredError()(-1, fake_output)
        )

    optimizer_g.apply_gradients(zip(
        tape.gradient(generator_loss, generator.trainable_variables), 
        generator.trainable_variables
    ))
    optimizer_d.apply_gradients(zip(
        tape.gradient(discriminator_loss, discriminator.trainable_variables), 
        discriminator.trainable_variables
    ))

def load_file(file, crop=True):
    image = tf.image.decode_jpeg(tf.io.read_file(file))[:, :, :3]
    if crop:
        image = tf.image.random_crop(image, [size, size, 3])
    return tf.cast(image, tf.float32) / 128 - 1

classes = [
    "../Datasets/safebooru_r63_256/train/male/*",
    "../Datasets/safebooru_r63_256/train/female/*"
]

datasets = []

example = load_file(
    "../Datasets/safebooru_r63_256/train/male/" +
    "00fdb833c64d7824edaa555277e494331b3882891f9422c6bca07611a5193b5f.png"
)
example = tf.random.normal((4, code_size))

for folder in classes:
    dataset = tf.data.Dataset.list_files(folder)
    dataset = dataset.shuffle(1000).repeat()
    dataset = dataset.map(load_file).batch(batch_size).prefetch(8)
    datasets += [dataset]

name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(os.path.join("logs", name))

progress = tqdm.tqdm(tf.data.Dataset.zip(tuple(datasets)))
try:
    for step, (image_x, image_y) in enumerate(progress):
        train_step(image_x, image_y, tf.constant(step, tf.int64))
        
        if step % 100 == 0:
            with summary_writer.as_default():
                tf.summary.image(
                    'fakes', 
                    tf.clip_by_value(
                        generator(example) * 0.5 + 0.5, 
                        0, 1
                    ), 
                    step, 4
                )
finally:
    pass