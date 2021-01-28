
import datetime, os
import tensorflow as tf
import tqdm

size = 256
filters = 128

batch_size = 4

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

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
        return input + self.module(input)

#optimizer = tf.keras.optimizers.Adam()
#optimizer = tf.keras.optimizers.RMSprop(1e-5)
optimizer = tf.keras.optimizers.SGD(1e-4)

def block():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, 3, 1, 'same', activation='relu'),
        tf.keras.layers.Conv2D(filters, 3, 1, 'same')
    ])

generator = tf.keras.Sequential([
    tf.keras.layers.Dense(filters, activation='relu'),
    tf.keras.layers.Dense(filters * 8 * 8),
    tf.keras.layers.Reshape((8, 8, -1)),
    # 8x8
    tf.keras.layers.UpSampling2D(interpolation='bilinear'),
    Residual(block()),
    tf.keras.layers.UpSampling2D(interpolation='bilinear'),
    Residual(block()),
    tf.keras.layers.UpSampling2D(interpolation='bilinear'),
    Residual(block()),
    tf.keras.layers.UpSampling2D(interpolation='bilinear'),
    Residual(block()),
    tf.keras.layers.UpSampling2D(interpolation='bilinear'),
    Residual(block()),
    # 256x256
    tf.keras.layers.Dense(filters, activation='relu'),
    tf.keras.layers.Dense(3),
])
discriminator = tf.keras.Sequential([
    tf.keras.layers.Dense(filters, activation='relu'),
    tf.keras.layers.Dense(filters),
    # 256x256
    Residual(block()),
    tf.keras.layers.AveragePooling2D(2, 2),
    Residual(block()),
    tf.keras.layers.AveragePooling2D(2, 2),
    Residual(block()),
    tf.keras.layers.AveragePooling2D(2, 2),
    Residual(block()),
    tf.keras.layers.AveragePooling2D(2, 2),
    Residual(block()),
    tf.keras.layers.AveragePooling2D(2, 2),
    # 8x8
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(filters, activation='relu'),
    #PairwiseSum(),
    tf.keras.layers.Dense(filters, activation='relu'),
    tf.keras.layers.Dense(1),
])

@tf.function
def train_step(source_images, target_images, step):
    target_images = tf.image.random_flip_left_right(target_images)

    with tf.GradientTape(persistent=True) as tape:
        generated_images = generator(tf.random.normal([batch_size, filters]))

        real_output = discriminator(target_images)
        fake_output = discriminator(generated_images)

        #generator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        #    tf.ones_like(fake_output), fake_output
        #)
        #generator_loss = tf.keras.losses.MeanSquaredError()(0, fake_output)
        generator_loss = tf.reduce_mean(
            -0.5 * fake_output + tf.math.softplus(fake_output)
        )

        discriminator_loss = 0.5 * sum([
            tf.keras.losses.BinaryCrossentropy(from_logits=True)(
                tf.ones_like(real_output), real_output
            ),
            tf.keras.losses.BinaryCrossentropy(from_logits=True)(
                tf.zeros_like(fake_output), fake_output
            )
        ])
        #discriminator_loss = 0.5 * (
        #    tf.keras.losses.MeanSquaredError()(1, real_output) +
        #    tf.keras.losses.MeanSquaredError()(-1, fake_output)
        #)

    optimizer.apply_gradients(zip(
        tape.gradient(generator_loss, generator.trainable_variables), 
        generator.trainable_variables
    ))
    optimizer.apply_gradients(zip(
        tape.gradient(discriminator_loss, discriminator.trainable_variables), 
        discriminator.trainable_variables
    ))

def load_file(file, crop=True):
    image = tf.image.decode_jpeg(tf.io.read_file(file))[:, :, :3]
    if crop:
        image = tf.image.random_crop(image, [size, size, 3])
    return tf.cast(image, tf.float32) / 128 - 1

def get_encoding_summary(encoding):
    return tf.clip_by_value(
        tf.transpose(
            encoding[..., :8], [2, 0, 1]
        )[..., None] * 0.5 + 0.5, 
        0, 1
    )

classes = [
    "../Datasets/safebooru_r63_256/train/male/*",
    "../Datasets/safebooru_r63_256/train/female/*"
]

datasets = []

#example = load_file(
#    "../Datasets/safebooru_r63_256/train/male/" +
#    "00fdb833c64d7824edaa555277e494331b3882891f9422c6bca07611a5193b5f.png"
#)
example = load_file(
    "../Datasets/safebooru_r63_256/test/female/" +
    "00af2f4796bcf58f445ab78e4f8a42f4931c28eec024de0e79872fa019575c5f.png"
)

for folder in classes:
    dataset = tf.data.Dataset.list_files(folder)
    dataset = dataset.shuffle(1000).repeat()
    dataset = dataset.map(load_file).batch(batch_size).prefetch(8)
    datasets += [dataset]

name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(os.path.join("logs", name))

progress = tqdm.tqdm(tf.data.Dataset.zip(tuple(datasets)))
for step, (image_x, image_y) in enumerate(progress):
    train_step(image_x, image_y, tf.constant(step, tf.int64))
    
    if step % 100 == 0:
        with summary_writer.as_default():
            #tf.summary.scalar('generator_loss', generator_loss, step)
            #tf.summary.scalar('discriminator_loss', discriminator_loss, step)
            tf.summary.image(
                'fakes', 
                tf.clip_by_value(
                    generator(
                        tf.random.normal([4, filters])
                    ) * 0.5 + 0.5, 
                    0, 1
                ), 
                step, 4
            )