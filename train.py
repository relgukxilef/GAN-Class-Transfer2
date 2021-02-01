
import datetime, os
import tensorflow as tf
#import tensorflow_probability as tfp
import tqdm

size = 256
pixel_size = 64

batch_size = 8

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

#optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.RMSprop(1e-4)

class Residual(tf.keras.layers.Layer):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def build(self, input_shape):
        self.module.build(input_shape)

    def call(self, input):
        return input + self.module(input)

class Block(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()

        self.filters = filters

    def build(self, input_shape):
        self.module = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.filters, 4, 1, 'same', activation='relu'
            ),
            tf.keras.layers.Conv2D(
                input_shape[-1], 3, 1, 'same', kernel_initializer='zeros'
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
        return tf.nn.depth_to_space(input, self.size)

class DownShuffle(tf.keras.layers.Layer):
    def __init__(self, size):
        super().__init__()

        self.size = size

    def call(self, input):
        return tf.nn.space_to_depth(input, self.size)

denoiser = tf.keras.Sequential([
    tf.keras.layers.Dense(pixel_size, activation='relu'),
    tf.keras.layers.Dense(pixel_size),
    # 256x256
    # DownShuffle(32) is not equivalent! 
    DownShuffle(2), DownShuffle(2), DownShuffle(2),
    DownShuffle(2), DownShuffle(2),
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

@tf.function
def train_step(source_images, target_images, step):
    target_images = tf.image.random_flip_left_right(target_images)

    with tf.GradientTape() as tape:
        generated_images = denoiser(
            tf.random.normal(tf.shape(target_images), target_images, 1.0)
        )

        generator_loss = tf.keras.losses.MeanSquaredError()(
            generated_images, target_images
        )

    variables = denoiser.trainable_variables

    optimizer.apply_gradients(zip(
        tape.gradient(generator_loss, variables), variables
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

example_a = load_file(
    "../Datasets/safebooru_r63_256/train/male/" +
    "00fdb833c64d7824edaa555277e494331b3882891f9422c6bca07611a5193b5f.png"
)
example_b = load_file(
    "../Datasets/safebooru_r63_256/test/female/" +
    "00af2f4796bcf58f445ab78e4f8a42f4931c28eec024de0e79872fa019575c5f.png"
)
example_b = tf.random.normal(tf.shape(example_b), example_b, 1.0)

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
                        denoiser(example_a[None]) * 0.5 + 0.5, 
                        0, 1
                    ), 
                    step, 4
                )
                tf.summary.image(
                    'denoised', 
                    tf.clip_by_value(
                        denoiser(example_b[None]) * 0.5 + 0.5, 
                        0, 1
                    ), 
                    step, 4
                )
finally:
    pass