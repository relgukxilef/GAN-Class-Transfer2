
import datetime, os
import tensorflow as tf
#import tensorflow_probability as tfp
import tqdm

size = 256
code_size = 1024
pixel_size = 32

batch_size = 8

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

#optimizer = tf.keras.optimizers.Adam(epsilon=0.01)
# higher learning rate causes dominance of KL loss
optimizer = tf.keras.optimizers.RMSprop(1e-4) 

class Residual(tf.keras.layers.Layer):
    def __init__(self, module, scale):
        super().__init__()

        self.module = module
        self.scale = scale

    def build(self, input_shape):
        self.module.build(input_shape)
        self.scale.build(input_shape)

    def call(self, input):
        return self.scale(input) + self.module(input)

class UpBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()

        self.filters = filters

    def build(self, input_shape):
        self.module = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(
                self.filters, 4, 2, 'same', activation='relu'
            ),
            tf.keras.layers.Conv2D(input_shape[-1], 3, 1, 'same')
        ])
        self.module.build(input_shape)

    def call(self, input):
        return self.module(input)

class DownBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()

        self.filters = filters

    def build(self, input_shape):
        self.module = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.filters, 3, 1, 'same', activation='relu'
            ),
            tf.keras.layers.Conv2D(input_shape[-1], 3, 2, 'same')
        ])
        self.module.build(input_shape)

    def call(self, input):
        return self.module(input)

encoder = tf.keras.Sequential([
    tf.keras.layers.Dense(pixel_size, activation='relu'),
    tf.keras.layers.Dense(pixel_size),
    # 256x256
    Residual(
        DownBlock(pixel_size), 
        tf.keras.layers.AveragePooling2D(2, 2)
    ),
    Residual(
        DownBlock(pixel_size * 4),
        tf.keras.layers.AveragePooling2D(2, 2)
    ),
    Residual(
        DownBlock(pixel_size * 4 * 4),
        tf.keras.layers.AveragePooling2D(2, 2)
    ),
    Residual(
        DownBlock(pixel_size * 4 * 4 * 4),
        tf.keras.layers.AveragePooling2D(2, 2)
    ),
    Residual(
        DownBlock(pixel_size * 4 * 4 * 4 * 4),
        tf.keras.layers.AveragePooling2D(2, 2)
    ),
    # 8x8
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(pixel_size * 4 * 4 * 4 * 4 * 4, activation='relu'),
    tf.keras.layers.Dense(code_size * 2),
])
decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(pixel_size * 4 * 4 * 4 * 4 * 4, activation='relu'),
    tf.keras.layers.Dense(pixel_size * 8 * 8),
    tf.keras.layers.Reshape((8, 8, -1)),
    # 8x8
    Residual(
        UpBlock(pixel_size * 4 * 4 * 4 * 4),
        tf.keras.layers.UpSampling2D(interpolation='bilinear')
    ),
    Residual(
        UpBlock(pixel_size * 4 * 4 * 4),
        tf.keras.layers.UpSampling2D(interpolation='bilinear')
    ),
    Residual(
        UpBlock(pixel_size * 4 * 4),
        tf.keras.layers.UpSampling2D(interpolation='bilinear')
    ),
    Residual(
        UpBlock(pixel_size * 4),
        tf.keras.layers.UpSampling2D(interpolation='bilinear')
    ),
    Residual(
        UpBlock(pixel_size),
        tf.keras.layers.UpSampling2D(interpolation='bilinear'),
    ),
    # 256x256
    tf.keras.layers.Dense(pixel_size, activation='relu'),
    tf.keras.layers.Dense(3),
])

@tf.function
def train_step(source_images, target_images, step):
    target_images = tf.image.random_flip_left_right(target_images)

    with tf.GradientTape() as tape:
        code = encoder(target_images)

        mean, log_scale = tf.split(code, 2, -1)

        scale = tf.exp(log_scale)
        code = tf.random.normal(tf.shape(mean), 0.0, 1.0) * scale + mean

        generated_images = decoder(code)

        generator_loss = tf.keras.losses.MeanSquaredError()(
            generated_images, target_images
        ) + 0.001 * tf.reduce_mean(
            -0.5 * (
                1 + 2 * log_scale - tf.square(mean) - 
                tf.square(scale)
            )
        )

    variables = decoder.trainable_variables + encoder.trainable_variables

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

#example = load_file(
#    "../Datasets/safebooru_r63_256/train/male/" +
#    "00fdb833c64d7824edaa555277e494331b3882891f9422c6bca07611a5193b5f.png"
#)
example = load_file(
    "../Datasets/safebooru_r63_256/test/female/" +
    "00af2f4796bcf58f445ab78e4f8a42f4931c28eec024de0e79872fa019575c5f.png"
)
noise = tf.random.normal((4, code_size))

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
        
        if step % 500 == 0:
            with summary_writer.as_default():
                tf.summary.image(
                    'reconstructed', 
                    tf.clip_by_value(
                        decoder(
                            encoder(example[None])[..., :code_size]
                        ) * 0.5 + 0.5, 
                        0, 1
                    ), 
                    step, 4
                )
                tf.summary.image(
                    'fakes', 
                    tf.clip_by_value(
                        decoder(noise) * 0.5 + 0.5, 
                        0, 1
                    ), 
                    step, 4
                )
finally:
    pass