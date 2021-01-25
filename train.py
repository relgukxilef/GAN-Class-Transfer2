
import datetime, os
import tensorflow as tf
import tqdm

size = 256
filters = 1024

batch_size = 1

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

class Dense2DBias(tf.keras.layers.Layer):
    def __init__(self, units, size):
        super().__init__()

        self.dense = tf.keras.layers.Dense(units, use_bias=False)
        self.size = size
        self.bias = self.add_weight(
            "bias",
            shape=[size, size, units]
        )

    def build(self, input_shape):
        self.dense.build(input_shape)

    def call(self, input):
        return self.dense(input) + self.bias

class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.dense0 = tf.keras.layers.Dense(filters, use_bias=False)
        self.dense1 = tf.keras.layers.Dense(filters, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, kernel_initializer='zeros')
        self.encoding = self.add_weight(
            "encoding",
            shape=[size, size, filters]
        )

    def call(self, image):
        x = tf.nn.relu(self.encoding + self.dense0(image))
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = self.dense1(x)
        return self.dense2(x)
            

optimizer = tf.keras.optimizers.Adam()
#optimizer = tf.keras.optimizers.RMSprop(1e-5)

encoder = tf.keras.Sequential([
    Dense2DBias(filters, size),
    tf.keras.layers.ReLU(),
    Dense2DBias(filters, size),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(filters, activation='relu'),
    tf.keras.layers.Dense(filters)
])
decoder = tf.keras.Sequential([
    tf.keras.layers.Reshape((1, 1, -1)),
    Dense2DBias(filters, size),
    tf.keras.layers.ReLU(),
    Dense2DBias(filters, size),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(3)
])
generator = tf.keras.Sequential([encoder, decoder])
discriminator = Discriminator()

@tf.function
def train_step(source_images, target_images, step):
    target_images = tf.image.random_flip_left_right(target_images)

    with tf.GradientTape(persistent=True) as tape:
        #generated_images = decoder(tf.random.normal([batch_size, filters]))
        generated_images = generator(target_images)

        generator_loss = tf.keras.losses.MeanSquaredError()(
            generated_images, target_images
        )

        #real_output = discriminator(target_images)
        #fake_output = discriminator(generated_images)

        #generator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        #    tf.ones_like(fake_output), fake_output
        #)
        #generator_loss = tf.keras.losses.MeanSquaredError()(0, fake_output)

        #discriminator_loss = 0.5 * sum([
        #    tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        #        tf.ones_like(real_output), real_output
        #    ),
        #    tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        #        tf.zeros_like(fake_output), fake_output
        #    )
        #])
        #discriminator_loss = 0.5 * (
        #    tf.keras.losses.MeanSquaredError()(1, real_output) +
        #    tf.keras.losses.MeanSquaredError()(-1, fake_output)
        #)

    optimizer.apply_gradients(zip(
        tape.gradient(generator_loss, generator.trainable_variables), 
        generator.trainable_variables
    ))
    #optimizer.apply_gradients(zip(
    #    tape.gradient(discriminator_loss, discriminator.trainable_variables), 
    #    discriminator.trainable_variables
    #))

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
    
    if step % 500 == 0:
        with summary_writer.as_default():
            #tf.summary.scalar('generator_loss', generator_loss, step)
            #tf.summary.scalar('discriminator_loss', discriminator_loss, step)
            tf.summary.image(
                'fakes', 
                tf.clip_by_value(
                    generator(
                        example[None]
                    ) * 0.5 + 0.5, 
                    0, 1
                ), 
                step, 4
            )
            for i, model in enumerate([encoder.layers[0].bias]):
                encoding = get_encoding_summary(model)
                tf.summary.image(
                    'encoding/' + str(i), encoding, step, 8
                )