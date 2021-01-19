
import datetime, os
import tensorflow as tf
import tqdm

size = 256
filters = 512

batch_size = 8

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.dense0 = tf.keras.layers.Dense(filters, use_bias=False)
        self.dense1 = tf.keras.layers.Dense(3)
        self.encoding = self.add_weight(
            "encoding",
            shape=[size, size, filters]
        )

    def call(self, latent):
        return self.dense1(tf.nn.relu(self.encoding + self.dense0(latent)))

class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.dense0 = tf.keras.layers.Dense(filters, use_bias=False)
        self.dense1 = tf.keras.layers.Dense(filters)
        self.dense2 = tf.keras.layers.Dense(1)
        self.encoding = self.add_weight(
            "encoding",
            shape=[size, size, filters]
        )

    def call(self, image):
        x = tf.nn.relu(self.encoding + self.dense0(image))
        x = tf.reduce_mean(x, [-2, -3], keepdims=True)
        x = tf.nn.relu(self.dense1(x))

        return self.dense2(x)
            

optimizer = tf.keras.optimizers.RMSprop()

generator = Generator()
discriminator = Discriminator()

@tf.function
def train_step(source_images, target_images, step):
    with tf.GradientTape(persistent=True) as tape:
        generated_images = generator(tf.random.normal([1, 1, filters]))

        real_output = discriminator(target_images)
        fake_output = discriminator(generated_images)

        generator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
            tf.ones_like(fake_output), fake_output
        )

        discriminator_loss = sum([
            tf.keras.losses.BinaryCrossentropy(from_logits=True)(
                tf.ones_like(real_output), real_output
            ),
            tf.keras.losses.BinaryCrossentropy(from_logits=True)(
                tf.zeros_like(fake_output), fake_output
            )
        ])

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

def get_encoding_summary(model):
    return tf.clip_by_value(
        tf.transpose(
            model.encoding[..., :8], [2, 0, 1]
        )[..., None] * 0.5 + 0.5, 
        0, 1
    )

classes = [
    "../Datasets/safebooru_r63_256/train/male/*",
    "../Datasets/safebooru_r63_256/train/female/*"
]

datasets = []

example = load_file(
    "../Datasets/safebooru_r63_256/train/male/" +
    "00fdb833c64d7824edaa555277e494331b3882891f9422c6bca07611a5193b5f.png", 
    False
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
    
    if step % 200 == 0:
        with summary_writer.as_default():
            #tf.summary.scalar('generator_loss', generator_loss, step)
            #tf.summary.scalar('discriminator_loss', discriminator_loss, step)
            tf.summary.image(
                'fakes', 
                tf.clip_by_value(
                    generator(
                        tf.random.normal([4, 1, 1, filters])
                    ) * 0.5 + 0.5, 
                    0, 1
                ), 
                step, 4
            )
            for i, model in enumerate([generator, discriminator]):
                encoding = get_encoding_summary(model)
                tf.summary.image(
                    'encoding/' + str(i), encoding, step, 8
                )