
import datetime, os
import tensorflow as tf
import tqdm

size = 256
filters = 512
kernel_size = 5
depth = 5
block_size = 16

batch_size = 1
optimizer = tf.keras.optimizers.RMSprop(1e-4)

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.convolutions = [
            tf.keras.layers.Conv2D(
                filters, kernel_size, 1, 'same', activation='relu'
            ) 
            for _ in range(depth)
        ]
        self.denses = [
            tf.keras.layers.Dense(
                3 * block_size * block_size, 
                kernel_initializer='zeros'
            )
            for _ in range(depth)
        ]

    def call(self, image):
        image = tf.nn.space_to_depth(image, block_size)
        for c, d in zip(self.convolutions, self.denses):
            image += d(c(image))
        image = tf.nn.depth_to_space(image, block_size)
        return image

    def activations(self, image):
        images = []
        image = tf.nn.space_to_depth(image, block_size)
        for i, (c, d) in enumerate(zip(self.convolutions, self.denses)):
            image += d(c(image))
            images += [tf.nn.depth_to_space(image, block_size)]
        return images

class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.convolutions = [
            tf.keras.layers.Conv2D(
                filters, kernel_size, 1, 'same', activation='relu'
            ) 
            for _ in range(depth)
        ]
        self.denses = [
            tf.keras.layers.Dense(
                3 * block_size * block_size, kernel_initializer='zeros'
            ) 
            for _ in range(depth)
        ]
        self.dense = tf.keras.layers.Dense(1)

    def call(self, image):
        image = tf.nn.space_to_depth(image, block_size)
        for c, d in zip(self.convolutions, self.denses):
            image += d(c(image))
        return self.dense(tf.keras.layers.GlobalAveragePooling2D()(image))

    def activations(self, image):
        images = []
        image = tf.nn.space_to_depth(image, block_size)
        for c, d in zip(self.convolutions, self.denses):
            image += d(c(image))
            images += [tf.nn.depth_to_space(image, block_size)]
        return images
            

generator = Generator()
discriminator = Discriminator()

@tf.function
def train_step(source_images, target_images, step):
    with tf.GradientTape(persistent=True) as tape:
        generated_images = generator(source_images)

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

    accuracy = 0.5 * (
        tf.reduce_mean(tf.cast(tf.greater(real_output, 0), tf.float32)) +
        tf.reduce_mean(tf.cast(tf.less(fake_output, 0), tf.float32))
    )

    return accuracy

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
    "00fdb833c64d7824edaa555277e494331b3882891f9422c6bca07611a5193b5f.png", 
    False
)[:362//block_size*block_size, :, :]

for folder in classes:
    dataset = tf.data.Dataset.list_files(folder)
    dataset = dataset.shuffle(1000).repeat()
    dataset = dataset.map(load_file).batch(batch_size).prefetch(8)
    datasets += [dataset]

name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(os.path.join("logs", name))

def scale_and_clip(image):
    return tf.clip_by_value(image * 0.5 + 0.5, 0, 1)

accuracy = 0
progress = tqdm.tqdm(tf.data.Dataset.zip(tuple(datasets)))
for step, (image_x, image_y) in enumerate(progress):
    accuracy += train_step(image_x, image_y, tf.constant(step, tf.int64))
    
    if step % 200 == 0:
        with summary_writer.as_default():
            #tf.summary.scalar('generator_loss', generator_loss, step)
            #tf.summary.scalar('discriminator_loss', discriminator_loss, step)
            tf.summary.scalar('discriminator_accuracy', accuracy / 200, step)
            tf.summary.image(
                'fakes', 
                scale_and_clip(generator(example[None])), 
                step, 1
            )
            for name, model in [("g", generator), ("d", discriminator)]:
                tf.summary.image(
                    name + "_activation", 
                    scale_and_clip(
                        tf.concat(model.activations(example[None]), 0)
                    ), 
                    step, 8
                )
        accuracy = 0