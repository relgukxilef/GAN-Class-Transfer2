
import datetime, os
import tensorflow as tf
import tqdm

filters = 16
kernel_size = 63

batch_size = 8

generator = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        filters, (kernel_size, 1), padding='same', use_bias=False
    ),
    #tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters, (1, kernel_size), padding='same'),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(3, (1, 1), padding='same'),
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        filters, (kernel_size, 1), padding='same', use_bias=False
    ),
    #tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters, (1, kernel_size), padding='same'),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(filters),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(1),
])

optimizer = tf.keras.optimizers.SGD(0.01)
#optimizer = tf.keras.optimizers.Adam()

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

def load_file(file, crop=True):
    image = tf.image.decode_jpeg(tf.io.read_file(file))[:, :, :3]
    if crop:
        image = tf.image.random_crop(image, [256, 256, 3])
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
                tf.clip_by_value(generator(example[None]) * 0.5 + 0.5, 0, 1), 
                step, 1
            )
    #progress.set_description("Loss: {}".format(5))