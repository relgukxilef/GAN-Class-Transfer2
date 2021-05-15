
import datetime, os
import tensorflow as tf

import tensorflow.keras.mixed_precision.experimental

filters = 512
color_size = 4
size = 128
layers = 1

batch_size = 8
steps_per_epoch = 10

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

tensorflow.keras.mixed_precision.experimental.set_policy(
    tensorflow.keras.mixed_precision.experimental.Policy('mixed_float16')
)

def mean_and_difference_distribution(parameters):
    # parameters[batch, height, width, channel, parameter]
    mean = parameters[..., :1]
    differences = tf.math.softplus(parameters[..., 1:])
    #differences = tf.nn.relu(parameters[..., 1:])
    offsets = tf.math.cumsum(differences, -1)
    mean_offset = tf.reduce_mean(offsets, -1, keepdims=True)
    return tf.concat([mean, mean + offsets], -1) - mean_offset

def categorical_sample(logits):
    return tf.reshape(tf.random.categorical(
        tf.reshape(logits, [-1, tf.shape(logits)[-1]]), 
        1
    ), tf.shape(logits)[:-1])

class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.length = size * size * 3
        self.dense_in = tf.keras.layers.Conv1D(filters, self.length)
        self.dense_final = tf.keras.layers.Dense(color_size)


    @tf.function
    def call(self, example):
        example = tf.cast(example, tf.float32) / color_size * 2 - 1

        shape = tf.shape(example)

        x = tf.reshape(
            example, 
            tf.concat([shape[:-3], [self.length, 1]], 0)
        )
        # x[batch, height * width * channel, 1]

        x = tf.pad(x, [[0, 0], [self.length, 0], [0, 0]])[:, :-1, :]

        x = self.dense_in(x)
        # TODO: try this with FFT

        x = tf.nn.relu(x)
        
        x = self.dense_final(x)

        x = tf.reshape(
            x, tf.concat([shape, [color_size]], 0)
        )

        return tf.cast(x, dtype=tf.float32)

    @tf.function
    def sample(self):
        @tf.function
        def value(buffer):
            x = buffer

            x = x * 2 - 1

            x = self.dense_in(x)

            x = tf.nn.relu(x)

            x = self.dense_final(x)

            x = (
                tf.cast(categorical_sample(x), tf.float16)[..., None] / 
                color_size
            )

            x = tf.concat([buffer[..., 1:, :], x], -2)

            return (x,)

        fake = tf.while_loop(
            lambda x: True, value, (tf.zeros([1, self.length, 1], tf.float16),), 
            maximum_iterations=self.length
        )

        x = tf.reshape(fake, [size, size, 3])

        return x

@tf.function
def load_file(path):
    return tf.io.read_file(path)

@tf.function
def prepare_example(file, crop=True):
    image = tf.image.decode_jpeg(file)[:, :, :3]
    image = tf.cast(image, tf.int32) * color_size // 256
    if crop:
        image = tf.image.random_crop(image, [size, size, 3])
    shape = tf.shape(image)
    y = tf.reshape(
        image, 
        tf.concat([shape[:-3], [tf.reduce_prod(shape[-3:])]], 0)
    )
    y = tf.pad(y[..., :-1], [[1, 0]])
    y = tf.reshape(y, shape)
    return (y, image)

classes = [
    "../Datasets/safebooru_r63_256/train/male/*",
    "../Datasets/safebooru_r63_256/train/female/*"
]

datasets = []

example = prepare_example(load_file(
    "../Datasets/safebooru_r63_256/test/female/" +
    "00af2f4796bcf58f445ab78e4f8a42f4931c28eec024de0e79872fa019575c5f.png"
))

for i, folder in enumerate(classes):
    dataset = tf.data.Dataset.list_files(folder)
    dataset = dataset.map(load_file).repeat()
    dataset = dataset.map(prepare_example).batch(batch_size).prefetch(16)
    datasets += [dataset]

name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_folder = os.path.join("logs", name)
os.makedirs(log_folder)
summary_writer = tf.summary.create_file_writer(log_folder)


model = Generator()

def log_sample(epochs, logs):
    fake = model.sample()[None]

    prediction = model(example[0][None])
    prediction = tf.cast(
        categorical_sample(prediction), tf.float32
    ) / color_size

    with summary_writer.as_default():
        tf.summary.image(
            'fake', fake, epochs + 1, 4
        )
        tf.summary.image(
            'prediction', prediction, epochs + 1, 4
        )
    del fake, prediction

prediction = model(example[0][None])
#loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
#    example[None], prediction
#)
fake = model.sample()


model.compile(
    tf.keras.mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(1e-4)
    ),
    #tf.keras.optimizers.SGD(0.1, 0.9, True),
    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    []
)

model.fit(
    datasets[1], steps_per_epoch=steps_per_epoch, epochs=100,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(log_folder, "model.{epoch:02d}.hdf5")
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_folder, write_graph=True, write_images=False, 
            profile_batch=0
        ),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=log_sample
        )
    ]
)