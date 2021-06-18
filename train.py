
import datetime, os
import tensorflow as tf

filters = 512
color_size = 4
size = 128

batch_size = 1
steps_per_epoch = 4000

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

def categorical_sample(logits):
    return tf.reshape(tf.random.categorical(
        tf.reshape(logits, [-1, tf.shape(logits)[-1]]), 
        1, tf.int32
    ), tf.shape(logits)[:-1])

class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.length = size * size
        self.kernel = tf.Variable(
            initial_value=tf.random.normal([3, 3, filters, self.length]),
            trainable=True,
        )
        self.bias = tf.Variable(
            initial_value=tf.zeros([filters]),
            trainable=True,
        )
        self.dense_final = tf.keras.layers.Dense(
            color_size, kernel_initializer='zeros'
        )

        # [channel, k, 1, space]
        self.mask = [
            [[[0]], [[0]], [[0]]], [[[1]], [[0]], [[0]]], [[[1]], [[1]], [[0]]]
        ]
        self.mask = tf.concat(
            [self.mask, tf.ones([3, 3, 1, self.length - 1])], -1
        )

    @tf.function
    def call(self, example):
        example = tf.cast(example, tf.float32) / color_size * 2 - 1

        shape = tf.shape(example)

        x = tf.reshape(
            example, 
            tf.concat([shape[:-3], [self.length, 3]], 0)
        )
        # x[batch, height * width, channel]

        # FFT only works on inner-most axis
        x = tf.transpose(x, [0, 2, 1])

        # x[batch, channel, height * width]
        # kernel[channel, iteration, feature, height * width]

        kernel = self.kernel * self.mask

        x = tf.signal.rfft(x, [self.length * 2])
        kernel = tf.signal.rfft(kernel, [self.length * 2])

        x = tf.einsum('...cs,cifs->...ifs', x, kernel)

        x = tf.signal.irfft(x, [self.length * 2])[..., :self.length]
        
        # [batch, iteration, feature, space]

        x = tf.transpose(x, [0, 3, 1, 2])
        
        # [batch, space, iteration, feature]

        x += self.bias

        x = tf.nn.relu(x)
        
        x = self.dense_final(x)

        # TODO: try predicting the image differential instead
        x = tf.reshape(
            x, tf.concat([shape, [color_size]], 0)
        )

        return tf.cast(x, dtype=tf.float32)

    @tf.function
    def sample(self):
        kernel = self.kernel * self.mask
        kernel = kernel[..., ::-1] # fft interprets kernel mirrored

        @tf.function
        def value(buffer):
            # buffer[batch, channel, space]
            # kernel[channel, iteration, feature, space]
            pre_features = tf.einsum(
                '...cs,cifs->...if', buffer[..., 1:], kernel[..., :-1]
            )

            pre_features += self.bias

            # pixel[batch, channel, space]
            pixel = tf.zeros(buffer[..., 0:0, 0:1].shape)

            for i in range(3):
                x = pre_features[..., i:i+1, :]

                if i > 0:
                    x += tf.einsum(
                        '...cs,cifs->...if', pixel, kernel[:i, i:i+1, :, -1:]
                    )

                x = tf.nn.relu(x)

                x = self.dense_final(x)

                x = (
                    tf.cast(categorical_sample(x), tf.float32) / 
                    color_size * 2 - 1
                )[..., None]

                pixel = tf.concat([pixel, x], -2) # channel dimension

            x = tf.concat([buffer[..., :, 1:], pixel], -1) # space dimension

            return (x,)

        fake = tf.while_loop(
            lambda x: True, value, (tf.zeros([2, 3, self.length]),), 
            maximum_iterations=self.length
        )[0] * 0.5 + 0.5

        fake = tf.transpose(fake, [0, 2, 1])

        x = tf.reshape(fake, [-1, size, size, 3])

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
    return (image, image)

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
    prediction = model(example[0][None])
    fake = model.sample() * color_size / (color_size - 1)

    prediction = tf.cast(
        categorical_sample(prediction), tf.float32
    ) / (color_size - 1)

    with summary_writer.as_default():
        tf.summary.image(
            'fake', fake, epochs + 1, 4
        )
        tf.summary.image(
            'prediction', prediction, epochs + 1, 4
        )
    del fake, prediction

#loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
#    example[None], prediction
#)
log_sample(-1, None)


model.compile(
    tf.keras.optimizers.Adam(1e-4),
    #tf.keras.mixed_precision.LossScaleOptimizer(
    #    tf.keras.optimizers.Adam(1e-4)
    #),
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