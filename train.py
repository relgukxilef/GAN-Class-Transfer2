
import datetime, os, math
import tensorflow as tf

filters = 256
kernel_size = 3
size = 256

batch_size = 2

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

@tf.function
def decode_file(file, crop=True):
    image = tf.image.decode_jpeg(file)[:, :, :3]
    if crop:
        image = tf.image.random_crop(image, [size, size, 3])
    return (tf.cast(image, tf.float32) / 128 - 1, 0)

@tf.function
def load_file(file, crop=True):
    return decode_file(tf.io.read_file(file), crop)

classes = [
    "../Datasets/safebooru_r63_256/train/male/*",
    "../Datasets/safebooru_r63_256/train/female/*"
]

datasets = []

for i, folder in enumerate(classes):
    dataset = tf.data.Dataset.list_files(folder)
    dataset = dataset.map(tf.io.read_file)
    dataset = dataset.cache('cache' + str(i))
    dataset = dataset.repeat()
    dataset = dataset.map(decode_file).batch(batch_size).prefetch(8)
    datasets += [dataset]



def matrix_exponential_initializer(shape, dtype=tf.float32):
    return tf.cast(tf.linalg.logm(
        tf.cast(tf.keras.initializers.Orthogonal()(shape), tf.complex64)
    ), dtype)
   
class Dense(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.matrix = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]), 
            initializer=tf.keras.initializers.Zeros()
        )
        self.bias = self.add_weight(
            shape=(input_shape[-1],), initializer=tf.keras.initializers.Zeros()
        )

    def call(self, x, inverse=False):
        if not inverse:
            self.add_loss(-tf.linalg.trace(self.matrix) / self.matrix.shape[0])
            return tf.linalg.matvec(
                tf.linalg.expm(self.matrix), x + self.bias
            )
        else:
            return tf.linalg.matvec(
                tf.linalg.expm(-self.matrix), x
            ) - self.bias

    def inverse(self, y):
        return self(y, True)
        
class Lift(tf.keras.layers.Layer):
    def __init__(self, predict, update):
        super().__init__()

        self.predict = predict
        self.update = update

    def build(self, input_shape):
        self.predict.build(input_shape[:-1] + input_shape[-1] // 2)
        self.update.build(input_shape[:-1] + input_shape[-1] // 2)

    def call(self, x, inverse=False):
        if not inverse:
            a, b = tf.split(x, 2, -1)
            a += self.predict(b)
            b += self.update(a)
            return tf.roll(tf.concat([a, b], -1), 1, -1)
        else:
            a, b = tf.split(tf.roll(x, -1, -1), 2, -1)
            b -= self.update(a)
            a -= self.predict(b)
            return tf.concat([a, b], -1)

    def inverse(self, y):
        return self(y, True)

class DiagonalDense(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.log_scale = self.add_weight(
            shape=(input_shape[-1],), 
            initializer=tf.keras.initializers.Zeros()
        )
        self.bias = self.add_weight(
            shape=(input_shape[-1],), initializer=tf.keras.initializers.Zeros()
        )

    def call(self, x, inverse=False):
        if not inverse:
            self.add_loss(-tf.reduce_mean(self.log_scale))
            return tf.exp(self.log_scale) * (x + self.bias)
        else:
            return tf.exp(-self.log_scale) * x - self.bias

    def inverse(self, y):
        return self(y, True)

class Reshape(tf.keras.layers.Layer):
    def __init__(self, target_shape, inverse_shape):
        super().__init__()
        
        self.forward = tf.keras.layers.Reshape(target_shape)
        self.backward = tf.keras.layers.Reshape(inverse_shape)

    def call(self, x, inverse=False):
        if not inverse:
            return self.forward(x)
        else:
            return self.backward(x)

class DownShuffle(tf.keras.layers.Layer):
    def __init__(self, block_size=2):
        super().__init__()
        
        self.block_size = block_size

    def call(self, x, inverse=False):
        if not inverse:
            return tf.nn.space_to_depth(x, self.block_size)
        else:
            return tf.nn.depth_to_space(x, self.block_size)

class ReLU(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return tf.math.maximum(x, 0)

class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.layer_list = []

        factor = 1
        for block in range(6):
            factor *= 2
            self.layer_list += [DownShuffle(),]
            for i in range(2):
                if block < 4:
                    self.layer_list += [Dense()]
                self.layer_list += [
                    Lift(
                        tf.keras.Sequential([
                            tf.keras.layers.Conv2D(
                                filters, kernel_size, 1, 'same'
                            ),
                            ReLU(),
                            tf.keras.layers.Conv2D(
                                3 * factor * factor // 2, kernel_size, 1, 
                                'same', kernel_initializer='zeros'
                            ),
                        ]),
                        tf.keras.Sequential([
                            tf.keras.layers.Conv2D(
                                filters, kernel_size, 1, 'same'
                            ),
                            ReLU(),
                            tf.keras.layers.Conv2D(
                                3 * factor * factor // 2, kernel_size, 1, 
                                'same', kernel_initializer='zeros'
                            ),
                        ]),
                    )
                ]

        self.layer_list += [ 
            Reshape(
                [size * size * 3], 
                [size // factor, size // factor, 3 * factor * factor]
            ),
            DiagonalDense(),
        ]

    def call(self, x, inverse=False):
        if not inverse:
            for layer in self.layer_list:
                x = layer(x)
        else:
            for layer in reversed(self.layer_list):
                x = layer(x, inverse=True)
        return x

    def inverse(self, y):
        return self(y, True)

@tf.function
def normal_negative_log_probability(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred)) / 2 + math.log(2 * math.pi) / 2


def log_sample(epochs, logs):
    example = tf.random.normal([4, size * size * 3])
    with summary_writer.as_default():
        fake = model.inverse(example) * 0.5 + 0.5
        tf.summary.image('fake', fake, epochs, 4)
        del fake


if __name__ == '__main__':
    model = Generator()

    name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    summary_writer = tf.summary.create_file_writer(os.path.join("logs", name))

    prediction = model(next(iter(datasets[1]))[0])

    model.compile(
        tf.keras.optimizers.RMSprop(0.00001), 
        #tf.keras.optimizers.SGD(0.1), 
        normal_negative_log_probability
    )

    model.fit(
        datasets[1], steps_per_epoch=1000, epochs=100,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_epoch_begin=log_sample
            )
        ]
    )