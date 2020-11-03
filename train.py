
import datetime, os
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm

filters = 16
kernel_size = 31
size = 256

batch_size = 1

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)


def load_file(file, crop=True):
    image = tf.image.decode_jpeg(tf.io.read_file(file))[:, :, :3]
    if crop:
        image = tf.image.random_crop(image, [size, size, 3])
    return (tf.cast(image, tf.float32) / 128 - 1, 0)

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
    dataset = dataset.repeat()
    dataset = dataset.map(load_file).batch(batch_size).prefetch(8)
    datasets += [dataset]

def lift(bijector, split, event_ndims):
    f = lambda x: tfp.bijectors.Blockwise(
        [
            bijector(x[..., split:]),
            tfp.bijectors.Identity(),
        ], [split, tf.shape(x)[-1] - split]
    )
    return tfp.bijectors.Inline(
        lambda x: f(x).forward(x),
        lambda x: f(x).inverse(x),
        lambda x: f(x).inverse_log_det_jacobian(x, event_ndims),
        lambda x: f(x).forward_log_det_jacobian(x, event_ndims),
        forward_min_event_ndims = event_ndims
    )
    
def lu_initializer(shape, dtype=None):
    matrix = tf.keras.initializers.Orthogonal()(shape)
    lower_upper, permutation = tf.linalg.lu(matrix)
    del matrix, permutation
    return lower_upper

def space_to_depth(event_shape, block_size):
    return tfp.bijectors.Chain([
        tfp.bijectors.Reshape([
            event_shape[0] // block_size[0], block_size[0],
            event_shape[1] // block_size[1], block_size[1],
            event_shape[2]
        ], event_shape),
        tfp.bijectors.Transpose([0, 2, 1, 3, 4]),
        tfp.bijectors.Reshape([
            event_shape[0] // block_size[0], event_shape[1] // block_size[1],
            block_size[0] * block_size[1] * event_shape[2]
        ], [
            event_shape[0] // block_size[0], event_shape[1] // block_size[1],
            block_size[0], block_size[1], event_shape[2]
        ]),
    ])

def depth_to_space(event_shape, block_size):
    return tfp.bijectors.Invert(space_to_depth(
        [
            event_shape[0] * block_size[0], event_shape[1] * block_size[1], 
            event_shape[2] // block_size[0] // block_size[1]
        ], block_size
    ))
   
class Dense(tf.keras.layers.Layer):
    def __init__(self, outputs):
        super().__init__()

        self.matrix = self.add_weight(
            shape=(outputs, outputs), 
            initializer=lu_initializer
        )
        self.bias = self.add_weight(
            shape=(outputs,), initializer=tf.keras.initializers.Zeros()
        )

    def bijector(self):
        return tfp.bijectors.Chain([
            tfp.bijectors.Affine(self.bias),
            tfp.bijectors.MatvecLU(
                self.matrix, [1, 2, 0]
            )
        ])

class Block(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.convolution = tf.keras.layers.Conv2D(
            filters, kernel_size, 1, 'same'
        )
        self.dense = tf.keras.layers.Dense(1)

        self.bijective_dense = Dense(3)

        self.batch_norm = tfp.bijectors.BatchNormalization()

    def nonlinearity(self, input):
        shift = self.dense(tf.nn.relu(self.convolution(input)))
        return tfp.bijectors.Affine(shift)

    def bijector(self):
        return tfp.bijectors.Chain([
            lift(lambda x: self.nonlinearity(x), 1, 3),
            self.bijective_dense.bijector(),
            self.batch_norm,
            #tfp.bijectors.Permute([1, 2, 0]),
        ])


class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.blocks = [Block() for _ in range(8)]

    def bijector(self):
        return tfp.bijectors.Chain([
            #space_to_depth([size, size, 3], [4, 4]),
        ] + [b.bijector() for b in self.blocks] + [
            #depth_to_space([size//4, size//4, 3*4*4], [4, 4]),
        ])

    def call(self, example):
        distribution = tfp.distributions.Independent(
            tfp.distributions.Normal(
                tf.zeros_like(example), 1.0
            ), 3
        )
        return tfp.distributions.TransformedDistribution(
            distribution, self.bijector()
        ).log_prob(example) / (size * size)

    def sample(self, size):
        distribution = tfp.distributions.Independent(
            tfp.distributions.Normal(
                tf.zeros(size), 1.0
            ), 3
        )
        return tfp.distributions.TransformedDistribution(
            distribution, self.bijector()
        ).sample()

model = Generator()

def log_sample(epochs, logs):
    with summary_writer.as_default():
        fake = model.sample([4, size, size, 3]) * 0.5 + 0.5
        tf.summary.image('fake', fake, epochs, 4)
        del fake


name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
summary_writer = tf.summary.create_file_writer(os.path.join("logs", name))

model.compile(
    tf.keras.optimizers.Adam(),#.SGD(1e-3), 
    lambda y_true, log_prob: -tf.reduce_mean(log_prob)
)

model.fit(
    datasets[1], steps_per_epoch=1000, epochs=100,
    callbacks=[
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=log_sample
        )
    ]
)