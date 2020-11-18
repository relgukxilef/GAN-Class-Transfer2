
import datetime, os
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm

size = 256

batch_size = 16

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
    "../Datasets/safebooru_r63_256/test/female/" +
    "00af2f4796bcf58f445ab78e4f8a42f4931c28eec024de0e79872fa019575c5f.png"
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
    def __init__(self, outer, inner):
        super().__init__()
        self.outer = outer
        self.inner = inner

        self.matrix = self.add_weight(
            shape=(inner, inner), 
            initializer=tf.keras.initializers.RandomNormal()
        )
        self.bias = self.add_weight(
            shape=(outer, inner), initializer=tf.keras.initializers.Zeros()
        )

    def call(self, x):
        x = tfp.bijectors.Affine(self.bias)(x)
        
        #x = tfp.bijectors.MatvecLU(
        #    self.matrix, list(reversed(range(self.outputs)))
        #)(x)

        x = tfp.bijectors.Inline(
            lambda x: tf.einsum(
                "ab,...ob->...oa", tf.linalg.expm(self.matrix), x
            ),
            lambda x: tf.einsum(
                "ab,...ob->...oa", tf.linalg.expm(-self.matrix), x
            ),
            lambda x: -tf.linalg.trace(self.matrix),
            lambda x: tf.linalg.trace(self.matrix),
            is_constant_jacobian=True, forward_min_event_ndims=1
        )(x)

        return x
        

class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.block1 = Dense(8 * 8 * 4 * 4 * 3, 8 * 8)
        self.block2 = Dense(8 * 8 * 4 * 4 * 3, 8 * 8)
        self.block3 = Dense(8 * 8 * 8 * 8, 4 * 4 * 3)

    def biject(self, x):
        # x[batch, width, height, channels]
        # organize in blocks
        x = tfp.bijectors.Reshape([8, 8, 4, 8, 8, 4, 3], [size, size, 3])(x)
        x = tfp.bijectors.Transpose([0, 3, 1, 4, 2, 5, 6])(x)
        # x[batch, y1, x1, y2, x2, y3, x3, channels]

        x = tfp.bijectors.Reshape(
            [8 * 8, 8 * 8, 4 * 4 * 3], [8, 8, 8, 8, 4, 4, 3]
        )(x)

        # outer most
        x = tfp.bijectors.Transpose([1, 2, 0])(x)
        x = tfp.bijectors.Reshape(
            [8 * 8 * 4 * 4 * 3, 8 * 8], [8 * 8, 4 * 4 * 3, 8 * 8]
        )(x)

        x = self.block1(x)

        # middle
        x = tfp.bijectors.Reshape(
            [8 * 8, 4 * 4 * 3, 8 * 8], [8 * 8 * 4 * 4 * 3, 8 * 8]
        )(x)
        x = tfp.bijectors.Transpose([1, 2, 0])(x)
        x = tfp.bijectors.Reshape(
            [4 * 4 * 3 * 8 * 8, 8 * 8], [4 * 4 * 3, 8 * 8, 8 * 8]
        )(x)

        x = self.block2(x)

        # inner most
        x = tfp.bijectors.Reshape(
            [4 * 4 * 3, 8 * 8, 8 * 8], [4 * 4 * 3 * 8 * 8, 8 * 8]
        )(x)
        x = tfp.bijectors.Transpose([1, 2, 0])(x)
        x = tfp.bijectors.Reshape(
            [8 * 8 * 8 * 8, 4 * 4 * 3], [8 * 8, 8 * 8, 4 * 4 * 3]
        )(x)

        x = self.block3(x)

        # re-block
        x = tfp.bijectors.Reshape(
            [8, 8, 8, 8, 4, 4, 3], [8 * 8 * 8 * 8, 4 * 4 * 3]
        )(x)
        x = tfp.bijectors.Transpose([0, 2, 4, 1, 3, 5, 6])(x)
        x = tfp.bijectors.Reshape(
            [8 * 8 * 4, 8 * 8 * 4, 3], [8, 8, 4, 8, 8, 4, 3]
        )(x)
        
        return x

    def call(self, example):
        distribution = tfp.distributions.Independent(
            tfp.distributions.Normal(
                tf.zeros_like(example), 1.0
            ), 3
        )
        return self.biject(distribution).log_prob(example) / (size * size)

    def sample(self, size):
        distribution = tfp.distributions.Independent(
            tfp.distributions.Normal(
                tf.zeros(size), 1.0
            ), 3
        )
        return self.biject(distribution).sample()

model = Generator()

def log_sample(epochs, logs):
    with summary_writer.as_default():
        fake = model.sample([4, size, size, 3]) * 0.5 + 0.5
        tf.summary.image('fake', fake, epochs, 4)
        del fake
        code = model.biject(tfp.bijectors.Identity()).inverse(example[0])
        tf.summary.image('code', code[None] * 0.5 + 0.5, epochs)
        del code


name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
summary_writer = tf.summary.create_file_writer(os.path.join("logs", name))

log_sample(0, None)

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