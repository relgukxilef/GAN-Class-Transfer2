
import datetime, os
import tensorflow as tf
import tensorflow_probability as tfp

filters = 256
kernel_size = 3
size = 256

batch_size = 8

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

def decode_file(file, crop=True):
    image = tf.image.decode_jpeg(file)[:, :, :3]
    if crop:
        image = tf.image.random_crop(image, [size, size, 3])
    return (tf.cast(image, tf.float32) / 128 - 1, 0)

def load_file(file, crop=True):
    return decode_file(tf.io.read_file(file), crop)

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

for i, folder in enumerate(classes):
    dataset = tf.data.Dataset.list_files(folder)
    dataset = dataset.map(tf.io.read_file).cache('cache' + str(i))
    dataset = dataset.repeat()
    dataset = dataset.map(decode_file).batch(batch_size).prefetch(8)
    datasets += [dataset]

class LogProb(tf.keras.Model):
    def __init__(self, distribution):
        super().__init__()
        self.distribution = distribution

    def call(self, sample):
        return self.distribution.log_prob(sample)

def depth_to_space(block_size=2):
    return tfp.bijectors.Inline(
        forward_fn=lambda x: tf.nn.depth_to_space(x, block_size),
        inverse_fn=lambda x: tf.nn.space_to_depth(x, block_size),
        forward_log_det_jacobian_fn=lambda x: tf.zeros([]),
        inverse_log_det_jacobian_fn=lambda x: tf.zeros([]),
        forward_event_shape_fn=lambda shape: tf.TensorShape(
            shape[:-3] + [shape[-3] * block_size] + [shape[-2] * block_size] + 
            [shape[-1] // block_size // block_size]
        ),
        inverse_event_shape_fn=lambda shape: tf.TensorShape(
            shape[:-3] + [shape[-3] // block_size] + [shape[-2] // block_size] + 
            [shape[-1] * block_size * block_size]
        ),
        forward_event_shape_tensor_fn=lambda shape: tf.concat([shape[:-3], [
            shape[-3] * block_size, shape[-2] * block_size, 
            shape[-1] // block_size // block_size
        ]], 0),
        inverse_event_shape_tensor_fn=lambda shape: tf.concat([shape[:-3], [
            shape[-3] // block_size, shape[-2] // block_size, 
            shape[-1] * block_size * block_size
        ]], 0),
        is_constant_jacobian=True,
        forward_min_event_ndims=3,
        inverse_min_event_ndims=3
    )

def matrix_exponential_initializer(shape, dtype=tf.float32):
    return tf.cast(tf.linalg.logm(
        tf.cast(tf.keras.initializers.Orthogonal()(shape), tf.complex64)
    ), dtype)
   
class _Dense(tf.keras.layers.Layer):
    def __init__(self, outputs):
        super().__init__()
        self.matrix = self.add_weight(
            shape=(outputs, outputs), 
            initializer=matrix_exponential_initializer
        )
        self.bias = self.add_weight(
            shape=(outputs,), initializer=tf.keras.initializers.Zeros()
        )

class Dense(tfp.bijectors.Inline):
    def __init__(self, outputs):
        _dense = _Dense(outputs)

        super().__init__(
            lambda x: 
                tf.linalg.matvec(tf.linalg.expm(_dense.matrix), x) + 
                _dense.bias,
            lambda x: 
                tf.linalg.matvec(
                    tf.linalg.expm(-_dense.matrix), x - _dense.bias
                ),
            lambda x: tf.exp(-tf.linalg.trace(_dense.matrix)),
            lambda x: tf.exp(tf.linalg.trace(_dense.matrix)),
            is_constant_jacobian=True,
            forward_min_event_ndims=1
        )

class Block(tfp.bijectors.Chain):
    def __init__(self, size):
        lift_convolution = tf.keras.layers.Conv2D(
            filters, kernel_size, 1, 'same', activation='relu'
        )
        lift_dense = tf.keras.layers.Dense(
            size // 2, kernel_initializer='zeros'
        )

        def shift_and_log_scale(x, outputs):
            shift = lift_dense(lift_convolution(x))
            return shift, None

        super().__init__([
            #tfp.bijectors.BatchNormalization(),
            Dense(size), 
            tfp.bijectors.RealNVP(
                size // 2, shift_and_log_scale_fn=shift_and_log_scale,
                is_constant_jacobian=True
            )
        ])


class Generator(tfp.bijectors.Chain):
    def __init__(self):
        # Beware: Chain([a, b])(x) means a(b(x))
        super().__init__(
            sum([
                [
                    depth_to_space(),
                    tfp.bijectors.Blockwise(
                        [
                            tfp.bijectors.Chain([Block(3 * 2 * int(2**i))] * 1), 
                            tfp.bijectors.Identity()
                        ], [3 * 2 * int(2**i), 3 * (int(4**i) - 2 * int(2**i))]
                    )
                ]
                for i in range(1, 8)
            ], []) + [
                tfp.bijectors.Reshape(
                    [2, 2, 3 * 128 * 128],
                    [256 * 256 * 3]
                )
            ]
        )


test = Dense(4)

test([1.0, 2, 3, 4])

model = Generator()
source_distribution = tfp.distributions.Sample(
    tfp.distributions.Normal(0., 1.), (size * size * 3,)
)
distribution = tfp.distributions.TransformedDistribution(
    source_distribution, model
)

example = source_distribution.sample([4])

def log_sample(epochs, logs):
    with summary_writer.as_default():
        fake = model(example) * 0.5 + 0.5
        tf.summary.image('fake', fake, epochs, 4)
        del fake


name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
summary_writer = tf.summary.create_file_writer(os.path.join("logs", name))

log_probability_model = LogProb(distribution)

log_probability_model(next(iter(datasets[1]))[0])

log_probability_model.compile(
    tf.keras.optimizers.RMSprop(), 
    #tf.keras.optimizers.SGD(1.0), 
    lambda y_true, log_prob: -tf.reduce_mean(log_prob / size / size)
)

log_probability_model.fit(
    datasets[1], steps_per_epoch=1000, epochs=100,
    callbacks=[
        tf.keras.callbacks.LambdaCallback(
            on_epoch_begin=log_sample
        )
    ]
)