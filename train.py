
import datetime, os
import tensorflow as tf
import tensorflow_probability as tfp


size = 32

batch_size = 1

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

class LogProb(tf.keras.Model):
    def __init__(self, distribution):
        super().__init__()
        self.distribution = distribution

    def call(self, sample):
        return self.distribution.log_prob(sample)


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

model = tfp.bijectors.Glow(
    (size, size, 3), 3, coupling_bijector_fn=tfp.bijectors.GlowDefaultNetwork,
    exit_bijector_fn=tfp.bijectors.GlowDefaultExitNetwork
)
distribution = tfp.distributions.TransformedDistribution(
    tfp.distributions.Sample(
        tfp.distributions.Normal(0., 1.), (size * size * 3,)
    ), model
)

def log_sample(epochs, logs):
    with summary_writer.as_default():
        fake = distribution.sample() * 0.5 + 0.5
        tf.summary.image('fake', fake, epochs, 4)
        del fake


name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
summary_writer = tf.summary.create_file_writer(os.path.join("logs", name))

log_probability_model = LogProb(distribution)

log_probability_model.compile(
    tf.keras.optimizers.Adam(),#.SGD(1e-3), 
    lambda y_true, log_prob: -tf.reduce_mean(log_prob)
)

log_probability_model.fit(
    datasets[1], steps_per_epoch=1000, epochs=100,
    callbacks=[
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=log_sample
        )
    ]
)