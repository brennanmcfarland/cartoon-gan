import tensorflow as tf
import random
from data import load_metadata, DataProvider
import globals
import layers
from dcgan import DCGAN


tf.set_random_seed(2435)  # for consistency in results when testing
random.seed()

globals.data_root = 'D:/HDD Data/cartoons/'
globals.output_root = 'D:/HDD Data/cartoons'

# get the metadata
metadata = load_metadata()
random.shuffle(metadata)
metadata_train = metadata[len(metadata)//10:]
metadata_test = metadata[:len(metadata_train)]
num_data = len(metadata)

classes = set([datum[0] for datum in metadata])
# build a dictionary mapping between name strings and ids
globals.class_to_id = dict((n, i) for i, n in enumerate(classes))
globals.id_to_class = dict((i, n) for i, n in enumerate(classes))
globals.num_classes = len(classes)

generator = layers.build_generator
discriminator = layers.build_discriminator
data_provider = DataProvider(metadata)
dcgan = DCGAN(generator, discriminator, data_provider)
dcgan.compile()
dcgan.train(epochs=2, batch_size=1, save_interval=1)
dcgan.add_block(layers.add_block)
dcgan.train(epochs=2, batch_size=1, save_interval=1)
