from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import binary_crossentropy
import numpy as np
from skimage import io
import matplotlib.pyplot as plt


def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the
    discriminator has a sigmoid output, representing the probability that samples are
    real or generated. In Wasserstein GANs, however, the output is linear with no
    activation function! Instead of being constrained to [0, 1], the discriminator wants
    to make the distance between its output for real and generated samples as
    large as possible.
    The most natural way to achieve this is to label generated samples -1 and real
    samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
    outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be)
    less than 0."""
    return K.mean(y_true * y_pred)


def combined_loss(y_true, y_pred):
    return wasserstein_loss(y_true, y_pred) + binary_crossentropy(y_true, y_pred)


class DCGAN():

    def __init__(self, g, d, data_provider):
        g_inputs, g_outputs, self.mapper = g()
        d_inputs, d_outputs = d()

        self.img_rows = 10
        self.img_cols = 28
        self.channels = 1
        self.latent_dim = 100

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.generator = Model(inputs=g_inputs, outputs=g_outputs)
        self.discriminator = Model(inputs=d_inputs, outputs=d_outputs)

        # [noise, self.constant_input, self.noise_input]
        gan_inputs = [Input(shape=(100,)), Input(shape=(5, 14, 32)), Input(shape=(5, 14, 32))]

        gan_images = self.generator(gan_inputs)
        gan_output = self.discriminator(gan_images)

        self.gan = Model(inputs=gan_inputs, outputs=gan_output)
        self.g_inputs = g_inputs
        self.g_outputs = g_outputs
        self.d_inputs = d_inputs
        self.d_outputs = d_outputs

        self.data_provider = data_provider
        self.stage = 0

    def compile(self):

        # optimizer = Adam()
        optimizer = Adam(0.0002, 0.5, clipnorm=1.0)
        self.discriminator.compile(loss=combined_loss,
                                   optimizer=optimizer)

        self.constant_input = K.variable(np.full((1, 5, 14, 32), .5, dtype='float32'))
        self.noise_input = K.variable(np.full((1, 5, 14, 32), 1, dtype='float32'))

        latent = Input(shape=(self.latent_dim,))
        img = self.generator([latent, self.constant_input, self.noise_input])

        # The discriminator takes generated images as input and determines validity
        self.discriminator.trainable = False
        valid = self.discriminator(img)

        # trains the generator to fool the discriminator
        gen_optimizer = Adam(clipnorm=1.0)
        self.gan.compile(loss=combined_loss, optimizer=gen_optimizer)
        self.gan.summary()
        plot_model(self.gan, to_file='gan_model.png')
        plot_model(self.generator, to_file='generator_model.png')
        plot_model(self.discriminator, to_file='discriminator_model.png')

    def add_block(self, block):
        # params needed: block input, n_input (for the block), mapper
        g_input = self.g_outputs
        d_output = self.discriminator
        d_inputs = self.d_inputs
        n_input = self.g_inputs[2]
        mapper = UpSampling2D()(self.mapper)
        new_g_output, discriminator = block(g_input, d_output, d_inputs, n_input, mapper)
        self.g_outputs = new_g_output
        self.generator = Model(inputs=self.g_inputs, outputs=self.g_outputs)
        self.discriminator = discriminator

        self.generator.summary()
        self.discriminator.summary()

        gan_inputs = [Input(shape=(100,)), Input(shape=(5, 14, 32)), Input(shape=(5, 14, 32))]
        gan_images = self.generator(gan_inputs)
        gan_output = self.discriminator(gan_images)

        self.gan = Model(inputs=gan_inputs, outputs=gan_output)
        self.compile()  # TODO: this probably resets learning params and we probably don't want it to do that
        self.data_provider.upscale()
        self.stage += 1

    def train(self, epochs, batch_size=128, save_interval=1):

        data_provider = self.data_provider

        # Adversarial ground truths
        # for whatever reason, valid=zeros is supposed to help convergence
        # and noise is supposed to help not overtrain, too
        # valid=ones is necessary for wasserstein
        valid = np.random.uniform(.9, .1, (batch_size, 1))
        fake = np.random.uniform(-.9, 1.0, (batch_size, 1))
        # valid = np.zeros((batch_size, 1))
        # fake = np.ones((batch_size, 1))

        steps_per_epoch = 100

        for epoch in range(epochs):

            d_loss = [10.0, 10.0]
            g_loss = 10.0
            d_loss_threshold = .7
            g_loss_threshold = .75
            steps = 0
            while steps < steps_per_epoch:

                # train discriminator to threshold loss or end of epoch, whichever first
                # self.discriminator.trainable = True
                while True:
                    if steps >= steps_per_epoch:
                        break
                    steps += 1

                    imgs = data_provider.get_batch()[1]

                    # Sample noise and generate a batch of new images
                    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                    gen_imgs = self.generator.predict([noise, self.constant_input, self.noise_input], steps=1)

                    # Train the discriminator (real classified as ones and generated as zeros)
                    d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                    # TODO: use shannon entropy instead to discourage counting everythign as valid/invalid?
                    d_loss = 0.5 * np.add(d_loss_real,
                                          d_loss_fake)  # TODO: get it to be an integer aggregating all values (what are they?)

                    # print(d_loss)
                    # if d_loss[0]*log(d_loss[0]) + d_loss[1]*log(d_loss[1]) < d_loss_threshold:
                    #    break

                    if d_loss < d_loss_threshold:
                        break
                # train generator to threshold loss or end of epoch, whichever first
                # halt training on the discriminator
                self.discriminator.trainable = False  # does this reset every training step?  where is it enabled again?
                while True:
                    if steps >= steps_per_epoch:
                        break
                    print('training generator')
                    steps += 1
                    # get a new batch of noise
                    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                    # Train the generator (wants discriminator to mistake images as real)
                    g_loss = self.gan.train_on_batch([noise, self.constant_input, self.noise_input], valid)

                    if g_loss < g_loss_threshold:
                        break

            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 1, 1
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        gen_img = self.generator.predict([noise, self.constant_input, self.noise_input], steps=1)[0]

        # Rescale images 0 - 1
        gen_img = 0.5 * gen_img + 0.5

        gen_img = np.squeeze(gen_img, -1)
        io.imshow(gen_img)
        plt.savefig("../images/mnist_%d_%d.png" % (self.stage, epoch))
        plt.close()