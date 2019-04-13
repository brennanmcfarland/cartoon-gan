import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Dropout, Activation, GaussianNoise, Reshape, Add, Flatten, LeakyReLU, Input
from tensorflow.keras.constraints import min_max_norm
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


def normalize_data_format(value):
    if value is None:
        value = 'channels_last'
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format


class AdaIN(Layer):
    def __init__(self, data_format=None, eps=1e-7, **kwargs):
        super(AdaIN, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.spatial_axis = [1, 2] if self.data_format == 'channels_last' else [2, 3]
        self.eps = eps

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs):
        image = inputs[0]
        if len(inputs) == 2:
            style = inputs[1]
            style_mean, style_var = tf.nn.moments(style, self.spatial_axis, keep_dims=True)
        else:
            style_mean = tf.expand_dims(K.expand_dims(inputs[1], self.spatial_axis[0]), self.spatial_axis[1])
            style_var = tf.expand_dims(K.expand_dims(inputs[2], self.spatial_axis[0]), self.spatial_axis[1])
        image_mean, image_var = tf.nn.moments(image, self.spatial_axis, keep_dims=True)
        out = tf.nn.batch_normalization(image, image_mean,
                                         image_var, style_mean,
                                         tf.sqrt(style_var), self.eps)
        return out


conv_weight_constraint = .01  # this constraint appears by empirical testing to be the most responsible for keeping the output range from blowing up
noise_weight_constraint = .01  # if this is set high the output doesn't change as much (probably because the noise weights dominate)
mapper_weight_constraint = 1000.0  # set this really high as it didn't seem to affect the output
gaussian_noise_mean = .5


def build_mapper():
    m_input = Input(shape=(100,), name='m_input')
    layer = Dense(128, kernel_constraint=min_max_norm(-mapper_weight_constraint, mapper_weight_constraint))(m_input)
    # layer = BatchNormalization(momentum=.9)(layer)
    # layer = Dropout(.3)(layer)
    layer = LeakyReLU(.25)(layer)

    layer = Dense(128, kernel_constraint=min_max_norm(-mapper_weight_constraint, mapper_weight_constraint))(layer)
    # layer = BatchNormalization(momentum=.9)(layer)
    # layer = Dropout(.3)(layer)
    layer = LeakyReLU(.25)(layer)

    layer = Dense(128, kernel_constraint=min_max_norm(-mapper_weight_constraint, mapper_weight_constraint))(layer)
    # layer = BatchNormalization(momentum=.9)(layer)
    # layer = Dropout(.3)(layer)
    layer = LeakyReLU(.25)(layer)

    layer = Dense(2240, kernel_constraint=min_max_norm(-mapper_weight_constraint, mapper_weight_constraint))(layer)
    layer = LeakyReLU(.25)(layer)

    m_output = Reshape((5, 14, 32), name='mapper_reshape')(layer)
    # mapper_model = Model(inputs=m_input, outputs=m_output)
    # plot_model(mapper_model, to_file='mapper_model.png')
    return m_input, m_output


def build_generator():
    mapper_input, mapper = build_mapper()
    # g_input = Input(tensor=K.variable(np.random.uniform(size=(1, 5, 14, 32)), name='constant_input'))
    g_input = Input(shape=(5, 14, 32), name='constant_input')
    layer = g_input
    layer = Conv2DTranspose(32, 3, padding='same',
                            kernel_constraint=min_max_norm(-conv_weight_constraint, conv_weight_constraint,
                                                           axis=[0, 1, 2]))(layer)
    layer = LeakyReLU(.25)(layer)
    # add AdaIN function with noise and mapper inputs
    # n_input = Input(tensor=K.variable(tf.zeros(shape=layer.shape)), name='noise_input')
    n_input = Input(shape=(5, 14, 32), name='noise_input')
    u_noise = GaussianNoise(gaussian_noise_mean)(n_input)
    u_noise = Flatten()(u_noise)  # is this necessary?
    noise = Dense(2240, kernel_constraint=min_max_norm(-noise_weight_constraint, noise_weight_constraint))(u_noise)
    noise = Reshape((5, 14, 32))(noise)
    layer = Add()([layer, noise])
    layer = AdaIN()([layer, mapper])
    # layer = BatchNormalization(momentum=.9)(layer)
    # the above again
    layer = Conv2DTranspose(32, 3, padding='same',
                            kernel_constraint=min_max_norm(-conv_weight_constraint, conv_weight_constraint,
                                                           axis=[0, 1, 2]))(layer)
    layer = LeakyReLU(.25)(layer)
    u_noise2 = GaussianNoise(gaussian_noise_mean)(n_input)
    u_noise2 = Flatten()(u_noise2)  # is this necessary?
    noise2 = Dense(2240, kernel_constraint=min_max_norm(-noise_weight_constraint, noise_weight_constraint))(u_noise2)
    noise2 = Reshape((5, 14, 32))(noise2)
    # layer = Add()([layer, noise2])
    # layer = AdaIN()([layer, mapper])
    # layer = BatchNormalization(momentum=.9)(layer)
    # layer = UpSampling2D()(layer)
    # and repeat
    # add AdaIN function with noise and mapper inputs
    # layer = Conv2DTranspose(32, 3, padding='same')(layer)
    # layer = LeakyReLU(.25)(layer)
    # layer = GaussianNoise(.5)(layer)
    # layer = AdaIN()([layer, mapper])
    # layer = BatchNormalization(momentum=.9)(layer)
    # the above again
    # layer = Conv2DTranspose(32, 3, padding='same')(layer)
    # layer = LeakyReLU(.25)(layer)
    # layer = GaussianNoise(.5)(layer)
    # layer = AdaIN()([layer, mapper])
    # layer = BatchNormalization(momentum=.9)(layer)
    # layer = UpSampling2D()(layer)

    layer = Conv2DTranspose(128, 3, padding='same',
                            kernel_constraint=min_max_norm(-conv_weight_constraint, conv_weight_constraint,
                                                           axis=[0, 1, 2]))(layer)
    layer = LeakyReLU(.25)(layer)
    layer = Conv2DTranspose(1, 2, padding='same',
                            kernel_constraint=min_max_norm(-conv_weight_constraint, conv_weight_constraint,
                                                           axis=[0, 1, 2]))(layer)
    # no (nonlinear) activation necessary for wasserstein
    # layer = Activation('sigmoid')(layer)
    print('generator output shape: ', layer.shape)

    g_output = layer

    g_inputs = [mapper_input, g_input, n_input]
    g_outputs = g_output

    g = Model(inputs=g_inputs, outputs=g_output)
    plot_model(g, to_file='model.png')
    print("plotted model")

    return g_inputs, g_outputs, mapper


# TODO: may want to have multiple AdaIn per block as in the original paper, but let's see if this works first
def add_block(g_input, d_output, d_inputs, n_input, mapper):
    # generator
    layer = g_input
    layer = UpSampling2D()(layer)
    layer = Conv2DTranspose(32, 3, padding='same',
                            kernel_constraint=min_max_norm(-conv_weight_constraint, conv_weight_constraint,
                                                           axis=[0, 1, 2]))(layer)
    layer = LeakyReLU(.25)(layer)
    u_noise2 = UpSampling2D()(n_input)
    u_noise2 = GaussianNoise(gaussian_noise_mean)(u_noise2)
    u_noise2 = Flatten()(u_noise2)  # is this necessary?
    noise2 = Dense(8960, kernel_constraint=min_max_norm(-noise_weight_constraint, noise_weight_constraint))(u_noise2)
    noise2 = Reshape((10, 28, 32))(noise2)
    layer = Add()([layer, noise2])
    layer = AdaIN()([layer, mapper])
    # may also be issues with the output layer here
    layer = Conv2DTranspose(128, 3, padding='same',
                            kernel_constraint=min_max_norm(-conv_weight_constraint, conv_weight_constraint,
                                                           axis=[0, 1, 2]))(layer)
    layer = LeakyReLU(.25)(layer)
    g_output = Conv2DTranspose(1, 2, padding='same',
                               kernel_constraint=min_max_norm(-conv_weight_constraint, conv_weight_constraint,
                                                              axis=[0, 1, 2]))(layer)

    # discriminator
    d_input = Input(shape=(10, 28, 1))
    dlayer = Conv2D(filters=32, kernel_size=(4, 4), strides=1, padding='same')(d_input)
    dlayer = BatchNormalization()(dlayer)
    dlayer = LeakyReLU(.25)(dlayer)
    dlayer = Dropout(.3)(dlayer)
    dlayer = Conv2D(filters=32, kernel_size=(4, 4), strides=1, padding='same')(dlayer)
    dlayer = LeakyReLU(.25)(dlayer)
    dlayer = Dropout(.3)(dlayer)
    dlayer = Conv2D(filters=1, kernel_size=(4, 4), strides=2, padding='same')(dlayer)
    submodel = Model(inputs=d_input, outputs=dlayer)
    # print(d_output.layers[0].input_shape)

    # gan_images = self.generator(gan_inputs)
    # gan_output = self.discriminator(gan_images)
    # self.gan = Model(inputs=gan_inputs, outputs=gan_output)
    partial_output = submodel(d_input)
    full_output = d_output(partial_output)
    tmp_model = Model(inputs=d_input, outputs=full_output)
    # print(tmp_model.layers[-1].output_shape)
    # print(dlayer.shape)
    # d_output = d_output(tmp_model)
    d_output = tmp_model
    return g_output, d_output


def build_discriminator():
    d_input = Input(shape=(5, 14, 1))
    layer = Conv2D(filters=32, kernel_size=(4, 4), strides=4, padding='same')(d_input)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(.25)(layer)
    layer = Dropout(.3)(layer)
    layer = Conv2D(filters=32, kernel_size=(4, 4), strides=4, padding='same')(layer)
    layer = LeakyReLU(.25)(layer)
    layer = Dropout(.3)(layer)
    layer = Flatten()(layer)
    layer = Dense(128, name='problem_layer')(layer)
    layer = Activation('sigmoid')(layer)
    layer = Dense(1)(layer)
    d_output = Activation('sigmoid')(layer)

    return d_input, d_output
