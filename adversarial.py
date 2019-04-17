from keras.layers import Dense, Reshape, Flatten,Dropout,LeakyReLU
from keras.layers import Input, Activation, BatchNormalization
from keras.models import Model,Sequential
from keras.optimizers import SGD,Adam,RMSprop
from keras.layers import Conv2D,UpSampling2D
from keras.regularizers import l1,l1_l2
from keras.datasets import mnist
import keras.backend as K
import pandas as pd
import numpy as np

from keras_adversarial import AdversarialModel, ImageGridCallback, simple_gan, gan_targets
from keras.adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling, AdversarialOptimizerAlternating
from image_utils import dim_ordering_fix, dim_ordering_input, dim_ordering_reshape, dim_ordeing_unfix


def gan_targets(n):
    #discriminator_fake, discriminator_real
    generator_fake = np.ones((n, 1))
    generator_real = np.ones((n, 1))
    discriminator_fake =  np.ones((n, 1))
    discriminator_real = np.ones((n, 1))

    return [generator_fake, generator_real, discriminator_fake, discriminator_real]



def model_generator():
    nch = 256
    g_input = Input(shape = [100])
    H = Dense(nch * 14 * 14, init = 'glorot_normal')(g_input)
    H = BatchNormalization(mode = 2)(H)
    H = Activation('relu')(H)
    H = dim_ordering_reshape(nch, 14)(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Conv2D(int(nch/2), 3,3, border_mode = 'same', init = 'glorot_normal')(H)
    H = BatchNormalization(mode = 2, axis= 1)(H)
    H = Activation('relu')(H)
    H = Conv2D(1,1,1, border_mode = 'same', init = 'glorot_normal')(H)
    g_V = Activation('sigmoid')(H)
    return Model(g_input, g_V)

def dim_ordering_fix(x):
    if K.image_dim_ordering() == "th":
        return  x
    else:
        return np.transpose(x, (0, 2, 3, 1))

def model_discriminator(input_shape = (1, 28, 28), dropout_rate = 0.5):
    d_input = dim_ordering_input(input_shape, name = 'input_x')
    nch = 512
    H = Conv2D(int(nch/2), 5, 5, subsample = (2, 2), border_mode = 'same', Activation = 'relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(0.5)(H)
    H = Conv2D(nch, 5, 5, subsample = (2, 2))(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(0.5)(H)
    H = Flatten()(H)
    H = Dense(int(nch)/2)(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(0.2)(H)
    d_V = Dense(1, activation='sigmoid')(H)
    return Model(d_input, d_V)

def mnist_process(x):
    x = x.astype('float32')/255.0
    return x

def mnist_data():
    (train_x, train_y),(test_x, test_y) = mnist.load_data()
    return  mnist_process(train_x), mnist_process(test_x)

model = AdversarialModel(base_mode = gan, player_paras = [generator.trainable_weights, discriminator.trainable_weights],
                         player_name = ["generator","discriminator"])
model.adversarial_compile(adversarial_optimizer = AdversarialOptimizerSimultaneous(),
                          player_optimizers = [Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
                          loss = 'binary_crossentropy')

def generator_sampler():
    zsamples = np.random.normal(size= (10 * 10, latent_dim))
    gen = dim_ordeing_unfix(geneartor.predict(zsamples))
    return  gen.reshape((10, 10, 28, 28))

if __name__ == "main":
    latent_dim = 100
    input_shape = (1, 28, 28)
    generator = model_generator()
    discriminator = model_discriminator(input_shape = input_shape)
    gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim, )))
    generator.summary()
    discriminator.summary()
    gan.summary()

    model = AdversarialModel(base_mode = gan, player_params = [generator.trainable_weights, discriminator.trainable_weights],
                             player_names = ["generator", "discriminator"])
    model.adversarial_compile(adversarial_optimizer = AdversarialOptimizerSimultaneous(),
                              player_optimizers = [Adam(1e-4, decay= 1e-4,), Adam(1e-3, decay= 1e-4)])

    generator_cb = ImageGridCallback("output/gan_convolutional/epoch-{:03d}.png",generator_sampler)
    train_x ,test_x = mnist_data()
    train_x = dim_ordering_fix(train_x.reshape((-1 ,1, 28, 28)))
    test_x  = dim_ordering_fix(train_x.reshape((-1 ,1, 28, 28)))

    y = gan_targets(train_x.shape[0])
    y_test = gan_targets(test_x.shape[0])

    history = model.fit(x = train_x, y = y, validation_data = (test_x, y_test),callbacks = [generator_cb], nb_epoch = 100,batch_size = 32 )
    df = pd.DataFrame(history.history)
    df.to_csv("output/gan_convolutional/history.csv")
    generator.save("output/gan_convolutional/generator.h5")
    discriminator.save("output/gan_convolutional/discriminator.h5")