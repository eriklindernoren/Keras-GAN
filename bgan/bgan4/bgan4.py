from __future__ import print_function, division

import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
#from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

def load_data():
    tracks = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")

    infosets = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")

    x = tracks.reshape((-1, 17,24))

    y = np.repeat(infosets[:, 0], 6)
    return (x,y)

class BGAN():
    """Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/"""
    def __init__(self):
        self.img_rows = 17
        self.img_cols = 24
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 50

        optimizer = Adam(0.000001)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.boundary_loss, optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128, input_dim=self.latent_dim,activation=tf.nn.leaky_relu))
#        model.add(LeakyReLU(alpha=0.2))
#        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256,activation=tf.nn.leaky_relu))
        model.add(Dense(256,activation=tf.nn.leaky_relu))
        model.add(Dense(256,activation=tf.nn.leaky_relu))
#        model.add(LeakyReLU(alpha=0.2))
#        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512,activation=tf.nn.leaky_relu))
        model.add(Dense(512,activation=tf.nn.leaky_relu))
        model.add(Dense(512,activation=tf.nn.leaky_relu))
#        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024,activation=tf.nn.leaky_relu))
#        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

#        model.add(Flatten(input_shape=self.img_shape))
        model.add(Conv2D(filters=16,kernel_size=3,input_shape=self.img_shape))
        model.add(Conv2D(filters=32,kernel_size=3))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(1024,activation=tf.nn.leaky_relu))
        model.add(Dense(512,activation=tf.nn.leaky_relu))
#        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256,activation=tf.nn.leaky_relu))
        model.add(Dense(128,activation=tf.nn.leaky_relu))
        model.add(Dense(64,activation=tf.nn.leaky_relu))
#        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def boundary_loss(self, y_true, y_pred):
        """
        Boundary seeking loss.
        Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/
        """
        return 0.5 * K.mean((K.log(y_pred) - K.log(1 - y_pred))**2)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _) = load_data()

        # Rescale -1 to 1
        
#        for i in range(0,X_train.shape[0]):
#            ma = np.max(X_train[i,:,:])
#            X_train[i,:,:] = X_train[i,:,:]/ma
        
        X_train = X_train/np.max(X_train)
        
#        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    bgan = BGAN()
    bgan.train(epochs=30000, batch_size=32, sample_interval=200)
