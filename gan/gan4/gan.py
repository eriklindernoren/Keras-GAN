from __future__ import print_function, division

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
#from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

import tensorflow as tf

import glob

import pickle

def load_data():
        x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")
        tracks = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")
    
        infosets = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")
        x = tracks.reshape((-1, 17,24))
        
        for i in x_files[1:5]:
            print(i)
            with open(i,'rb') as x_file:
                print(i)
                xi = pickle.load(x_file)
                x = np.concatenate((x,xi),axis=0)
                print(x.shape)
    
            
        
        
        
#        x = tracks
    
        y = np.repeat(infosets[:, 0], 6)
        return (x,y)

class GAN():
    def __init__(self):
        self.img_rows = 17
        self.img_cols = 24
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.00004, 0.5)
        optimizer2 = Adam(0.00001, 0.5)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer2)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim,activation=tf.nn.leaky_relu))
        model.add(Dense(128,activation=tf.nn.leaky_relu))
        model.add(BatchNormalization(momentum=0.8))
#        model.add(Dropout(rate=0.5))
        model.add(Dense(256,activation=tf.nn.leaky_relu))
        model.add(BatchNormalization(momentum=0.8))
#        model.add(Dropout(rate=0.5))
        model.add(Dense(512,activation=tf.nn.leaky_relu))
        model.add(BatchNormalization(momentum=0.8))
#        model.add(Dropout(rate=0.5))
        model.add(Dense(1024,activation=tf.nn.leaky_relu))
#        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(rate=0.5))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

#        model.add(Flatten(input_shape=self.img_shape))
#        model.add(Conv2D(filters=16,kernel_size=(3,3),input_shape=self.img_shape))
#        model.add(Conv2D(filters=32,kernel_size=(6,6)))
#        model.add(MaxPooling2D())
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512,activation=tf.nn.leaky_relu))
#        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256,activation=tf.nn.leaky_relu))
#        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _) = load_data()

        # Rescale -1 to 1
        X_train = X_train.astype('float32') / np.max(X_train) - 1.
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

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
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
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000000000000000000000000000000000000000000000, batch_size=32, sample_interval=200)
