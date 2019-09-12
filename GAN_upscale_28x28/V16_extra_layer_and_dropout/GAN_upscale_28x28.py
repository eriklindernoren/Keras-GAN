# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:14:02 2019

@author: gerhard
"""

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D,Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.initializers import RandomNormal

import matplotlib.pyplot as plt

import sys

import numpy as np

import glob

import pickle

import tensorflow

import keras


def load_data():
        x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")
        
        with open(x_files[0],'rb') as x_file:
            x = pickle.load(x_file)
        
        for i in x_files[1:10]:
            print(i)
            with open(i,'rb') as x_file:
                print(i)
                xi = pickle.load(x_file)
                x = np.concatenate((x,xi),axis=0)
                print(x.shape)
        return(x)
            
def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def smooth_positive_labels(y):
	return y - 0.3 + (np.random.random(y.shape) * 0.5)

def smooth_negative_labels(y):
	return y + np.random.random(y.shape) * 0.3

from numpy.random import choice

def noisy_labels(y, p_flip):
	# determine the number of labels to flip
	n_select = int(p_flip * y.shape[0])
	# choose labels to flip
	flip_ix = choice([i for i in range(y.shape[0])], size=n_select)
	# invert the labels in place
	y[flip_ix] = 1 - y[flip_ix]
	return y



class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 3

        optimizer_discr = SGD(0.00003)
        optimizer_gen = Adam(0.00001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer_discr,
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
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_gen)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.5))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.5))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.5))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()
        model.add(keras.layers.GaussianNoise(0.0001,input_shape=self.img_shape))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
    
    


    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train = load_data()
        
        new_x = np.zeros((X_train.shape[0],28,28))

        for i in range(0,X_train.shape[0]):
            x_new_i = np.zeros((28,28))
            x_old_i = X_train[i,:,:]
            x_new_i[5:x_old_i.shape[0]+5,2:x_old_i.shape[1]+2] = x_old_i
            new_x[i,:,:] = x_new_i
            
        X_train = new_x
        
        del new_x

        # Rescale -1 to 1
#        X_train = X_train / 127.5 - 1.
        X_train = scale(X_train)
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
#        valid = np.ones((batch_size, 1))
#        fake = np.zeros((batch_size, 1))
        
        valid = np.full(shape=(batch_size,1),fill_value=0.99)
#        valid = noisy_labels(valid,0.05)
        valid = smooth_positive_labels(valid)
        fake = np.full(shape=(batch_size,1),fill_value=0.01)
#        fake = noisy_labels(fake,0.05)
        fake = smooth_negative_labels(fake)

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
#            for double_whammy in range(0,5):
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
            if epoch % 100 == 0:
                self.sample_images2(epoch)

    def sample_images(self, epoch):
#        r, c = 5, 5
        noise = np.random.normal(0, 1, (2, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        plt.imshow(gen_imgs[1,:,:,0],cmap='gray')
#        plt.axis('off')
#        fig, axs = plt.subplots(r, c)
#        cnt = 0
#        for i in range(r):
#            for j in range(c):
#                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
#                axs[i,j].axis('off')
#                cnt += 1
        plt.savefig("images/%d.png" % epoch)
        plt.close()
        
    def sample_images2(self, epoch):
#        r, c = 5, 5
        noise = np.random.normal(0, 1, (2, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = gen_imgs[:,5:22,2:26,:]
        np.save("simulated_data/"+str(epoch)+".npy",arr=gen_imgs)


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000000000, batch_size=64, sample_interval=100)