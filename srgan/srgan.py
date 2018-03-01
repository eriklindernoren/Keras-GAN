from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

import keras.backend as K

class SRGAN():
    def __init__(self):
        # Input shape
        self.lr_height = 64                 # Low resolution height
        self.lr_width = 64                  # Low resolution width
        self.hr_height = self.lr_height*4   # High resolution height
        self.hr_width = self.lr_width*4     # High resolution width
        self.channels = 3
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        self.n_residual_blocks = 16

        optimizer = Adam(0.0002, 0.5)

        # Where going to use a pre-trained VGG19 model and minimize the mse
        # between the image embeddings of the high res. images and generated images
        # at the later blocks
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Configure data loader
        self.dataset_name = 'img_align_celeba'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # Low res. and high res. images
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)

        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)

        # Extract image features of the generated img at the three last blocks
        f1, f2, f3 = self.vgg(fake_hr)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        validity = self.discriminator(fake_hr)

        self.combined = Model([img_lr, img_hr], [validity, f1, f2, f3])
        self.combined.compile(loss=['binary_crossentropy', 'mse', 'mse', 'mse'],
                              loss_weights=[1e-3, 0.33, 0.33, 0.33],
                              optimizer=optimizer)


    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        last three blocks of the model
        """
        vgg = VGG19(weights="imagenet")
        # Set outputs to outputs of last conv. layers in block 3, 4 and 5
        vgg.outputs = [vgg.layers[9].output,
                       vgg.layers[14].output,
                       vgg.layers[19].output]

        img = Input(shape=self.hr_shape)

        # Extract image features
        f1, f2, f3 = vgg(img)

        return Model(img, [f1, f2, f3])

    def build_generator(self):

        def residual_block(layer_input):
            d = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same')(layer_input)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(64, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(u)
            return u

        # Image input
        d0 = Input(shape=self.lr_shape)
        d1 = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same')(d0)

        # Propogate through residual blocks
        r = residual_block(d1)
        for _ in range(1 - self.n_residual_blocks):
            r = residual_block(r)

        # Upsampling
        u1 = deconv2d(r)
        u2 = deconv2d(u1)

        output_img = Conv2D(self.channels, kernel_size=3, strides=1, padding='same', activation='tanh')(u2)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)

        d9 = Dense(self.df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)

    def train(self, epochs, batch_size=1, save_interval=50):

        start_time = datetime.datetime.now()

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # From low res. image generate high res. version
            fake_hr = self.generator.predict(imgs_lr)

            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + self.disc_patch)

            # Extract ground truth image features at last three blocks of VGG19
            f1, f2, f3 = self.vgg.predict(imgs_hr)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, f1, f2, f3])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("%d time: %s" % (epoch, elapsed_time))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 2

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2, is_testing=True)
        fake_hr = self.generator.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d.png" % (self.dataset_name, epoch))
        plt.close()

        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('images/%s/%d_lowres%d.png' % (self.dataset_name, epoch, i))


if __name__ == '__main__':
    gan = SRGAN()
    gan.train(epochs=30000, batch_size=1, save_interval=50)
