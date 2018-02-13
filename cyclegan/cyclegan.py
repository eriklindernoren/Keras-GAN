from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import datetime

import matplotlib.pyplot as plt

import sys

import numpy as np

class CycleGAN():
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.conv_dim = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()
        self.g_AB.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.g_BA.compile(loss='binary_crossentropy', optimizer=optimizer)

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Reconstruct images to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        self.combined = Model([img_A, img_B], [valid_A, valid_B, reconstr_A, reconstr_B])
        self.combined.compile(loss=['mse', 'mse', 'mae', 'mae'],
                                    loss_weights=[1, 1, 10, 10],
                                    optimizer=optimizer)

    def build_generator(self):

        # U-Net Generator

        input_img = Input(shape=self.img_shape)

        # Downsampling
        d1 = Conv2D(self.conv_dim, kernel_size=4, strides=2, padding='same', input_shape=self.img_shape)(input_img)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d1 = BatchNormalization(momentum=0.8)(d1)
        d2 = Conv2D(self.conv_dim*2, kernel_size=4, strides=2, padding='same')(d1)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = BatchNormalization(momentum=0.8)(d2)

        # Residual
        r1 = Conv2D(self.conv_dim*2, kernel_size=3, strides=1, padding='same')(d2)
        r1 = LeakyReLU(alpha=0.2)(r1)
        r2 = Conv2D(self.conv_dim*2, kernel_size=3, strides=1, padding='same')(r1)
        r2 = LeakyReLU(alpha=0.2)(r2)

        # Upsampling
        u1 = UpSampling2D(2)(r2)
        u1 = Conv2D(self.conv_dim, kernel_size=4, strides=1, padding='same')(u1)
        u1 = LeakyReLU(alpha=0.2)(u1)
        u1 = BatchNormalization(momentum=0.8)(u1)
        u2 = UpSampling2D(2)(u1)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u2)

        return Model(input_img, output_img)

    def build_discriminator(self):

        img = Input(shape=self.img_shape)

        # Shared discriminator layers
        model = Sequential()
        model.add(Conv2D(self.conv_dim, kernel_size=4, strides=2, input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.8))
        model.add(Conv2D(self.conv_dim*2, kernel_size=4, strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.conv_dim*4, kernel_size=4, strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(1))

        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale MNIST to 32x32
        X_train = np.array([scipy.misc.imresize(x, [self.img_rows, self.img_cols]) for x in X_train])

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Images in domain A and B (rotated)
        X1 = X_train[:int(X_train.shape[0]/2)]
        X2 = X_train[int(X_train.shape[0]/2):]
        X2 = scipy.ndimage.interpolation.rotate(X2, 90, axes=(1, 2))

        half_batch = int(batch_size / 2)

        start_time = datetime.datetime.now()

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminators
            # ----------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X1.shape[0], half_batch)
            imgs_A = X1[idx]
            imgs_B = X2[idx]

            # Translate images to opposite domain
            fake_B = self.g_AB.predict(imgs_A)
            fake_A = self.g_BA.predict(imgs_B)

            # Train the discriminators (original images = real / Translated = Fake)
            dA_loss_real = self.d_A.train_on_batch(imgs_A, np.ones((half_batch, 1)))
            dA_loss_fake = self.d_A.train_on_batch(fake_A, np.zeros((half_batch, 1)))
            dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

            dB_loss_real = self.d_B.train_on_batch(imgs_B, np.ones((half_batch, 1)))
            dB_loss_fake = self.d_B.train_on_batch(fake_B, np.zeros((half_batch, 1)))
            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

            d_loss = 0.5 * np.add(dA_loss, dB_loss)


            # ------------------
            #  Train Generators
            # ------------------

            idx = np.random.randint(0, X1.shape[0], batch_size)
            imgs_A = X1[idx]
            imgs_B = X2[idx]

            # The generators wants the discriminators to label the translated images as real
            valid = np.array([1] * batch_size)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("%d time: %s" % (epoch, elapsed_time))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                idx = np.random.randint(0, X1.shape[0], 4)
                imgs_A = X1[idx]
                imgs_B = X2[idx]
                self.save_imgs(epoch, imgs_A, imgs_B)

    def save_imgs(self, epoch, imgs_A, imgs_B):
        r, c = 4, 4

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_A, fake_B, imgs_B, fake_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("cyclegan/images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = CycleGAN()
    gan.train(epochs=30000, batch_size=2, save_interval=200)
