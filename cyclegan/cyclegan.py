from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
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
        # Translate images back to original domain
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
        """U-Net Generator"""

        input_img = Input(shape=self.img_shape)

        # (32 x 32 x 1)
        d1 = Conv2D(32, kernel_size=4, strides=2, padding='same', input_shape=self.img_shape)(input_img)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d1 = BatchNormalization(momentum=0.8)(d1)
        # (16 x 16 x 32)
        d2 = Conv2D(64, kernel_size=4, strides=2, padding='same')(d1)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = BatchNormalization(momentum=0.8)(d2)
        # (8 x 8 x 64)
        d3 = Conv2D(128, kernel_size=4, strides=2, padding='same')(d2)
        d3 = LeakyReLU(alpha=0.2)(d3)
        d3 = BatchNormalization(momentum=0.8)(d3)
        # (4 x 4 x 128)
        d4 = Conv2D(256, kernel_size=4, strides=2, padding='same')(d3)
        d4 = LeakyReLU(alpha=0.2)(d4)
        d4 = BatchNormalization(momentum=0.8)(d4)
        # (2 x 2 x 256)
        u1 = UpSampling2D(size=2)(d4)
        u1 = Conv2D(128, kernel_size=4, strides=1, padding='same')(u1)
        u1 = LeakyReLU(alpha=0.2)(u1)
        u1 = BatchNormalization(momentum=0.8)(u1)
        u1 = Concatenate()([u1, d3])
        # (4 x 4 x 256)
        u2 = UpSampling2D(size=2)(u1)
        u2 = Conv2D(64, kernel_size=4, strides=1, padding='same')(u2)
        u2 = LeakyReLU(alpha=0.2)(u2)
        u2 = BatchNormalization(momentum=0.8)(u2)
        u2 = Concatenate()([u2, d2])
        # (8 x 8 x 128)
        u3 = UpSampling2D(size=2)(u2)
        u3 = Conv2D(32, kernel_size=4, strides=1, padding='same')(u3)
        u3 = LeakyReLU(alpha=0.2)(u3)
        u3 = BatchNormalization(momentum=0.8)(u3)
        u3 = Concatenate()([u3, d1])
        # (16 x 16 x 64)
        u2 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u2)

        return Model(input_img, output_img)

    def build_discriminator(self):

        img = Input(shape=self.img_shape)

        model = Sequential()
        model.add(Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.8))
        model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(1, kernel_size=4, strides=1, padding='same'))

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

            # Sample a half batch of images from both domains
            idx = np.random.randint(0, X1.shape[0], half_batch)
            imgs_A = X1[idx]
            imgs_B = X2[idx]

            # Translate images to opposite domain
            fake_B = self.g_AB.predict(imgs_A)
            fake_A = self.g_BA.predict(imgs_B)

            valid = np.ones((half_batch, 4, 4, 1))
            fake = np.zeros((half_batch, 4, 4, 1))

            # Train the discriminators (original images = real / translated = Fake)
            dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
            dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
            dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

            dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
            dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

            # Total disciminator loss
            d_loss = 0.5 * np.add(dA_loss, dB_loss)


            # ------------------
            #  Train Generators
            # ------------------

            # Sample a batch of images from both domains
            idx = np.random.randint(0, X1.shape[0], batch_size)
            imgs_A = X1[idx]
            imgs_B = X2[idx]

            # The generators want the discriminators to label the translated images as real
            valid = np.ones((batch_size, 4, 4, 1))

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
        r, c = 6, 4

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

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
