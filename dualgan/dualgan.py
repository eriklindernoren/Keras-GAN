from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

class DUALGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_dim = self.img_rows*self.img_cols

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d1 = self.build_discriminator()
        self.d1.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d2 = self.build_discriminator()
        self.d2.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generators
        self.g1 = self.build_generator()
        self.g1.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.g2 = self.build_generator()
        self.g2.compile(loss='binary_crossentropy', optimizer=optimizer)

        # For the combined model we will only train the generator
        self.d1.trainable = False
        self.d2.trainable = False

        # The generator takes images from their respective domains as inputs
        X1 = Input(shape=(self.img_dim,))
        X2 = Input(shape=(self.img_dim,))

        # Generators translates the images to the opposite domain
        X1_translated = self.g1(X1)
        X2_translated = self.g2(X2)

        # The discriminators determines validity of translated images
        valid1 = self.d1(X2_translated)
        valid2 = self.d2(X1_translated)

        # Generators translate the images back to their original domain
        X1_recon = self.g2(X1_translated)
        X2_recon = self.g1(X2_translated)

        # The combined model  (stacked generators and discriminators)
        self.combined = Model([X1, X2], [valid1, valid2, X1_recon, X2_recon])
        self.combined.compile(loss=[self.wasserstein_loss, self.wasserstein_loss, 'mae', 'mae'],
                                    optimizer=optimizer,
                                    loss_weights=[1, 1, 100, 100])

    def build_generator(self):

        X = Input(shape=(self.img_dim,))

        model = Sequential()
        model.add(Dense(256, input_dim=self.img_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))

        X_translated = model(X)

        return Model(X, X_translated)

    def build_discriminator(self):

        img = Input(shape=(self.img_dim,))

        model = Sequential()
        model.add(Dense(512, input_dim=self.img_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.channels))

        validity = model(img)

        return Model(img, validity)

    def sample_generator_input(self, X, batch_size):
        # Sample random batch of images from X
        idx = np.random.randint(0, X.shape[0], batch_size)
        return X[idx]

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        # Domain A and B (rotated)
        X1 = X_train[:int(X_train.shape[0]/2)]
        X2 = scipy.ndimage.interpolation.rotate(X_train[int(X_train.shape[0]/2):], 90, axes=(1, 2))

        X1 = X1.reshape(X1.shape[0], self.img_dim)
        X2 = X2.reshape(X2.shape[0], self.img_dim)

        clip_value = 0.1
        n_critic = 4
        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # Train the discriminator for n_critic iterations
            for _ in range(n_critic):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Sample generator inputs
                imgs1 = self.sample_generator_input(X1, half_batch)
                imgs2 = self.sample_generator_input(X2, half_batch)

                # Translate images to their opposite domain
                X1_translated = self.g1.predict(imgs1)
                X2_translated = self.g2.predict(imgs2)

                # Retranslate images to their original domain
                X1_recon = self.g2.predict(X1_translated)
                X2_recon = self.g1.predict(X2_translated)

                valid = np.ones((half_batch, 1))
                fake = np.zeros((half_batch, 1))

                # Train the discriminators
                d1_loss_real = self.d1.train_on_batch(imgs1, valid)
                d1_loss_fake = self.d1.train_on_batch(X2_translated, fake)

                d2_loss_real = self.d2.train_on_batch(imgs2, valid)
                d2_loss_fake = self.d2.train_on_batch(X1_translated, fake)

                d1_loss = 0.5 * np.add(d1_loss_real, d1_loss_fake)
                d2_loss = 0.5 * np.add(d2_loss_real, d2_loss_fake)

                # Clip discriminator weights
                for d in [self.d1, self.d2]:
                    for l in d.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                        l.set_weights(weights)

            # ------------------
            #  Train Generators
            # ------------------

            # Sample generator inputs from each domain
            imgs1 = self.sample_generator_input(X1, batch_size)
            imgs2 = self.sample_generator_input(X2, batch_size)

            # The generators wants the discriminators to label the generated samples
            # as valid (ones)
            valid = np.ones((batch_size, 1))

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs1, imgs2], [valid, valid, imgs1, imgs2])

            # Plot the progress
            print ("%d [D1 loss: %f, acc.: %.2f%%] [D2 loss: %f, acc.: %.2f%%] [G loss: %f]" \
                % (epoch, d1_loss[0], 100*d1_loss[1], d2_loss[0], 100*d2_loss[1], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch, X1)

    def save_imgs(self, epoch, X):
        r, c = 3, 4

        # Original images
        imgs = self.sample_generator_input(X, c)
        # Images translated to their opposite domain
        X_translated = self.g1.predict(imgs)
        # Images translated back to their original domain
        X_recon = self.g2.predict(X_translated)

        gen_imgs = np.concatenate([imgs, X_translated, X_recon])
        gen_imgs = gen_imgs.reshape((3, 4, self.img_rows, self.img_cols, 1))

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[i, j, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("dualgan/images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = DUALGAN()
    gan.train(epochs=30000, batch_size=32, save_interval=200)
