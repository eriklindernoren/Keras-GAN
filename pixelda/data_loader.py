import scipy
from glob import glob
import numpy as np
from keras.datasets import mnist
from skimage.transform import resize as imresize
import pickle
import os
import urllib
import gzip

class DataLoader():
    """Loads images from MNIST (domain A) and MNIST-M (domain B)"""
    def __init__(self, img_res=(128, 128)):
        self.img_res = img_res

        self.mnistm_url = 'https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz'

        self.setup_mnist(img_res)
        self.setup_mnistm(img_res)

    def normalize(self, images):
        return images.astype(np.float32) / 127.5 - 1.

    def setup_mnist(self, img_res):

        print ("Setting up MNIST...")

        if not os.path.exists('datasets/mnist_x.npy'):
            # Load the dataset
            (mnist_X, mnist_y), (_, _) = mnist.load_data()

            # Normalize and rescale images
            mnist_X = self.normalize(mnist_X)
            mnist_X = np.array([imresize(x, img_res) for x in mnist_X])
            mnist_X = np.expand_dims(mnist_X, axis=-1)
            mnist_X = np.repeat(mnist_X, 3, axis=-1)

            self.mnist_X, self.mnist_y = mnist_X, mnist_y

            # Save formatted images
            np.save('datasets/mnist_x.npy', self.mnist_X)
            np.save('datasets/mnist_y.npy', self.mnist_y)
        else:
            self.mnist_X = np.load('datasets/mnist_x.npy')
            self.mnist_y = np.load('datasets/mnist_y.npy')

        print ("+ Done.")

    def setup_mnistm(self, img_res):

        print ("Setting up MNIST-M...")

        if not os.path.exists('datasets/mnistm_x.npy'):

            # Download the MNIST-M pkl file
            filepath = 'datasets/keras_mnistm.pkl.gz'
            if not os.path.exists(filepath.replace('.gz', '')):
                print('+ Downloading ' + self.mnistm_url)
                data = urllib.request.urlopen(self.mnistm_url)
                with open(filepath, 'wb') as f:
                    f.write(data.read())
                with open(filepath.replace('.gz', ''), 'wb') as out_f, \
                        gzip.GzipFile(filepath) as zip_f:
                    out_f.write(zip_f.read())
                os.unlink(filepath)

            # load MNIST-M images from pkl file
            with open('datasets/keras_mnistm.pkl', "rb") as f:
                data = pickle.load(f, encoding='bytes')

            # Normalize and rescale images
            mnistm_X = np.array(data[b'train'])
            mnistm_X = self.normalize(mnistm_X)
            mnistm_X = np.array([imresize(x, img_res) for x in mnistm_X])

            self.mnistm_X, self.mnistm_y = mnistm_X, self.mnist_y.copy()

            # Save formatted images
            np.save('datasets/mnistm_x.npy', self.mnistm_X)
            np.save('datasets/mnistm_y.npy', self.mnistm_y)
        else:
            self.mnistm_X = np.load('datasets/mnistm_x.npy')
            self.mnistm_y = np.load('datasets/mnistm_y.npy')

        print ("+ Done.")


    def load_data(self, domain, batch_size=1):

        X = self.mnist_X if domain == 'A' else self.mnistm_X
        y = self.mnist_y if domain == 'A' else self.mnistm_y

        idx = np.random.choice(list(range(len(X))), size=batch_size)

        return X[idx], y[idx]
