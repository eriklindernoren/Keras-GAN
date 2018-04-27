from __future__ import print_function, division
import scipy

import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os


# Configure MNIST and MNIST-M data loader
data_loader = DataLoader(img_res=(32, 32))

mnist, _ = data_loader.load_data(domain="A", batch_size=25)
mnistm, _ = data_loader.load_data(domain="B", batch_size=25)

r, c = 5, 5

for img_i, imgs in enumerate([mnist, mnistm]):

    #titles = ['Original', 'Translated']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(imgs[cnt])
            #axs[i, j].set_title(titles[i])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("%d.png" % (img_i))
    plt.close()
