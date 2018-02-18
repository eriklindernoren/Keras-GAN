import scipy
from glob import glob
import numpy as np

class DataLoader():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load_data(self, domain, batch_size=1, img_width=64, img_height=64, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = scipy.misc.imresize(img, [img_height, img_width])

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, [img_height, img_width])
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
