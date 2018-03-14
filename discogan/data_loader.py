import scipy
from glob import glob
import numpy as np

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type_A = "trainA" if not is_testing else "testA"
        data_type_B = "trainB" if not is_testing else "testB"
        path_A = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type_A))
        path_B = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type_B))

        idx = np.random.choice(range(len(path_A)), size=batch_size)
        batch_A = [path_A[i] for i in idx]
        batch_B = [path_B[i] for i in idx]

        imgs_A, imgs_B = [], []
        for img_path_A, img_path_B in zip(batch_A, batch_B):
            img_A = self.imread(img_path_A)
            img_B = self.imread(img_path_B)

            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)

            if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
