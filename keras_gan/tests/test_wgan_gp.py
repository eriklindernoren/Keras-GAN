import os
import shutil

from unittest import main, TestCase

from keras_gan.wgan_gp import WGANGP


class TestWGANGPBasicUse(TestCase):

    def setUp(self):
        shutil.rmtree("./images", ignore_errors=True)
        shutil.rmtree("./models", ignore_errors=True)
        os.mkdir("./images")
        os.mkdir("./models")

    def tearDown(self):
        shutil.rmtree("./images", ignore_errors=True)
        shutil.rmtree("./models", ignore_errors=True)

    def test_wgan_save(self):
        gan = WGANGP()

        # save model
        path_dict = gan.save()

        # There should be 2 hdf5 files in the models folder
        hdf5_fns = [fn for fn in os.listdir("./models") if fn.endswith("hdf5")]
        self.assertEqual(len(hdf5_fns), 2)

        # There should be one json file in the models folder
        json_fns = [fn for fn in os.listdir("./models") if fn.endswith("json")]
        self.assertEqual(len(json_fns), 3)

        gan2 = WGANGP.load(**path_dict)

    def test_basic_workflow(self):
        # Construct WGANGP
        gan = WGANGP()

        # Run 7 epochs with sample_interval of 3
        gan.train(epochs=7, batch_size=32, sample_interval=3)

        # There should be 2 images in the images folder
        image_fns = [fn for fn in os.listdir("./images") if fn.endswith("png")]
        self.assertEqual(len(image_fns), 2)


if __name__ == "__main__":
    main()
