from keras.optimizers import Adam


class GANBase(object):

    def __init__(self, optimizer=Adam(0.0002, 0.5), verbose=True):
        self.optimizer = optimizer
        self.verbose = verbose

    def get_optimizer(self):
        return self.optimizer

    def build_generator(self, *args, **kwargs):
        raise NotImplemented

    def build_critic(self, *args, **kwargs):
        raise NotImplemented

    def load_dataset(self):
        raise NotImplemented

    def train(self, *args, **kwargs):
        raise NotImplemented

    def train_discriminator(self, *args, **kwargs):
        raise NotImplemented

    def train_generator(self, *args, **kwargs):
        raise NotImplemented

    def sample_images(self):
        raise NotImplemented

    def save_model(self):
        raise NotImplemented
