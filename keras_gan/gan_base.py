class GANBase(object):

    def __init__(self):
        pass

    def build_generator(self):
        raise NotImplemented

    def build_critic(self):
        raise NotImplemented

    def train(self):
        raise NotImplemented

    def train_discriminator(self):
        raise NotImplemented

    def train_generator(self):
        raise NotImplemented

    def sample_images(self, epoch):
        raise NotImplemented

    def save_model(self):
        raise NotImplemented
