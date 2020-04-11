import tensorflow as tf
import wandb


class GeneratorLogger:
  def __init__(self, generator, latent_dim):
  	self.generator = generator
  	self.latent_dim

  	r, c = 5, 5
    self.noise = np.random.normal(0, 1, (r * c, self.latent_dim))

  def logGeneratedImages(self):
    gen_imgs = self.generator.predict(self.noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    print(type(gen_imgs))
	wandb.log({"gan_generated": [wandb.Image(gen)
	                    for gen in gen_imgs]})
	