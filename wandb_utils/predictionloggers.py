import tensorflow as tf
import wandb


class GeneratorLogger(tf.keras.callbacks.Callback):
  def __init__(self, generator, latent_dim):
  	self.generator = generator
  	self.latent_dim

    super(GeneratorLogger, self).__init__()

  def on_epoch_end(self, logs, epoch):
      
  	r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, self.latent_dim))
    gen_imgs = self.generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/mnist_%d.png" % epoch)
    plt.close()





	sample_idx = 54
	sample_segments, _ = self.testgen[sample_idx]  

	segs = []
	r_segs = [] ## reconstruction

	for i in range(32):
	inputs = [sample_segments[i].reshape((1,)+sample_segments[i].shape)]
	r_input = self.model.predict(inputs)

	fig1 = plt.figure(figsize=(6,2))
	plt.plot(sample_segments[i])
	plt.savefig('ori.jpg')
	plt.close()
	img = Image.open('ori.jpg')
	segs.append(np.asarray(img))

	fig2 = plt.figure(figsize=(6,2))
	plt.plot(r_input.reshape(r_input.shape[1:]))
	plt.savefig('reconstruction.jpg')
	plt.close()
	img = Image.open('reconstruction.jpg')
	r_segs.append(np.asarray(img))

	wandb.log({"original": [wandb.Image(seg)
	                    for seg in segs]})
	wandb.log({"reconstructed": [wandb.Image(r_seg)
	                    for r_seg in r_segs]})