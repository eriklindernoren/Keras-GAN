# Keras-GAN

## About
Keras implementations of Generative Adversarial Networks (GANs) suggested in research papers. Since I'm training these models on my Macbook Pro they will be limited in their complexity (and therefore the quality of the generated images) compared to the implementations suggested in the papers. Short training sessions are prioritized.

## Table of Contents
- [Keras-GAN](#keras-gan)
  * [About](#about)
  * [Table of Contents](#table-of-contents)
  * [Implementations](#implementations)
    + [Auxiliary Classifier GAN](#ac-gan)
    + [Adversarial Autoencoder](#adversarial-autoencoder)
    + [Bidirectional GAN](#bigan)
    + [Context-Conditional GAN](#cc-gan)
    + [Context Encoder](#context-encoder)
    + [Deep Convolutional GAN](#dcgan)
    + [Generative Adversarial Network](#gan)
    + [InfoGAN](#infogan)
    + [Semi-Supervised GAN](#sgan)
    + [Wasserstein GAN](#wgan)
 

## Implementations   
### AC-GAN
Implementation of Auxiliary Classifier Generative Adversarial Network.

[Code](acgan/acgan.py)

Paper: https://arxiv.org/abs/1610.09585

<p align="center">
    <img src="http://eriklindernoren.se/images/acgan2.png" width="640"\>
</p>

### Adversarial Autoencoder
Implementation of Adversarial Autoencoder.

[Code](aae/adversarial_autoencoder.py)

Paper: https://arxiv.org/abs/1511.05644

<p align="center">
    <img src="http://eriklindernoren.se/images/aae.png" width="640"\>
</p>

### BiGAN
Implementation of Bidirectional Generative Adversarial Network.

[Code](bigan/bigan.py)

Paper: https://arxiv.org/abs/1605.09782

### CC-GAN
Implementation of Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks.

Inpainting using a GAN where the generator is conditioned on a randomly masked image. In this implementation
images of dogs and cats taken from the Cifar-10 dataset are used. These images are of very low resolution and
the results are therefore not as nice as in the implementation described in the paper. In this implementation I have
also decided to combine the adversarial loss with an l2 loss which measures the generated image's similarity to the original
images. These losses are weighted similar to the approach described by Pathak et al. (2016)  
in their paper [Context Encoders: Feature Learning by Inpainting](https://arxiv.org/abs/1604.07379).

[Code](ccgan/ccgan.py)

Paper: https://arxiv.org/abs/1611.06430

<p align="center">
    <img src="http://eriklindernoren.se/images/ccgan.png" width="640"\>
</p>

### Context Encoder
Implementation of Context Encoders: Feature Learning by Inpainting.

[Code](context_encoder/context_encoder.py)

Paper: https://arxiv.org/abs/1604.07379

<p align="center">
    <img src="http://eriklindernoren.se/images/context_encoder.png" width="640"\>
</p>

### DCGAN
Implementation of Deep Convolutional Generative Adversarial Network.

[Code](dcgan/dcgan.py)

Paper: https://arxiv.org/abs/1511.06434

<p align="center">
    <img src="http://eriklindernoren.se/images/dcgan2.png" width="640"\>
</p>

### GAN
Implementation of Generative Adversarial Network with a MLP generator and discriminator.

[Code](gan/gan.py)

Paper: https://arxiv.org/abs/1406.2661

<p align="center">
    <img src="http://eriklindernoren.se/images/gan.png" width="640"\>
</p>

### InfoGAN
Implementation of InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets.

[Code](infogan/infogan.py)

Paper: https://arxiv.org/abs/1606.03657


### SGAN
Implementation of Semi-Supervised Generative Adversarial Network.

[Code](sgan/sgan.py)

Paper: https://arxiv.org/abs/1606.01583

<p align="center">
    <img src="http://eriklindernoren.se/images/sgan.png" width="640"\>
</p>

### WGAN
Implementation of Wasserstein GAN (with DCGAN generator and discriminator).

[Code](wgan/wgan.py)

Paper: https://arxiv.org/abs/1701.07875

<p align="center">
    <img src="http://eriklindernoren.se/images/wgan2.png" width="640"\>
</p>
