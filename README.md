# Keras-GAN

## About
Keras implementations of Generative Adversarial Network (GAN) models. Since I'm running these models from my Macbook Pro
they will be limited in their complexity (and therefore the quality of the generated images). Short training sessions are prioritized.

## Table of Contents
- [Keras-GAN](#keras-gan)
  * [About](#about)
  * [Table of Contents](#table-of-contents)
  * [Implementations](#implementations)
    + [Auxiliary Classifier GAN](#ac-gan)
    + [Context-Conditional GAN](#cc-gan)
    + [Deep Convolutional GAN](#dcgan)
    + [Generative Adversarial Network](#gan)
    + [Semi-Supervised GAN](#sgan)
    + [Wasserstein GAN](#wgan)
 

## Implementations   
### AC-GAN
Implementation of a Auxiliary Classifier Generative Adversarial Network.

Reference: https://arxiv.org/abs/1610.09585

<p align="center">
    <img src="http://eriklindernoren.se/images/acgan2.png" width="640"\>
</p>

### CC-GAN
Implementation of Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks.

Inpainting using a GAN where the generator is conditioned on a randomly masked image. In this implementation
images of dogs and cats taken from the Cifar-10 dataset are used. These images are of very low resolution and
the results are therefore not as nice as in the implementation described in the paper. In this implementation I have
also decided to combine the adversarial loss with an l2 loss which measures the generated images similarity to the original
images. These losses are weighted similar to the approach Pathak et al. (2016) described 
in their paper [Context Encoders: Feature Learning by Inpainting](https://arxiv.org/abs/1604.07379).

Reference: https://arxiv.org/abs/1611.06430

<p align="center">
    <img src="http://eriklindernoren.se/images/ccgan.png" width="640"\>
</p>

### DCGAN
Implementation of a Deep Convolutional Generative Adversarial Network.

Reference: https://arxiv.org/abs/1511.06434

<p align="center">
    <img src="http://eriklindernoren.se/images/dcgan2.png" width="640"\>
</p>

### GAN
Implementation of a Generative Adversarial Network with a MLP generator and discriminator.

Reference: https://arxiv.org/abs/1406.2661

<p align="center">
    <img src="http://eriklindernoren.se/images/gan.png" width="640"\>
</p>

### SGAN
Implementation of a Semi-Supervised Generative Adversarial Network.

Reference: https://arxiv.org/abs/1606.01583

<p align="center">
    <img src="http://eriklindernoren.se/images/sgan.png" width="640"\>
</p>

### WGAN
Implementation of Wasserstein GAN (with DCGAN generator and discriminator).

Reference: https://arxiv.org/abs/1701.07875

<p align="center">
    <img src="http://eriklindernoren.se/images/wgan2.png" width="640"\>
</p>
