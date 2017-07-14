# Keras-GAN

## About
Keras implementations of Generative Adversarial Network (GAN) models. Since I'm running these models from my Macbook Pro
they will be limited in their complexity (and therefore generated image quality). Short training sessions are prioritized.

## Table of Contents
- [Keras-GAN](#keras-gan)
  * [About](#about)
  * [Table of Contents](#table-of-contents)
  * [Implementations](#implementations)
    + [Auxiliary Classifier GAN](#ac-gan)
    + [Deep Convolutional GAN](#dcgan)
    + [Generative Adversarial Network](#gan)
    + [Semi-Supervised GAN](#sgan)
    + [Wasserstein GAN](#wgan)
 
## Implementations   
### AC-GAN
Implementation of the Auxiliary Classifier Generative Adversarial Network.

Reference: https://arxiv.org/abs/1610.09585

<p align="center">
    <img src="http://eriklindernoren.se/images/acgan2.png" width="640"\>
</p>
<p align="center">
    Figure: Generated handwritten digits.
</p>

### DCGAN
Implementation of the Deep Convolutional Generative Adversarial Network.

Reference: https://arxiv.org/abs/1511.06434

<p align="center">
    <img src="http://eriklindernoren.se/images/dcgan2.png" width="640"\>
</p>
<p align="center">
    Figure: Generated handwritten digits.
</p>

### GAN
Implementation of a Generative Adversarial Network with an MLP generator and discriminator.

Reference: https://arxiv.org/abs/1406.2661

<p align="center">
    <img src="http://eriklindernoren.se/images/gan.png" width="640"\>
</p>
<p align="center">
    Figure: Generated handwritten digits.
</p>

### SGAN
Implementation of Semi-Supervised Generative Adversarial Network.

Reference: https://arxiv.org/abs/1606.01583

### WGAN
Implementation of Wasserstein GAN (with DCGAN generator and discriminator).

Reference: https://arxiv.org/abs/1701.07875

<p align="center">
    <img src="http://eriklindernoren.se/images/wgan2.png" width="640"\>
</p>
<p align="center">
    Figure: Generated handwritten digits.
</p>
