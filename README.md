# Keras-GAN

## About
Keras implementations of Generative Adversarial Network (GAN) models. Since I'm running these models from my Macbook Pro
they will be limited in their complexity and short training sessions will be prioritized.

## Table of Contents
- [Keras-GAN](#keras-gan)
  * [About](#about)
  * [Table of Contents](#table-of-contents)
  * [Implementations](#implementations)
    + [Auxiliary Classifier GAN](#acgan)
    + [Deep Convolutional GAN](#dcgan)
    + [Wasserstein GAN](#dcgan)
 
## Implementations   
### ACGAN
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

### WGAN
Implementation of Wasserstein GAN (with DCGAN generator and discriminator).

Reference: https://arxiv.org/abs/1701.07875

<p align="center">
    <img src="http://eriklindernoren.se/images/wgan2.png" width="640"\>
</p>
<p align="center">
    Figure: Generated handwritten digits.
</p>
