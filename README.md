# Keras-GAN

## About
Keras implementations of Generative Adversarial Networks (GANs) suggested in research papers. If dense layers produce reasonable results for a given model I will often prefer them over convolutional layers. The reason is that I would like to enable people without GPUs to test these implementations out. These models are in some cases simplified versions of the ones ultimately described in the papers, but I have chosen to focus on getting the core ideas covered instead of getting every layer configuration right. However, because of this the results will not always be as nice as in the papers.

## Table of Contents
- [Keras-GAN](#keras-gan)
  * [About](#about)
  * [Table of Contents](#table-of-contents)
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [Auxiliary Classifier GAN](#ac-gan)
    + [Adversarial Autoencoder](#adversarial-autoencoder)
    + [Bidirectional GAN](#bigan)
    + [Boundary-Seeking GAN](#bgan)
    + [Conditional GAN](#cgan)
    + [Context-Conditional GAN](#cc-gan)
    + [Context Encoder](#context-encoder)
    + [Coupled GANs](#cogan)
    + [CycleGAN](#cyclegan)
    + [Deep Convolutional GAN](#dcgan)
    + [DualGAN](#dualgan)
    + [Generative Adversarial Network](#gan)
    + [InfoGAN](#infogan)
    + [LSGAN](#lsgan)
    + [Pix2Pix](#pix2pix)
    + [Semi-Supervised GAN](#sgan)
    + [Wasserstein GAN](#wgan)

## Installation
    $ git clone https://github.com/eriklindernoren/Keras-GAN
    $ cd Keras-GAN
    $ sudo pip3 install -r requirements.txt

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

### BGAN
Implementation of Boundary-Seeking Generative Adversarial Networks.

[Code](bgan/bgan.py)

Paper: https://arxiv.org/abs/1702.08431

### CC-GAN
Implementation of Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks.

[Code](ccgan/ccgan.py)

Paper: https://arxiv.org/abs/1611.06430

<p align="center">
    <img src="http://eriklindernoren.se/images/ccgan.png" width="640"\>
</p>

### CGAN
Implementation of Conditional Generative Adversarial Nets.

[Code](cgan/cgan.py)

Paper:https://arxiv.org/abs/1411.1784

<p align="center">
    <img src="http://eriklindernoren.se/images/cgan.gif" width="640"\>
</p>

### Context Encoder
Implementation of Context Encoders: Feature Learning by Inpainting.

[Code](context_encoder/context_encoder.py)

Paper: https://arxiv.org/abs/1604.07379

<p align="center">
    <img src="http://eriklindernoren.se/images/context_encoder.png" width="640"\>
</p>

### CoGAN
Implementation of Coupled generative adversarial networks.

[Code](cogan/cogan.py)

Paper: https://arxiv.org/abs/1606.07536

### CycleGAN
Implementation of Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.

[Code](cyclegan/cyclegan.py)

Paper: https://arxiv.org/abs/1703.10593

<p align="center">
    <img src="http://eriklindernoren.se/images/cyclegan.png" width="640"\>
</p>

```
$ cd cyclegan
$ bash download_dataset.sh apple2orange
$ python3 cyclegan.py
```   

<p align="center">
    <img src="http://eriklindernoren.se/images/cyclegan_gif.gif" width="640"\>
</p>



### DCGAN
Implementation of Deep Convolutional Generative Adversarial Network.

[Code](dcgan/dcgan.py)

Paper: https://arxiv.org/abs/1511.06434

<p align="center">
    <img src="http://eriklindernoren.se/images/dcgan2.png" width="640"\>
</p>

### DualGAN
Implementation of DualGAN: Unsupervised Dual Learning for Image-to-Image Translation.

[Code](dualgan/dualgan.py)

Paper: https://arxiv.org/abs/1704.02510

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

<p align="center">
    <img src="http://eriklindernoren.se/images/infogan.png" width="640"\>
</p>

### LSGAN
Implementation of Least Squares Generative Adversarial Networks.

[Code](lsgan/lsgan.py)

Paper: https://arxiv.org/abs/1611.04076

### Pix2Pix
Implementation of Unpaired Image-to-Image Translation with Conditional Adversarial Networks.

[Code](pix2pix/pix2pix.py)

Paper: https://arxiv.org/abs/1611.07004

<p align="center">
    <img src="http://eriklindernoren.se/images/pix2pix_architecture.png" width="640"\>
</p>

```
$ cd pix2pix
$ bash download_dataset.sh facades
$ python3 pix2pix.py
```   

<p align="center">
    <img src="http://eriklindernoren.se/images/pix2pix2.png" width="640"\>
</p>

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
