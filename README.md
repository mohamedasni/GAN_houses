# Generative Adversarial Network For Houses

## Description

Use generative adversarial neural network (GAN) in order to create a model that is able to generate realistic houses with realistic features.

We started by creating a GAN that could generate hand writting numbers based on the MNIST data. This was a stepping stone for us to creating our own model that would generate realistic houses with realistic prices and features.

Before attempting to create a GAN for houses and their features, we needed to treat our data set first. To do so we created an autoencoder that not only encoded the data but also normalized it. Once we completed this step we were now prepared to create our GAN for houses and their features.

Our first version of such a GAN did in fact work and did in fact generate houses. However, we were not satisfied with the results. After certain optimizations we developed our second version of our GAN. You can see our results in `./gan_houses_v2/result.txt` and `./gan_houses_v2/figure_1.png`.

* Autoencoder: `./autoencoder/`
* MNIST data GAN: `./gan_mnist/`
* Houses GAN version 1: `./gan_houses_v1/`
* Houses GAN version 2: `./gan_houses_v2/`

## Sources

* https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/
* https://github.com/bstriner/keras-adversarial
* Tensorflow documentation
* Keras documentation

## Setup

## How to run Autoencoder

## How to run MNIST GAN

## How to run GAN Houses V1 & V2
