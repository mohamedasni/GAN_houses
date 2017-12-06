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
You need a few different libraries to run all the models:
* python 3.6
* tensorflow
* keras
* numpy
* matplotlib
* keras_adversarial

To install keras_adversarial, please follow the directions at this location: https://github.com/bstriner/keras-adversarial

## How to run Autoencoder
To run the autoencoder, you simply need to have the file `train.csv` in the same folder as autoencoder.py
Make sure tensorflow and keras are installed properly and are recognised by the python environment.
After that, you should be able to simply run the script like this:
`python autoencoder.py`
The model will then compile and train.

## How to run MNIST GAN

## How to run GAN Houses V1
To run the V1 House GAN, you need to already have ran the autoencoder because the autoencoder creates the proper training and validation files. After the files are there, you should be able to run the House GAN V1 like so:
`python gan_houses_v1.py`

## How to run GAN Houses V2
This is one a little trickier. You also have to make sure the training and validation data are in the current working directory. Then you have to make sure that keras_adversarial as been installed correctly. To do so, make sure you install keras_adversarial using the proper environment if using Anaconda. To verify if it is install correctly, open a simple python interpreter and run the following command :
`import keras_adversarial`
If python cannot find the module, there is a problem with your installation.

After that, you can run the script using the same command:
`python gan_houses_v2.py`

## How we ran these models
We ran all the following models in the PyCharm IDE, we used a specialy crafted Anaconda environment. 
We have also added the yml file that describes all the installed modules in our environment. Please refer to: 
`tensorsimo.yml`
