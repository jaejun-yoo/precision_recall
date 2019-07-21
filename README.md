# Improved Precision and Recall Metric for Assessing Generative Models 

This project is done as a part of the deep learning interview stage for NAVER Corp. It is mostly based on the [paper](https://arxiv.org/pdf/1904.06991.pdf) with the same name. The paper provides an improved precision and recall metric for General Adversarial Networks(GANs) using a binary function and k-nearest neighbors. The interviewer asked to calculate the precision and recall metric defined in this paper by applying it to two sets of 10000 images from a dataset such as celebA or CIFAR-10, and then explain the precision and recall values alongside in-depth analysis.

## Calculation Process

Within the paper provided, the metrics were calculated by comparing a set of feature vectors from a distribution of real images and a distribution of generated images. The images were generated by using a StyleGAN provided by NVIDIA, and the feature vectors were extracted from a pre-trained VGG-16 classifier.

The project contains two Python files.

### calculate.py

The calculate.py file then extracts the feature vector from the image files using the pretrained model of VGG16. It then estimates the manifold using k-NN neighbors and returns the precision and recall value. The function pretrained_model in calculate.py makes use of the VGG16 from Keras (with additional layers : based on [this thread](https://github.com/keras-team/keras/issues/4465), which is used to extract feature vectors of each image. Using the feature vectors extracted from this step, we calculate the precision and recall value following the pseudo-code provided in Appendix A in the aforementioned [paper](https://arxiv.org/pdf/1904.06991.pdf). 

### generate_image.py 

The generate_image.py file generates an image that has been trained by the DCGAN explained above, and returns 10000 generated images. The whole code is based on [this notebook](https://github.com/naokishibuya/deep-learning/blob/master/python/dcgan_celeba.ipynb) with several adjustments, which proposes a Deep Convolutional Generative Adversarial Network using the same celebA dataset. 

## Results and Analysis

This author wanted to generate 10000 images from a DCGAN and then calculate the precision and recall values of a set of 10000 images from celebA and 10000 images generated by the DCGAN, but the time it took to train a DCGAN using approximately 200000 images from celebA, and then calculating the feature vectors for each 10000 images took more than 8 hours. Hence, for practical computational purposes, the author will calculate the precision and recall values of the first 20000 images in the celebA dataset. (The code to calculate the precision and recall values for the celebA dataset images and the images produced by the DCGAN is commented in calculate.py)

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
