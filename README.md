# Improved Precision and Recall Metric for Assessing Generative Models 

This project is done as a part of the deep learning interview stage for NAVER Corp. It is based on the following [paper](https://arxiv.org/pdf/1904.06991.pdf) which provides an improved precision and recall metric to measure the performance of a General Adversarial Network(GAN) using k-nearest neighbors. The interviewer asked to calculate the precision and recall metric defined in this paper by applying it to two sets of 10000 images from a dataset such as celebA or CIFAR-10, and then explain the precision and recall values alongside in-depth analysis.

## Calculation Process

Within the paper provided, the metrics were calculated by comparing a set of feature vectors from a distribution of real images and a distribution of generated images. The images were produced via a StyleGAN provided by NVIDIA, and the feature vectors were extracted from a pre-trained VGG-16 classifier. In this project, we try to replicate the paper's experiments, but we use a Deep Convolutional Generative Adversarial Network (DCGAN) instead of a StyleGAN. 

The project contains two Python files.

### calculate.py

The calculate.py file returns the precision and recall values of two image sets. The function pretrained_model in calculate.py makes use of the VGG16 using Keras (with additional layers : based on [this thread](https://github.com/keras-team/keras/issues/4465)), which is used to extract feature vectors of each image. Using the feature vectors extracted from this step, the manifold_estimator function tries to estimate the true manifold of the data using k-nearest neighbors, and returns the fraction value of feature vectors that lie within the estimated manifold. The knn_precision_recall function then calculates the precision and recall value of the real and generated images. These functions were designed by following the pseudo-code provided in Appendix A in the aforementioned [paper](https://arxiv.org/pdf/1904.06991.pdf). 

### generate_image.py 

The generate_image.py file generates a number of images that has been trained by a DCGAN, and returns 10000 generated images. The whole code is based on [this notebook](https://github.com/naokishibuya/deep-learning/blob/master/python/dcgan_celeba.ipynb) with several adjustments, which utilizes a DCGAN using the same celebA dataset. 

## Results and Analysis

This author wanted to generate 10000 images from a DCGAN and then calculate the precision and recall values between the set of 10000 images from celebA and the 10000 generated images. However, the time expended to train a DCGAN using approximately 200000 images from celebA, and then calculating the feature vectors for each of the 10000 images took more than 8 hours. Hence, for practical computational purposes, the author will calculate the precision and recall values of the first 20000 images in the celebA dataset. (The code to calculate the precision and recall values for the celebA dataset images and the images generated by the DCGAN is commented in calculate.py)

By definition, precision is the fraction of the generated images that are realistic, and recall is the fraction of real images within the manifold covered by the generator. In other words, a high precision value indicates a high quality of generated samples, and a high recall value implies that the generator can generate many realistic samples that can be found in the "real" image distribution. 

An exemplary example of illustrating precision and recall using a toy dataset is provided by [this paper](https://arxiv.org/pdf/1711.10337.pdf) in Figure 2, which also explores evaluation measures of a GAN. 

The precision and recall code has been repeated several times, using the first 10000 images as the "real images" and the next 10000 images as the "generated images". Overall the results return consistent values of precision and recall : the precision value lies between 0.6 ~ 0.7, and the recall value lies between 0.5 ~ 0.6. A fairly high value of precision implies that the "generated image" set contains a high quality of images, which is expected as we are using the images of celebA as a substitute of the "generated images". A considerable recall value can be attributed to the variation within the images in celebA, as often variation within a dataset also improves recall. 

In practice, however, it is not easy to provide a clear interpretation of the values, as we are using images from the same celebA dataset as a substitute for the "generated" images due to the time constraints of this project and computational limitations.

## Conclusion

There is no one exclusive way to measure the performances of a Generative Adversarial Network that has been defined yet. However, the precision and recall calculation method provided within this [paper](https://arxiv.org/pdf/1904.06991.pdf) shows strong promise of further developments in assessing generative models. This project tries to replicate what the improved precision and recall metric provided within the paper, and calculates the values of precision and recall using the celebA dataset. 


## References 

* Additional Papers : https://papers.nips.cc/paper/7769-assessing-generative-models-via-precision-and-recall.pdf
