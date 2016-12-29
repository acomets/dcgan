# Unsupervised Representational Learning with Deep Convolutional Generative Adversarial Networks

#### This work was conducted in the context of a group project for the Neural Networks & Deep Learning class at Columbia University, Dept. of Electrical Engineering (https://sites.google.com/site/mobiledcc/documents/neuralnetworksanddeeplearning)

#### Team members: Antoine Comets, Gary Sztajnman (https://github.com/gary159)

Deep Convolutional Generative Adversarial Networks project on MNIST and LFW datasets

### Abstract

This paper presents the results of deep convolutional generative adversarial networks (DCGANs) developed in Theano and tested on a dataset of images of handwritten digits, faces and restaurants. We based our work on previous paper by Radford and Al, and aimed to replicate their results when training our model on new datasets : MNIST handwritten digits, Labeled Faces in the Wild (LFW) and the LSUN restaurant dataset. We were able to successfully replicate some results, e.g. smooth transitions when doing linear interpolations in therepresentations space and we believe our model could generate accurate images of restaurant if we let it run for a higher number of epochs.

### 1. Introduction

Learning useful feature representations from large datasets of unlabeled data has seen important researchinterest, especially in the context of computer vision. Classic approaches to unsupervised representationlearning include hierarchical clustering methods [2] or also training stacked autoencoders [3].

Generative networks can prove powerful in learning features from a dataset of images and generating imageswith those features. A major breakthrough [4] made training generative networks much easier: applyingadversarial training by simultaneously training a generator and a discriminator model. The idea is that thediscriminator model learns to determine whether a sample is from the true data distribution or the generator model distribution, while the generator model learns to produce better and better output to fool the discriminator.

### 2. Summary of the Original Paper#### 2.1 Methodology of the Original PaperIn the paper [5] Radford et al. propose a more stable set of architectures for training generative adversarialnetworks (GANs) and try to visualize some of the representations learned to illustrate their performance onunsupervised learning tasks.

#### 2.2 Key Results of the Original Paper

First of all, the set of constraints on the architectural topology of Convolutional GANs (DCGANs) that Radford et al. [5] give appears to make them stable to train in most settings.Representations learned by the discriminator can be used as feature extractors on supervised learning taskswhen labeled data is scarce. Radford et al. [5] showed that using GANs as a feature extractor on CIFAR-10 achieves 82.8% accuracy, outperforming all K-means based approaches, and it achieves state-of-the-art at 22.48% test error on SVHN.
Radford et al. also provide evidence, by walking on the representations manifold that is learnt, that the model does not seem to memorize training examples since there are no sharp transitions and generated images change smoothly when plotting linear interpolations of input vectors.Besides, the paper also shows visualizations of the discriminator features learned by doing guidedbackpropagation. It also illustrates manipulations of the generator representations by making the model “forget” to draw certain objects, e.g. a window.
Finally, Radford et al. provide an example of vector arithmetics in the representation space. They demonstratethat simple arithmetic operations reveal rich linear structure in representation space and link them to semantic properties of the generated images.

### 3. MethodologyOur approach was to implement the architecture guidelines for Deep Convolutional GANs and test thecode on datasets other than those that were used in the paper. In particular, we replicated walking in the latent space to observe smooth transitions and simple arithmetics on vectors in representation space.
#### 3.1. Objectives and Technical ChallengesFirst we implemented DCGANs, but we were faced with several technical hurdles. We originally tried to reuse code from tutorials and homeworks from the class, defining a new class for our DCGAN model. However, since the input to the discriminator in a GAN is sometimes true data from the training set and sometimes the output of the generator, creating a class was not tractable. Besides, one challenge in designing training adversarial networks is correctly defining the parameter update rules. Therefore we eventually decided to use functions for the generator and discriminator, which returned the outputs of the respective models.
Then we focused on replicating the linear interpolations in the representation space on MNIST and vector arithmetics on faces from Labeled Faces in the Wild (LFW).

#### 3.2. Problem Formulation and Design
The goal is to learn the generator’s distribution $p_g$ over data $\mathscr{x}$. We first define a prior on input noise $p_z(z)$ then represent the generator function by a mapping $G(z; θ_g)$, where $G$ is a differentiable function represented by the generative CNN with parameters $θ_g$ . Similarly we define the discriminator CNN by a function $D(\mathscr{x} ; θ_d)$ with a scalar output. $D(\mathscr{x})$ represents the probability that $\mathscr{x}$ came from the true training data set rather than $p_g$ . We train $D$ to maximize the probability of assigning the correct label to both training examples and samples from $G$. We simultaneously train $G$ to minimize $log(1 − D(G(z)))$. In other words, we optimize the following minimax problem:

\[ \min_G \max_D V(D,G) = \mathbb{E}_{\mathscr{x} \sim p_\mathrm{data}(\mathscr{x})} [\log D(\mathscr{x})] + \mathbb{E}_{\mathscr{x} \sim p_z(z) [\log (1 - D(G(z)))] \]

This objective function results in a saddle point that is a maximum for D and a minimum for G and thereforetraining may be very unstable.
Since the output layer is a sigmoid layer, our objective function uses the binary cross-entropy.Besides, we followed the architecture guidelines provided in the paper and explained below.

### 4. ImplementationWe will provide details about the architecture and specifications of our model then discuss our software implementation.
#### 4.1. Deep Learning Network
We followed the guidelines provided in the paper [5]:

1. As discussed in the All convolutional net [6], we replace pooling functions in convolutional layers by strided convolutions, with integer or fractional stride corresponding to downsampling in the discriminator, and upsampling in the generator2. We endeavored to use as few fully-connected layers as possible, only in the first layer of the generator to connect the 100 dimensional input vectors.3. We used batch normalization to stabilize learning by normalizing the input to each unit to have zero mean and unit variance, in all layers except the generator output layer and the discriminator input layer.4. We used the rectified linear unit (ReLU) activation function in all layers of the generator except the output layer where we used tanh, and used leaky ReLU activation in all layers of the discriminator except the output layer which is a single sigmoid output.

The architecture of the network is shown on fig. 1.

![alt text](https://openai.com/assets/research/generative-models/gencnn-afe135ff8d2725325a22455a488562b0e1cb7ac6a3f60b3cecb373fd043eb202.svg "DCGAN Generator architecture")

Fig. 1 DCGAN Generator architecture used for LSUN and LFW

#### 4.2. Software Design
We coded the model in Theano. First we defined a generator and discriminator functions, taking the parameters – here only weights since we scaled the images to the range of the tanh activation function in the discriminator input layer and we apply batchnorm at each layer – as inputs, comprising all the layers and returning their respective outputs.
We also coded – or used pre-existing – functions for loading the data from the various datasets.

Finally, we wrote a training function, starting from the one we used in class and in homeworks and defining all the weight parameters of the model, our cost functions for the generator and discriminator and our update rules. Besides we used an adam optimizer that we used in previous homeworks and we define two theano functions for training the discriminator and the generator, and applying the relevant updates.

### 5. Results
#### 5.1. Project ResultsOn MNIST, we generated satisfactory output. We observed that the cost of the discriminator was oscillating while the cost of the generator came down. Indeed, the discriminator can easily tell the difference between noise and true images at the beginning of the training but then the generated images become more and more plausible, while at the same time it has seen many images and it has become better at differentiating the two. The images we generate after training for 60 epochs are shown on fig. 2.

![alt text](https://github.com/acomets/dcgan/img/mnist_generated.png “Randomly generated outputs on MNIST”)

Fig. 2. Randomly generated outputs from our DCGAN on the MNIST dataset.

Besides, when we do linear interpolation of vectors in the 100-dimensional representations space, we obtain relatively smooth transitions, shown on fig. 3.

![alt text](https://github.com/acomets/dcgan/img/mnist_linear_interpolation.png “Linear interpolations on MNIST”)

Fig. 3. Linear interpolations between vectors in the representations space.

Our output on the LFW faces were less visually successful and realistic. See example below. Unfortunately we were unable to try the vector arithmetics at this stage.

![alt text](https://github.com/acomets/dcgan/img/faces_generated.png “Randomly generated faces”)

Fig. 4. Examples of randomly generated faces after 128 epochs of training on LFW

#### 5.2. Comparison of Results
In our paper, the smooth transitions were shown on sample images of bedrooms from the LSUN dataset, so we searched for other papers online with similar results on handwritten digits. Fig. 6 below shows results from [4].

![alt text](https://github.com/acomets/dcgan/img/lin_int_original1.png “Linear interpolations from original paper”)
![alt text](https://github.com/acomets/dcgan/img/lin_int_original2.png “Linear interpolations from original paper”)


Fig. 6. Linear interpolations between vectors in the representations space from Goodfellow et al. [4]

#### 5.3. Discussion of Insights GainedThis project made us learn many concept on Generative networks and adversarial training. Besides we discovered many applications of generative models for example in semi-supervised and supervised learning tasks, filters of the discriminator can be used as feature extractors to improve the performance of classifiers when labeled data is scarce.
#### 6. Conclusion
This project was a great complement to the work we did in class on neural networks. Building on our approach of CNNs and reconstruction of noisy input images (dropped out pixels), we discovered the powerful technique of adversarial training on some concrete examples.
We successfully implemented a DCGAN in Theano and overcame the challenges of this type of models by defining it with fonctions instead of classes.### 7. Acknowledgement
We thank Mehmet Kerem Turkcan for his time and support, and for sharing some of his previous difficulties training GANs.### 8. References
[1] https://bitbucket.org/e_4040_ta/e4040_project_gsac[2] A. Coates and A. Y. Ng, “Learning feature representations with k-means”, in Neural Networks: Tricks of the Trade, pp. 561–580. Springer, 2012.
[3] P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, and P.-A. Manzagol, “Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion”, The Journal of Machine Learning Research, 11:3371–3408, 2010.[4] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, et al. “Generative adversarial nets”, NIPS 2014, Département d’Informatique et de Recherche Opérationnelle, Université de Montréal.[5] A. Radford, L. Metz, S. Chintala, “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks”, Under review for ICLR 2016, January 7, 2016.[6] J. T. Springenberg, A. Dosovitskiy, T. Brox, M. Riedmiller, “Striving For Simplicity: The All Convolutional Net”, ICLR 2015, Department of Computer Science, University of Freiburg, April 13, 2015.[7] T. Salimans, I. J. Goodfellow, W. Zaremba, V. Cheung, A. Radford, X. Chen, “Improved Techniques for Training GANs”, June 10, 2016.

