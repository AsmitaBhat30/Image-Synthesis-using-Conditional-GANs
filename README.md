# Image-Synthesis-using-Conditional-GANs

The aim of this project is to train a simple conditional GAN model to generate new images for different class labels of MNIST digits data (PART-1) and also interpolate the generated images to compare the significance of class label with their noise vector (PART-2). Furthermore, the feature vectors of these images and their labels are together evaluated in some latent dimension of a binary classifier - MLP(PART-3).


***PART-1***
GANS are very popular DL models for synthesizing new images. GAN models are taught how to synthesize new images from the underlying training images' distribution. After the generator is trained, it is ready to generate images on its own. These generated images can be from any class of images that we have trained on and will look similar to those of training images.

But is there a way to control how these images will look while they are being generated? Or, in other words, how do we synthesize images of a particular class or an image that describes a particular text description? 

GANs make this possible too! We can generate the images that we would like by conditioning the generator on a class label (if we want an image from a particular task) or on a text description (including captions).

In this project, I have developed a conditional GAN (CGAN) model on MNIST digit dataset to conditionally generate images of different classes. 

On completion of the training of the cGAN model, I have saved the trained generator model which can be loaded again to generate new images.

***PART-2***

Since, we also provide the class label along with the noise vector to our model, an important question here would be to know how much impact does the label have in comparison to the noise vector in this image generation task.

To find this out, I have linearly interpolated between two generated images of different classes.
 
The Linear Interpolation task is done with:
1. Only label
2. Only noise
3. Both noise and label

From the results obtained from linear interpolation, it is evident that the label has great impact on image generation. The interpolation where label is given works very well for interpolating between images of two classes whereas, interpolation with only noise has no impact.

***PART-3***

In this part I am also exploring the latent space of a binary neural network classifier that takes real images with matching and mis-matching labels and similarly matching and mis-matching samples for generated images. This classifier is trained to classify between matching image-label pairs and non-matching image label pairs irrespective of whether the images are real or generated. 

The intuition here is that the activations of an intermediate layer of the classifier in latent space will try to establish semantic relationship between the matching samples and that there doesn't exist any relationship between non-matching samples. We should not forget the fact that the matching samples include both real and generated images. So, the results will suggest how the model is treating the real and generated samples. It is expected that the model dosen't differentiate between the two which would be a good indication because it suggests that the model can't differente between real and generated samples and that the generator has done a good job in generating images.

To find if this stands true, I have calculated the average distances of these intermediate activations to show what pattern exists in the latent space. The results have been discussed below.


For this experiment, I used 1000 real images and 1000 generated images. 50% of the total samples were assigned correct class label and the remaining 50% were assigned any random incorrect class labels. Images together with their class labels form the input to the model. 

New binary labels are generated for this classification task. '1' for matching image and its correct class label and '0' for mismatched image and label pair.

Also, activations of an intermediate layer are taken. Then these activations are divided into four categories: 
1. Generated Images + Real Labels
2. Real Images + Real Labels
3. Generated Images + Fake Labels
4. Real Images + Fake Labels

The average distance between the activations of each category is computed and displayed. 

On an average, the binary classifier gives almost 99.9% accuracy to classify the matching samples from the non-matching ones. 
Also, I found out that the avearge distance of the real images irrespective of their labels is approx. same for both matching and non-matching data and the same stands true for generated images.
Additional scope to this project would be to try to compute distances of activations for different layers of the network and see if any other pattern exists.


###########################################################################################

The code for training a conditional GAN model can be found in Conditional GANs MNIST.ipynb file.

The code for Interpolation and Binary Classification for matching and non-matching image-label pair samples can be found in Interpolation and Binary Classifier.ipynb file.

