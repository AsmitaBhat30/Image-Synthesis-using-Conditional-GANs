# Image-Synthesis-using-Conditional-GANs

***PART-1***
As we know, GANs are used for synthesizing new images after training it on Real samples. But what if we want to synthesize images of a particular class? GANs make this possible too! We can generate images of a particular class label by conditioning the generator on class label(or on some text description).

In this project, I have developed a conditional GAN (CGAN) model on MNIST dataset. After training the cGAN, I save the trained generator model to generate new images of a any given class label.

***PART-2***
Also, how do we know if label has an impact on the image generation and if yes, how much in comparison with the noise vector? To find this out, I have linearly interpolated between two generated images of different classes. 
Linear Interpolation is done with:
1. Only label
2. Only noise
3. Both noise and label

It is evident from linear interpolation that the label has great impact on image generation. The interpolation with only label and both noise and label works well however, interpolation with only noise has no impact, which is obvious since it is any random vector.

***PART-3***
In this part, I tried to find out if the simple binary neural network classifier can successfully classify between the matching image-label pair and non-matching image-label pair (non-matching meaning, an image with any random label other than the real label for that image). For this experiment, I used 1000 real images and 1000 generated images. 50% of the total samples were assigned correct class label and the remaining 50% were assigned any random incorrect class. Images together with their class labels are the inputs to my model. 

New binary labels are generated. '1' for matching image and its correct class label and '0' for mismatched image and label pair.

Also, activations of an intermediate layer are taken. Then these activations are divided into four categories: 
1. Generated Images + Real Labels
2. Real Images + Real Labels
3. Generated Images + Fake Labels
4. Real Images + Fake Labels

The average distance between the activations of each category is computed and displayed. 

On an average, the binary classifier gives almost 99.9% accuracy to classify the matching samples from the non-matching ones. 
Also, I found out that the avearge distance of the real images irrespective of their labels is approx. same for both matching and non-matching data and the same stands true for generated images.
Additional scope to this project would be to try to compute distances of activations for different layers of the network and see if any other pattern exists.
I am open to any discussions as to why was there difference in the average distances of real and generated samples irrespective of their labels.

#####################################################################################################

The code for training a conditional GAN model can be found in Conditional GANs MNIST.ipynb file.

The code for Interpolation and Binary Classification for matching and non-matching image-label pairs can be found in Interpolation and Binary Classifier.ipynb file.

