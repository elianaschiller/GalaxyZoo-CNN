# GalaxyZoo-CNN
A convolutional neural network which classifies galaxy morphology (either elliptical or disk shaped). 

## Description
I tested many different network architectures, but I've only included my final model here, which achieved an 86.5% accuracy. This is a good test accuracy for galaxy morphology, which is relatively qualitative and difficult to definitively classify. My final CNN is a relatively simple model, with only 5 convolutional layers, and one data sample that I split into training, validation, and test samples. I reduced overfitting using dropout layers, as well as image augmentation. 

The code here includes the model architecture itself, and the training + testing code. Running the model architecture code creates checkpoints with each epoch trained, and saves the best checkpoint as a model file to use in testing. However, I've also included the best model file I generated while testing (`CNN_Model_Checkpoints7.keras`), which you can use if you'd like to obtain the same test accuracy I did. I trained my model for <4 hours, which was sufficient to obtain a good test accuracy, but it is possible to reduce overfitting further and train for many more epochs/hours. 

## Getting Started 

### Dependencies
This code uses the Tensorflow machine learning package with the keras package embedded in it ([installation instructions here](https://www.tensorflow.org/install)). If your laptop has GPU, try to install tensorflow with GPU support. You'll also need to install NumPy, Pandas, SciPy, and scikit-learn. 

### The Data
I trained this network on 61,578 galaxy images from Kaggle's Galaxy Zoo competition, linked [here](https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge). I only used the zip file of training images, called `images_training_rev1.zip`, which should go in a directory called `images_train_rev1`. 

The table of galaxy IDs and the probabilities for the morphology of each galaxy is in the CVS formatted table `training_solutions_rev1.csv`. You will need to alter the path of this file in your code. 

## Acknowledgement
- [a-kravtsov](https://github.com/a-kravtsov)

