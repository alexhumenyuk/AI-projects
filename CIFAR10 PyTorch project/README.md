# **CIFAR-10 Image Classification with PyTorch**

## **Description**
This deep learning project is an implementation of a Convolutional Neural Network (CNN) to classify images into the 10 classes available in the CIFAR-10 dataset. This is a paradigmatic use case for this type of network, which performs much better at this task than a regular artificial neural network.

## **Model Architecture**
This CNN contains two convolutional layers, one pooling layer, and three fully connected layers, the last of which has 10 neurons corresponding to the 10 possible classes in the dataset.

## **Performance**
With 10 epochs and only 2 minutes of training on a GTX 1080 Ti GPU, the model achieves a classification accuracy of 70.9%.

## **Usage**
The model also accepts downloaded images located in the same directory as the Jupyter notebook, although their filenames must be manually added in the last code cell. The code resizes and formats them so that the model can recognize them.

## **Test Images**
The two test images presented at the end of the script are included in the repository. The current accuracy of the model is enough to classify them correctly. If we want the model to recognize more complex images, more epochs should be used and dropout layers can be added to prevent overfitting (the code to include them is commented).
