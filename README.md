# session8
Session8

## Description
This repository CIFAR10 dataset is trained with a 
ake this network:
* C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10,
*   Keep the parameter count less than 50000
* One layer is added to antoher i.e. x + conv(x)
* Training is done with three different normalization i. Batch Normalization ii. Layer Normalization iii. Group Normalization
* Random Crop is used for augmentation
* 10 misclassified images are displayed for Batch normalization
* Various plots (training accuracy, test accuracy, test loss are plotted)

## Training Accuracy of Models

* Batch Normalization: 76.92
* Layer Normalization: 76.96
* Group Normalization: 76.08

## Test Accuracy of Models

* Batch Normalization: 78.59
* Layer Normalization: 79.12
* Group Normalization: 78.33

## Observation of normalization techniques
- Batch normalization makes the neural network training more efficient and stable.
- Layered normalization is used to normalize the distributions of intermediate layers. It enables smoother gradients, faster training, and better generalization accuracy
- From the training and test accuracy for the network used, for all the three techniques have similar perfomance
- 


## 10 misclassified images with Batch Normalization

![alt text](https://github.com/monimoydebp/session8/blob/main/misclassified_image_list.png)

## Epoch vs Train Acccuracy Plot

![alt text](https://github.com/monimoydebp/session8/blob/main/epoch_vs_train_accuracy.png)

## Epoch vs Test Loss Plot

![alt text](https://github.com/monimoydebp/session8/blob/main/epoch_vs_test_loss.png)


## Epoch vs Test Accuracy Plot

![alt text](https://github.com/monimoydebp/session8/blob/main/epoch_vs_test_accuracy.png)
