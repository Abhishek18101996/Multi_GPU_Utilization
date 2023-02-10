ASL Sign Language to Speech


1.	Introduction

What is that person saying? American Sign Language is a natural language that serves as the predominant sign language of Deaf communities in the United States of America. ASL is a complete and organized visual language that is expressed by employing both manual and non manual features. However, with the advent of technology we can devise a more sophisticated solution which can translate this sign language into speech. In this project we aim to do so with the help of a Deep Learning Model trained to classify the images and then convert the classified image into speech. Also in the case of implementing these models efficiency with respect to time and performance becomes of utmost importance as we ought to achieve the output in reasonable time and cost. And with respect to performance we are trying to enhance the efficiency with help of CUDA in Pytorch which is explained in detail in the methodology section.

 ![image](https://user-images.githubusercontent.com/25953950/218148207-2f9710b4-e4ee-4747-b710-3238e9f00311.png)




2.	Methodology

Algorithms: We aim to design an image classification model using the Convolutional Neural Network Architecture in order to train the model to be able to classify all the signs appropriately. Although, we are thinking of a more appropriate approach in order to implement this solution. In our opinion it would make more sense to use transfer learning and save time to build the model. As with transfer learning we are only left with the task to load the pretrained model and modify the last layer into the required number of classification as per our problem statement.

Parallel Architecture: CUDA enables us to perform compute-intensive operations faster by parallelizing tasks across GPUs. We are planning to implement parallel architecture for both the sheer amount of data and the model through Data Parallelism and Model Parallelism.

For Data Parallelism we will be using the same model in every thread but supply different chunks of data every time. This implies that data parallelism will use the same weights for training on the same thread however, on separate batches of data every time. This will help us fine-tune the model as per the image data and improve efficiency as well as save time.

For Model Parallelism we will be splitting the model among different threads. This will split the weight of the network equally among the threads and all the threads work on a single-mini batch of data. Here the output generated after every layer will be stacked and synchronized and provide an input to the next layer.


3.	Description of the dataset

The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26 are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING.
 
 ![image](https://user-images.githubusercontent.com/25953950/218148256-382a244c-e922-4e97-acb3-9a67eb49b016.png)

These 3 classes are very helpful in real-time applications, and classification.
The test data set contains a mere 29 images, to encourage the use of real-world test images.


4.	Data Sources (download link)

The dataset is taken from a kaggle problem which can be found here -> https://www.kaggle.com/datasets/grassknoted/asl-alphabet?select=asl_alphabet_train


