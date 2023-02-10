ASL Sign Language to Speech


1.	Introduction

What is that person saying? American Sign Language is a natural language that serves as the predominant sign language of Deaf communities in the United States of America. ASL is a complete and organized visual language that is expressed by employing both manual and non manual features. However, with the advent of technology we can devise a more sophisticated solution which can translate this sign language into speech. In this project we aim to do so with the help of a Deep Learning Model trained to classify the images and then convert the classified image into speech. Also in the case of implementing these models efficiency with respect to time and performance becomes of utmost importance as we ought to achieve the output in reasonable time and cost. And with respect to performance we are trying to enhance the efficiency with help of CUDA in Pytorch which is explained in detail in the methodology section.

 ![image](https://user-images.githubusercontent.com/25953950/218148207-2f9710b4-e4ee-4747-b710-3238e9f00311.png)




2.	Methodology

Algorithms: We aim to design an image classification model using the Convolutional Neural Network Architecture in order to train the model to be able to classify all the signs appropriately. Although, we are thinking of a more appropriate approach in order to implement this solution. In our opinion it would make more sense to use transfer learning and save time to build the model. As with transfer learning we are only left with the task to load the pretrained model and modify the last layer into the required number of classification as per our problem statement.

Parallel Architecture: CUDA enables us to perform compute-intensive operations faster by parallelizing tasks across GPUs. We are planning to implement parallel architecture for both the sheer amount of data and the model through DataParallel and DistributedDataParallel.

DataParallel
Data parallelism is parallelization across multiple processors in parallel computing environments. It focuses on distributing the data across different nodes, which operate on the data in parallel. 

The implementation of DataParallel is illustrated in figure below - 

![image](https://user-images.githubusercontent.com/25953950/218149098-bb683db8-5392-4b16-b8e4-76f94a72db54.png)

In general, DataParallel parallelizes the application of the given module by splitting the input across the specified devices by chunking in the batch dimension. In the forward pass, the module is replicated on each device, and each replica handles a portion of the input. During the backward pass, gradients from each replica are summed into the original module.

Results for DataParallel method have been evaluated for 2 different models (ResNet, inceptionv3) both of them are pre-trained models. For the ResNet model we have plotted the time taken as depicted in the below chart.
 
 ![image](https://user-images.githubusercontent.com/25953950/218149848-b1db73bf-05c0-4150-86c3-52e5332dc0ef.png)

From the trend it is clearly evident that parallelization thus improves the efficiency and reduces the time taken by model to yield the results but after certain increase in number of GPUs the trend is observed to be slightly decreasing on every increase in Number of GPU. Hence, it makes more sense to provide not more than 2 GPUs in this case to our ResNet model.

DistributedDataParallel
DistributedDataParallel (DDP) implements data parallelism at the module level which can run across multiple machines. Applications using DDP should spawn multiple processes and create a single DDP instance per process. DDP uses collective communications in the torch.distributed package to synchronize gradients and buffers. DDP uses multi-process parallelism, and hence there is no GIL contention across model replicas

To use DDP on a host of N GPUs, we spawn up N processes, ensuring that each process exclusively works on a single GPU from 0 to N-1.

The graph shows the time taken to train the model on complete dataset using different number of GPU’s

![image](https://user-images.githubusercontent.com/25953950/218150086-5975ef20-b72a-45cc-a28f-0490c0d717c6.png)

Inference - As the number of GPUs increases the time taken by the model also decreases. The results for both the procedures that we have implemented in our project show similar trend in decline in the time taken as the number of GPUs increases except for the latter method where there is a slight increase in the time when GPU = 4 from time when GPU = 3 which can be ignored as the model’s performance can be  considered equivalent.

Analysing the results for different models

![image](https://user-images.githubusercontent.com/25953950/218150449-faa37a82-4ff9-4f3d-8997-67767375ba51.png)


3.	Description of the dataset

The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26 are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING.
 
 ![image](https://user-images.githubusercontent.com/25953950/218148256-382a244c-e922-4e97-acb3-9a67eb49b016.png)

These 3 classes are very helpful in real-time applications, and classification.
The test data set contains a mere 29 images, to encourage the use of real-world test images.


4.	Data Sources (download link)

The dataset is taken from a kaggle problem which can be found here -> https://www.kaggle.com/datasets/grassknoted/asl-alphabet?select=asl_alphabet_train


