Description of all the files and dataset


1. Link of the dataset - https://www.kaggle.com/datasets/grassknoted/asl-alphabet
2. asl_classification_ddp.py - Python file to train the models using ddp module for different GPUs
To run the file  -  ‘python asl_classification_ddp.py alexnet 4’

3. asl_DP_resnet.py - Python file to train the resnet models on the dataset for different GPUs using DP module
4. asl_DP_inception.py - Python file to train the inception models on the dataset for different GPUs using DP module
5. ASL_TransferLearning_final.ipynb - Python notebook to show performance of different models using DP modules on sample of 10k datapoints.
6. asl_streamlit.py - Python file to load the model and host it on the server so that user can upload a image on we browser and gets the predicted class.
7. DP_10k_images_multiGPUs - This module consists different ipynb notebooks for DP module for different number of GPUs.

Introduction
I. Background
The 2 methodologies we came across and decided to move forward with for our analysis are as   following:
DataParallel
Data parallelism is parallelization across multiple processors in parallel computing environments. It focuses on distributing the data across different nodes, which operate on the data in parallel. 

The implementation of DataParallel is illustrated in figure below - 

 

In general, DataParallel parallelizes the application of the given module by splitting the input across the specified devices by chunking in the batch dimension. In the forward pass, the module is replicated on each device, and each replica handles a portion of the input. During the backward pass, gradients from each replica are summed into the original module.

DistributedDataParallel
DistributedDataParallel (DDP) implements data parallelism at the module level which can run across multiple machines. Applications using DDP should spawn multiple processes and create a single DDP instance per process. DDP uses collective communications in the torch.distributed package to synchronize gradients and buffers. DDP uses multi-process parallelism, and hence there is no GIL contention across model replicas

To use DDP on a host of N GPUs, we spawn up N processes, ensuring that each process exclusively works on a single GPU from 0 to N-1.

The below figure illustrates the implementation of DDP for a pre-defined model(in this case a ToyModel) - 

 




II. Motivation
What is that person saying? American Sign Language is a natural language that serves as the predominant sign language of Deaf communities in the United States of America. ASL is a complete and organized visual language that is expressed by employing both manual and non manual features. However, with the advent of technology we can devise a more sophisticated solution which can translate this sign language into speech.
 
In this project we aim to do so with the help of a Deep Learning Model trained to classify the images and then convert the classified image into speech. Also to enhance the model’s performance and it’s efficiency we are implementing this project using DataParallelism and DistributedDataParallelism techniques taught in the class.
III. Goals: 
Implementing Parallel architecture with the help of CUDA. CUDA enables us to perform compute-intensive operations faster by parallelizing tasks across GPUs. We are planning to implement parallel architecture for both the sheer amount of data and the model through Data Parallelism and Model Parallelism.
For Data Parallelism we will be using the same model in every thread but supply different chunks of data every time. This implies that data parallelism will use the same weights for training on the same thread however, on separate batches of data every time. This will help us fine-tune the model as per the image data and improve efficiency as well as save time.
For Model Parallelism we will be splitting the model among different threads. This will split the weight of the network equally among the threads and all the threads work on a single-mini batch of data. Here the output generated after every layer will be stacked and synchronized and provide an input to the next layer.
IV. Methodology
With the goal being set for the project to classify images appropriately and improve model’s performance using parallel architecture we have implemented 2 approaches to do so. 
1.	Using Pythorch’s DataParallel module
2.	Using Pythorch’s DataDistributedParallel module
The steps followed in this project is described below - 
a.	Description of Dataset
The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26 are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING.
 
These 3 classes are very helpful in real-time applications, and classification. The test data set contains a mere 29 images, to encourage the use of real-world test images.

 


b.	Preparing the Dataset
The dataset used for this project consists of images which are required to be converted into a desirable computationally efficient format in order to be fed as input to our model. We have built a helper function to do so called prepare_dataset() as shown in the image below. 
 
As part of preprocessing the data we first apply image transformation to convert the input images into tensors and further normalization to remove anomalies and eliminate transitive dependency if any. And in order to efficiently train our model later we sample the data and define training and validation subsets.
c.	Building the Model
As discussed we have implemented 2 parallel approaches to train our model as with DataParallel and DistributedDataParallel. In both the cases, we have defined the model with the help of a helper function described in below sub-modules.
i) Using Pythorch’s DataParallel module
The model_loader() function takes the batch_size and data_subsets as arguments. The function first samples the data as per batch_size and loads them as per training and validation data loader.
 
Then we initialize our model and convert the model using torch’s DataParallel class by passing the model as the object resnet = nn.DataParallel(resnet) Adn. Once the model has been instantiated we modify the model’s fully connected(fc) layer in order to yield output into the required number of categories, in this case the length of the classes variable as seen from the code. 
ii) Using Pythorch’s DataDistributedParallel module
Before using the data preprocessed from the previous step we first Data Parallelism using the DataLoader() function from torch. By adjusting num_workers parameter we can define in how many partitions the data should be divided and then each partition will be given to a single GPU at a time for processing. After defining the model we implement Model Parallelism using the DistributedDataParallel class from torch.parallel module. Once the model is initialized we re-define the final fully-connected(fc) layer of the model as per our requirement. We can also define different models and modify their final layers as per their architecture.
 
d.	Training the Model
Now, we are required to train the model. And the training process for every model varies subjectively. First we define the training procedure for DataParallel Mechanism as follows.
i) Training of model using DataParallel module (for 10k samples)
We have defined the train() helper function in order to train the model. We receive all the hyper parameters and arguments to train the model from the calling main function. The training procedure is simple and straightforward. We also record time taken each time whenever we increase the number of GPUs per session to run our code and compare the model’s performance. 
  
The detailed analysis of the time taken under different numbers of GPUs is illustrated in the below chart with resnet Model.

For number of GPU = 1 
 
For number of GPU = 2 
 
For Number of GPU = 3
 
For Number of GPU = 4
 
The detailed analysis of the time taken under different numbers of GPUs is illustrated in the below chart with resnet Model.
For GPU = 1, 2, 3, 4 as illustrated in below chart.
GPU = 1  						GPU = 2
  



GPU = 3						GPU = 4
  
As evident from the above results we can infer that as the number of GPUs increases the time taken by the model also decreases.
Note: The implementation of all the models using DP was performed on different instances of Discovery cluster with number of GPUs specified as (1,2,3,4) at the start of each session.

ii) Training of model using DistributedDataParallel (for complete dataset)
We are using the same model_loader() function that we have defined in the previous step in order to incorporate our model as per DDP parallelism.
We have defined the train_model() helper function in order to train the model. We receive all the hyper parameters and arguments to train the model from the calling main function. The training procedure is simple and straightforward. We also record time taken each time whenever we increase the number of GPUs per session to run our code and compare the model’s performance. 
 
In order to implement Multiprocessing using the DDP module we call the spawn function on the training helper function all_process() along the model arguments. We also pass additional arguments which help the object to define how to parallelize the model’s performance. 
 
 
We define the main function in order to define the entry point for our program to start parallelizing the whole operation going forward from this step. We are implementing the program for different sets of models with different GPUs and analyzing their performance with respect to time taken for completion. We store this information to analyze the results later.
We also have defined an evaluation_accuracy() helper function in order to evaluate the model’s performance as well while improving its efficiency by parallelizing it.
 
The detailed analysis of the time taken under different numbers of GPUs is illustrated in the below chart with resnet model using data distributed parallel module
For GPU = 1
 
For GPU = 2
 
For GPU  = 3
 




For GPU = 4
 

As evident from the above results we can infer that as the number of GPUs increases the time taken by the model also decreases. The results for both the procedures that we have implemented in our project show similar trend in decline in the time taken as the number of GPUs increases except for the latter method where there is a slight increase in the time when GPU = 4 from time when GPU = 3 which can be ignored as the model’s performance can be  considered equivalent.
Architecture of different models using in this project
Alexnet
 





GoogleNet
 


VGG 19
 




Resnet 50
 

InceptionV3
 






V. Result and Analysis
Now that we have done all the steps to complete this project, we are required to evaluate the project’s performance and analyze the results with respect to different arguments passed while testing the project. We will analyze the results for both the approaches and for different models as well in order to get a clear understanding of how well we managed to parallelize the entire process.
DataParallel method
Results for DataParallel method have been evaluated for 2 different models (ResNet, inceptionv3) both of them are pre-trained models. For the ResNet model we have plotted the time taken as depicted in the below chart.
 
From the trend it is clearly evident that parallelization thus improves the efficiency and reduces the time taken by model to yield the results but after certain increase in number of GPUs the trend is observed to be slightly decreasing on every increase in Number of GPU. Hence, it makes more sense to provide not more than 2 GPUs in this case to our ResNet model.
For the inceptionv3 model we have plotted the time taken as depicted in the below chart.
 

We can observe that the trend for this model doesn’t decrease on just 1 increase in the number of GPUs. We can also derive an inference that the time can be more efficiently managed if we provide more GPUs to the process. However, the number of epochs for which the model was run is also significantly more than the ResNet model.



Comparison of train and test accuracy and time taken using DataParallel method using 4 different models
 
From the above figure we can infer that after running each model for 5 epochs each model behaves differently and yields a different outcome. 
 
The model alexnet gives the best outcome with 98% accuracy and training within 111 seconds. Whereas the model googlenet yields worst performance among all with training accuracy of 92% and running time of 143 seconds.
 
Similarly, the test accuracy yields best for alexnet model and worst for googlenet model. Below we have run the resnet model on multi-GPUs. The different time taken by different models to run 10000 samples using DataParallel method along the output for different number of GPUs used for resnet model is illustrated in following charts.
 
Results for the DistributedDataParallel method have been evaluated for 4 different models (ResNet, alexnet, VGG16, googlnet) all of them are pre-trained models. And the training and testing accuracy for all the models is illustrated in the following charts.
As we have already discussed, the alexnet model yields the best performance and is most suitable to perform in our project’s use case.








Distributed Data Parallel method
The graph shows the time taken to train the model on complete dataset using different number of GPU’s
 
Inference - As the number of GPUs increases the time taken by the model also decreases. The results for both the procedures that we have implemented in our project show similar trend in decline in the time taken as the number of GPUs increases except for the latter method where there is a slight increase in the time when GPU = 4 from time when GPU = 3 which can be ignored as the model’s performance can be  considered equivale

Note: Apart from these two primary and strong approaches we also tried to implement Process Based Parallelism for our models with unfruitful results. As these models are very complex and hence require high level infrastructure to be implemented in parallel, hence,  we cannot implement them using Process bases Parallelism

Deploying a Streamlit based web application
We have created a webapp service for the trained ML model where a user can upload the images from the system and he can get the required results.
 

The Webapp snapshot is attached . The server is running on localhost on port 8501. The user has uploaded the file M and it’s showing M as output

 


Conclusion
With deep learning methodologies like CNN and help from pre-trained models from ImageNet we were able to achieve the goal we set at the starting of our project. The models accurately performed the classification of the test images. With an accuracy of 98% for the alexnet model and the usage of DDP data and model based parallelism we were able to implement the project and optimize the solution to come up with the most efficient solution. 







References

1.	https://www.kaggle.com/datasets/grassknoted/asl-alphabet
2.	https://pytorch.org/docs/stable/notes/ddp.html
3.	https://www.kaggle.com/code/swamita/asl-classification-using-cnns-keras-99-89-acc
4.	https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
5.	https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html
6.	https://rc-docs.northeastern.edu/en/latest/using-discovery/usingslurm.html
7.	https://theaisummer.com/distributed-training-pytorch/
8.	https://stackoverflow.com/questions/66825429/parallel-processing-for-testing-ml-model-with-pool-apply-async-does-not-allow-ac
9.	https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
10.	https://gist.github.com/sgraaf/5b0caa3a320f28c27c12b5efeb35aa4c
11.	https://anhreynolds.com/blogs/alexnet.html
12.	https://www.mdpi.com/2076-3417/12/6/3014/html
13.	https://towardsdatascience.com/the-annotated-resnet-50-a6c536034758
14.	https://towardsdatascience.com/an-intuitive-guide-to-deep-network-architectures-65fdc477db41
