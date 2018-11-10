## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

# **Traffic Sign Recognition** 

## Writeup



[//]: # (Image References)
[image1]: ./distibution.png "Distribution"
[image2]: ./random.png "Random"
[image3]: ./preprocess.png "Preprocess"
[image8]: Custom/1.png "Stop"
[image7]: Custom/2.jpg "Right of way at the next intersection"
[image5]: Custom/3.jpg "No entry"
[image4]: Custom/4.jpg "Yield"
[image6]: Custom/5.png "go straight or right"


---


### Data Set Summary & Exploration


I used numpy and matplotlib libraries to explore the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12360
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset.

I did an exploratory visualization of the data set to see how many images are present in each class. The histogram shown below gives the distribution of data in each class for both train and test images.
![alt text][image1]

A bunch of random images from the training data and their corresponding labels is shown below.
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First I converted the data into grayscale using the formula 0.21R + 0.72G + 0.07B. Grayscale images are easier to process and the size of the data becomes smaller and thus operations can be run on them faster.

After the image is converted to grayscale I normalized the image to make it zero mean and equal variance. Shown below is the image before and after the preprocessing steps mentioned above.
![alt text][image3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the basic LeNet model for classifcation of traffic signs. My model consisted of the following layers:

| Layer         		|     Description	        					                | 
|:---------------------:|:-------------------------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					                | 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	                |
| RELU					| outputs 28x28x6								                |
| Max pooling	      	| 2x2 stride, 2x2 kernel, Valid padding, outputs 14x14x6 		|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16  	                |
| RELU					| outputs 10x10x16								                |
| Max pooling	      	| 2x2 stride, 2x2 kernel, Valid padding, outputs 5x5x16 		|
| Flatten   	      	| outputs 400       	                                        |
| Fully connected		| input 400, outputs 120        								|
| RELU  				| outputs 120               									|
| Fully connceted		| input 120, outputs 84											|
| RELU					| outputs 84     												|
| Fully connected		| input 84, outputs 43											|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model using the GPU provided by Udacity. I tried out several hyperparameters and optimizers, realized that Adam optimizer gave pretty stable and quick convergence, started with a learning rate of 1e-2 and moved up and down to see which one gives a stable error reduction. Trained the model for 50 epochs with a batch size of 64.

Hyperparametes used are: 

Learning rate : 0.0015

Batch size : 64

Epochs : 50

Optimizer : Adam

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

There are several CNN architectures for classification like VGG, ALexNet, ResNet, etc. Most of the networks are developed for imagenet classifcation and the images are 224x224. They cannot be used directly for these images which are 32x32. So, the best bet would be to try out architectures used for MNIST. So, I used the very popular and easy to implement LeNet architecture (by Yann LeCun et. al). LeNet has fewer parameters to train compared to other architectures like AlexNet or VGG plus.

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.9375
* test set accuracy of 0.929
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The image with the yield and the stop sign might be difficult to classify because because of the watermark in the images and the fact that the image is not zoomed into the sign but rather has a significant amount of background in it.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                         |     Prediction	           			       | 
|:--------------------------------------:|:-------------------------------------------:| 
| Yield      							 | Yield								       |
| No entry								 | No entry								       |
| Go straight or right					 | Go straight or right 				       |
| Right-of-way at the next intersection  | Right-of-way at the next intersection       |
| Stop    		                         | No passing for vehicles over 3.5 metric tons| 

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The images are not clear and 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the second image, the model is sure that this is a yield sign (probability of 1), and the image does contain a yield sign. The top five soft max probabilities were

| Probability           	|    Prediction 				| 
|:-------------------------:|:-----------------------------:| 
| 1.00000         			| Yield							| 
| 0.00000     				| Stop							|
| 0.00000					| Ahead only					|
| 0.00000	      			| Traffic Signals 				|
| 0.00000				    | Speed limit(20km/h)			|


For the second image, the model is sure that this is a No entry sign (probability of 1), and the image does contain a No entry sign. The top five soft max probabilities were


| Probability           	|    Prediction 				| 
|:-------------------------:|:-----------------------------:| 
| 1.00000         			| No entry						| 
| 0.00000     				| Speed limit (20km/h)          |
| 0.00000					| Stop					        |
| 0.00000	      			| Speed limit (30km/h)			|
| 0.00000				    | Speed limit(50km/h)			|

For the third image, the model is sure that this is a Go straight or right sign (probability of 1), and the image does contain a Go straight or right sign. The top five soft max probabilities were

| Probability         	    |     Prediction 				 | 
|:-------------------------:|:------------------------------:| 
| 1.00000         			| Go straight or right		 	 | 
| 0.00000     				| Ahead only				     |
| 0.00000					| Dangerous curve to the right	 |
| 0.00000	      			| Bumpy road				     |
| 0.00000				    | End of no passing				 |

For the fourth image, the model is sure that this is a Go straight or right sign (probability of 1), and the image does contain a Go straight or right sign. The top five soft max probabilities were

| Probability         	    |     Prediction	   				    | 
|:-------------------------:|:-------------------------------------:| 
| 1.00000         			| Right-of-way at the next intersection	| 
| 0.00000     				| Roundabout mandatory					|
| 0.00000					| Priority road						    |
| 0.00000	      			| Speed limit (20km/h) 				    |
| 0.00000				    | Speed limit (30km/h) 				    |


For the fifth image, the model is not entirely sure that this is a No passing for vehicles over 3.5 metric tons sign (probability of 0.92) , because the image contains a stop sign. The top five soft max probabilities were

| Probability         	    |     Prediction	        					    | 
|:-------------------------:|:-------------------------------------------------:| 
| 0.92901        			| No passing for vehicles over 3.5 metric tons	    | 
| 0.07049     				| Right-of-way at the next intersection			    |
| 0.00049					| End of no passing by vehicles over 3.5 metric tons|
| 0.00000	      			| Vehicles over 3.5 metric tons prohibited			|
| 0.00000				    | Slippery road 						            |


The performance and generalization capabilities can be improved by data augmentation techniques and using regularization techniques like dropout, L2, etc.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



