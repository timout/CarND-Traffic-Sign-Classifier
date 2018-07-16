# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[imgperclass]: output/output_12_0.png "Sample Images per class"
[numsamples]: output/output_14_0.png "Number Of Samples in training set"
[numsamples_t]: output/output_21_0.png "Number Of Samples in training split"
[numsamples_v]: output/output_22_0.png "Number Of Samples in validation split"
[miscimages]: output/output_41_0.png "Downloaded Images"

---

#### 1. Link to project notebook

Link to my [project code](https://github.com/timout/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used python/pandas/numpy to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

##### Sample images per class

![sample images per class][imgperclass]

##### Number of images per sign class

![sample images per class][numsamples]

##### Training (after split): Number of images per sign class

![sample images per class][numsamples_t]

##### Validation (after split): Number of images per sign class

![sample images per class][numsamples_v]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. 

I decided not to grayscale the data because colours may be relevant and to decrease influence of shadows, night time, or blur, do not suffer as extreme a penalty.   
Also for simplicity I decided to use normalization only with range 0,1].

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.)
The architecture used is similiar to the LeNet architecture that was implemented in the Udacity LeNet lab.
The only differences are Dropout and L2 regularization.

My final model consisted of the following layers:
1. Input: 32x32x3 RGB image
2. Convolution layer 1   
   Input: (32, 32, 3) Output: (5, 5, 32)
   * 2D Convolution: STRIDES = (1, 1, 1, 1), PADDING = 'VALID'
   * ReLU Activation
   * 2D Max Pooling: KSIZE = (1, 2, 2, 1), STRIDES = (1, 2, 2, 1), PADDING = 'VALID'
3. Convolution layer 2  
   Input: (5, 5, 32) Output: (5, 5, 64)
   * 2D Convolution: STRIDES = (1, 1, 1, 1), PADDING = 'VALID'
   * ReLU Activation
   * 2D Max Pooling: KSIZE = (1, 2, 2, 1), STRIDES = (1, 2, 2, 1), PADDING = 'VALID'
4. Flatten: Flattens the input while maintaining the batch_size  
   Input: (5, 5, 64) Output: 1600
5. Fully connected layer 1  
   Input: 1600 Output: 1024
   * WX+b
   * ReLU Activation
   * Dropout: 0.7
6. Fully connected layer 2  
   Input: 1024 Output: 512
   * WX+b
   * ReLU Activation
   * Dropout: 0.7   
7. Output layer  
    Input: 512 Output: 43
    * WX+B 


#### 3. Describe how you trained your model. 
* Training and test data were already separated (downloaded pickled files train.p and test.p).
* Training data were split into a validation (20%) set and a training set (80%).
* Before every epoch execution training data was shuffled. 
* To reduce overfitting dropout = 0.7 was added to model.
* L2 regularization was added to Cross Entropy error to penalize large errors. As result weights were not changing too fast. 1E-6 value was chosen base on tutorial recomendations.

##### Model Parameters
1. Epochs: 150
2. Batch size: 128
3. Learning rate: 0.001
4. Truncated normal mean: 0.0
5. Truncated normal standard deviation: 0.1
6. Dropout keep rate: 0.7
7. L2 regularization strength: 1E-6
8. Loss optimization algorithm: Adam
9. Training / validation split 80/20.

All those parameters were chosen base on recommendations given in the coursework and Tensorflow tutorial.  
I have started with Epochs number = 50 and Dropout = 0.5. Gradually changing them I found the best result with 150 and 0.7.  

My final model results were:
* Training Accuracy = 0.99989
* Validation Accuracy = 0.99713
* Training Loss = 0.01612
* Validation Loss = 0.05230
* Test Set Accuracy = 0.95313

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 11 German traffic signs that I found on the web:

![Downloaded images][miscimages] 

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


Image 1
filename: rs_01.jpg (Road work) was identified correctly with confidence 100

Image 2
filename: rs_02.jpg (Speed limit (70km/h)) was identified correctly with confidence 100

Image 3
filename: rs_03.jpg (Turn right ahead) was identified correctly with confidence 100

Image 4
filename: rs_04.jpeg (Yield) was identified correctly with confidence 100

Image 5
filename: rs_05.jpg (Pedestrians Only) was identified incorrectly with confidence 100 (sign was not in training set)

Image 6
filename: rs_06.jpg (Right-of-way at the next intersection) was identified correctly with confidence 100

Image 7
filename: rs_07.jpg (Wild animals crossing) was identified incorrectly with confidence 100 (sign was not in training set)

Image 8
filename: rs_08.jpg (Priority road) was identified correctly with confidence 100

Image 9
filename: rs_09.jpg (Man with boat crossing) was identified incorrectly with confidence 100 (sign was not in training set)

Image 10
filename: rs_10.jpeg (Drunk man crossing) was identified incorrectly with confidence 100 (sign was not in training set)

Image 11
filename: rs_11.jpg (Wild animals crossing) was identified correctly with confidence 100