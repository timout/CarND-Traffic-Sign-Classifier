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
[test_1]: output/output_43_7.png "Test Image 1"
[test_2]: output/output_43_17.png "Test Image 2"
[test_3]: output/output_43_19.png "Test Image 3"
[test_4]: output/output_43_3.png "Test Image 4"
[test_5]: output/output_43_11.png "Test Image 5"
[test_6]: output/output_43_21.png "Test Image 6"
[test_7]: output/output_43_9.png "Test Image 7"
[test_8]: output/output_43_13.png "Test Image 8"
[test_9]: output/output_43_1.png "Test Image 9"
[test_10]: output/output_43_5.png "Test Image 10"
[test_11]: output/output_43_15.png "Test Image 11"

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

I decided not to grayscale the data because colors may be relevant and to decrease influence of shadows, night time, or blur.   
Also for simplicity I decided to use normalization only with range [0,1].

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
* Training and test data were already separated (train.p and test.p).
* Training data were split into a validation (20%) set and a training set (80%).
* Before every epoch execution training data was shuffled. 
* To reduce overfitting dropout = 0.7 was added to model.
* L2 regularization was added to Cross Entropy error to penalize large errors. As result weights were changing slowly. 1E-6 value was chosen base on tutorial recomendations.

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

The code for making predictions on my final model is located in the 27th code-cell of the Ipython notebook.  
Here are 11 German traffic signs that I found on the web:

![Downloaded images][miscimages] 

Some of the image types were not in training set so I downloaded them out of curioucity.  

#### Image 1 (Road work)

![Road work][test_1] 

The image was identified correctly with confidence 100%.  
5 top predictions:  
 1. Class_id:25 (Road work), confidence:100%
 2. Class_id:11 (Right-of-way at the next intersection), confidence:0%
 3. Class_id:0 (Speed limit (20km/h)), confidence:0%
 4. Class_id:1 (Speed limit (30km/h)), confidence:0%
 5. Class_id:2 (Speed limit (50km/h)), confidence:0%

#### Image 2 (Speed limit (70km/h))

![Speed limit (70km/h)][test_2] 

The image was identified correctly with confidence 100%.  
5 top predictions:  
 1. Class_id:4 (Speed limit (70km/h)), confidence:100%
 2. Class_id:33 (Turn right ahead), confidence:0%
 3. Class_id:2 (Speed limit (50km/h)), confidence:0%
 4. Class_id:19 (Dangerous curve to the left), confidence:0%
 5. Class_id:35 (Ahead only), confidence:0%

#### Image 3 (Turn right ahead)

![Turn right ahead][test_3] 

The image was identified correctly with confidence 100%.  
5 top predictions:  
 1. Class_id:33 (Turn right ahead), confidence:100%
 2. Class_id:0 (Speed limit (20km/h)), confidence:0%
 3. Class_id:1 (Speed limit (30km/h)), confidence:0%
 4. Class_id:2 (Speed limit (50km/h)), confidence:0%
 5. Class_id:3 (Speed limit (60km/h)), confidence:0%

#### Image 4 (Yield)

![Yield][test_4] 

The image was identified correctly with confidence 100%.  
5 top predictions:  
 1. Class_id:13 (Yield), confidence:100%
 2. Class_id:0 (Speed limit (20km/h)), confidence:0%
 3. Class_id:1 (Speed limit (30km/h)), confidence:0%
 4. Class_id:2 (Speed limit (50km/h)), confidence:0%
 5. Class_id:3 (Speed limit (60km/h)), confidence:0%

#### Image 5 (Pedestrians Only)

![Pedestrians Only][test_5] 

The image was identified incorrectly with confidence 100.  
Image type was not in traning set - I was just curious how it will be identified the model. 
5 top predictions:  
 1. Class_id:34 (Turn left ahead), confidence:100%
 2. Class_id:0 (Speed limit (20km/h)), confidence:0%
 3. Class_id:1 (Speed limit (30km/h)), confidence:0%
 4. Class_id:2 (Speed limit (50km/h)), confidence:0%
 5. Class_id:3 (Speed limit (60km/h)), confidence:0%

#### Image 6 (Right-of-way at the next intersection)

![Right-of-way at the next intersection][test_6] 

The image was identified correctly with confidence 100%.  
5 top predictions:  
 1. lass_id:11 (Right-of-way at the next intersection), confidence:100%
 2. Class_id:0 (Speed limit (20km/h)), confidence:0%
 3. Class_id:1 (Speed limit (30km/h)), confidence:0%
 4. Class_id:2 (Speed limit (50km/h)), confidence:0%
 5. Class_id:3 (Speed limit (60km/h)), confidence:0%

#### Image 7 (Wild animals crossing)

![Wild animals crossing][test_7] 

The image was identified incorrectly with confidence 100%.  
Technically training set contained the class but I suspected the image was not in traning set so I was just curious how it will be identified the model.  
5 top predictions:  
 1. Class_id:21 (Double curve), confidence:100%
 2. Class_id:11 (Right-of-way at the next intersection), confidence:0%
 3. Class_id:19 (Dangerous curve to the left), confidence:0%
 4. Class_id:10 (No passing for vehicles over 3.5 metric tons), confidence:0%
 5. Class_id:31 (Wild animals crossing), confidence:0%

#### Image 8 (Priority road)

![Priority road][test_8] 

The image was identified correctly with confidence 100%.  
5 top predictions:  
 1. Class_id:12 (Priority road), confidence:100%
 2. Class_id:0 (Speed limit (20km/h)), confidence:0%
 3. Class_id:1 (Speed limit (30km/h)), confidence:0%
 4. Class_id:2 (Speed limit (50km/h)), confidence:0%
 5. Class_id:3 (Speed limit (60km/h)), confidence:0%

#### Image 9 (Man with boat crossing)

![Man with boat crossing][test_9] 

The image was identified incorrectly with confidence 100%.  
The image class was not in traning set so I was just curious how it will be identified the model.  
5 top predictions:  
 1. Class_id:4 (Speed limit (70km/h)), confidence:100%
 2. Class_id:18 (General caution), confidence:0%
 3. Class_id:26 (Traffic signals), confidence:0%
 4. Class_id:14 (Stop), confidence:0%
 5. Class_id:33 (Turn right ahead), confidence:0%

#### Image 10 (Drunk man crossing)

![Drunk man crossing][test_10] 

The image was identified incorrectly with confidence 71%.  
The image class was not in traning set so I was just curious how it will be identified by the model.  
5 top predictions:  
 1. Class_id:10 (No passing for vehicles over 3.5 metric tons), confidence:71%
 2. Class_id:9 (No passing), confidence:17%
 3. Class_id:4 (Speed limit (70km/h)), confidence:9%
 4. Class_id:5 (Speed limit (80km/h)), confidence:2%
 5. Class_id:22 (Bumpy road), confidence:1%

#### Image 11 (Wild animals crossing)

![Wild animals crossing][test_11] 

The image was identified correctly with confidence 100%.  
5 top predictions:  
 1. Class_id:31 (Wild animals crossing), confidence:100%
 2. Class_id:10 (No passing for vehicles over 3.5 metric tons), confidence:0%
 3. Class_id:25 (Road work), confidence:0%
 4. Class_id:19 (Dangerous curve to the left), confidence:0%
 5. Class_id:11 (Right-of-way at the next intersection), confidence:0%


"New Images" were logically devided into 2 groups: 
1. Known to the model (7)  
2. Unknown to the model (4)  
All images from the "known group" were classified correctly with confidence 100%. Surprisingly model was able to classify all "known" images with EPOCHS = 10 only but with increasing EPOCHS it lost that ability, I tested up to 256 epochs. Dropout increase fixed that problem.    
All images from "unknown group" were classified incorrectly mostly with confidence 100%. 100% was a confusing result but inability to classify I guess was expected since model did not have any logic to generalize. Image 7 (Wild animals crossing) was a very good example of that inability to generalize. That is very good point for improvement.
