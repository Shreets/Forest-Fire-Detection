# Forest-Fire-Detection

The main idea behind this project is to enable detection of fire in the forested areas using deep learning and CNN architecture. The dataset consists of data with images labeled as ‘fire’ (for images with forest fire) and no fire (for images with normal vegetative lands). These images are used to train the model to recognize the pattern in images for the characteristics of forested areas with and without fire hazard occurrence. The model will however be tested on a real time video feed. A video input will further be segmented into images and fed into the model to make final evaluations. In implementing this project various feature engineering techniques will be explored to find the most effective and accurate solution. The image data set consists of multiple specific objects of which the significant ones must be selected for the model to be able to effectively identify and distinguish one condition shown in the image from another.

## Implementation
Implementation has been done in multiple steps. Starting with  research where each of us looked into papers that worked on forest fires. We inferred  that transfer learning on the Inception V3 model would eventually give better results. Next we proceed with data analysis, here we have gone through images across different classes in the dataset and inferred that certain feature extraction techniques would improve the model at the end result. 
We then worked on feature extraction techniques like edge detection, image segmentation, Noise filtering, Color Transformation ORB, Censure and corner detection e.t.c. After integrating and applying them to the dataset, we proceeded to build a basic transfer learning model on InceptionV3 with 50 epochs to test our architecture and test the model’s performance in this first phase of testing.
	
In the second phase, we then experimented with feature engineering in order to achieve the optimum performance from the model. Several combinations of the extracted features were used to train the model separately. Based on the maximum performance recorded by the feature combination the final set of features were selected which for us was a combination of all the features, that was ; image segmentation, edge detection, key points detection, thresholding, censure detection, noise filtering and color transformation. 
Further the model was tuned with varying the hyperparameters such as batch size, epocha, number of units. The recorded  accuracy and results for other performance metrics like ROC curve, confusion Matrix will be discussed further in the next section. These results were achieved after testing the integrated application with all possible scenarios of video footage captured during forest fires.


## MODEL ARCHITECTURE

### Training Architecture
![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/image12.png)
*Fig. Architecture to generate set of features to train the model*

![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/image23.png)         
*Fig. Overall Architecture to Train Model for Fire Detection*

**i) Image input :** The data set has images in their raw form where they have been labeled as fire, no fire or start fire. The image needs to be further processed before using it to train the model.

**ii) Data and feature engineering :** The images are processed in order to fit the model so that model can make precise predictions. The images are rescaled and then the most relevant features are extracted that can distinguish existence or absence of fire in images. In doing so multiple feature engineering techniques were applied to highlight the key aspects in the images. The techniques used were noise filtering, image segmentation, edge detection, color transformation, thresholding, keypoints detection and censure detection. Despite attempting to highlight and extract the most prominent features that are the signs of forest fire, not all features can be equally helpful. So, the most relevant features that in combination yielded the most accurate results were prepared to train the model.

**iii) Training Model :** The input that goes into the model are images, hence Convolutional Neural Network is being used. To save the resources on training huge data in neural networks we are using the initially training weights and retraining them with a newer set of inputs, i.e transfer learning is being implemented using InceptionV3 model. Certain layers of the networks are frozen and only part of the layers are trained with new data.

**iv) Prediction :** The model takes in the input data and predicts the probabilities for each class. The probabilities are then transformed to a single prediction for a particular class.

**v) Model Evaluation :** The training sets were being used to train the model while the validation and test sets were held out to tune and test the model accuracy respectively. Once the model was trained. The data from the test set were passed through the trained model to predict the outcome. The accuracy recorded in this set is the actual overall accuracy of the model.

**vi) Hyperparameter Tuning :** Once the model is tested with the test set, it can be decided if the model’s performance is adequate or needs to be improved. This is done by changing the hyperparameters of the model that tiles the best result. This is done using the validation datasets.

Once the model is trained and the optimum performance is achieved, then the model can be deployed to use in real life applications to check for forest fires. Although the model is being trained with images, in application, the model uses video feed to detect fire in forests. So the architecture for the deployed model differs from that during the training.


### Testing Architecture 

![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/image14.png)

Fig. Architecture of Trained Model for Fire Detection

**i) Video input :** The data set has images in their raw form where they have been labeled as fire, no fire or start fire. The image needs to be further processed before using it to train the model.

**ii) Segmenting Videos :** The continuous video feed input is converted to multiple image segments frame by frame in order to test for presence of fire in them.

**iii) Data and feature engineering :** Data engineering and feature engineering in deployed model words in the same manner as mentioned for the training architecture.

**v) Prediction :**  The features obtained from the processed images are then fed into the trained model. The model attempts to correctly predict the class of the images that have not been provided with any labels. These predictions are based on the patterns that the model had previously learned from the selected set of features during the model training process. This gives the output vector with probabilities of the possible classes which is then transformed to a single prediction for a particular class.


## Detail design of Features 
The process of extracting important and useful properties for modeling from raw data is commonly known as feature engineering. This usually entails extracting information from images, such as color, texture, and shape. There are many approaches to execute feature engineering, and the methodology that is taken depends on the type of data we are dealing with. As our goal is to identify real fire, we will try various relevant features as follows:

### 1.Noise Filter: 
It's a variety of operations used to reduce the noise from data collected on the construction and infrastructure areas. We tried with  three different filters to make our images clear and smooth before moving on to the next step in the image processing. The first filter we used was the Gaussian filter which is a Low pass filter that is used to remove noise and as we all know blurs images. The second filter we applied  was the Bilateral Filter, which also uses a technique to smooth images while preserving edges. The third filter  was the Median filter, which is well known for maintaining edges during noise removal also it uses in order to reduce random noise.
![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/image33.png)

*Fig. Gaussian Noise Filter*

### 2.Image Segmentation:
The technique of dividing a picture into various portions and segments is known as image segmentation. This method is used to identify boundaries in an image as lines and curves, as well as to name each pixel in the image. Thus, segmentation aids in streamlining and altering an image's representation to make it simpler and more understandable to examine. Image segmentation is particularly helpful for splitting and grouping images. Machine learning, computer vision, artificial intelligence, medical imaging, recognition tasks, video surveillance, object detection, and other fields all use image segmentation extensively. It affects a variety of fields, including space research and healthcare.
We will use Otsu's segmentation which is categorized as a threshold-based segment. In Otsu's Segmentation, the input image is first processed before attempting to obtain the image's histogram, which will display the distribution of pixels in the image. Here, our attention is on peak value. The threshold value must then be calculated and compared to the image pixels in the following phase. If the value is more than the threshold but less than black, set the pixel to white. It carries out automatic thresholding as a result.
![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/image32.png)

*Fig. Image Segmentation*

### 3. Color Transformation : RGB to LAB:
Designed using the frequently used and well known RGB color palette, LAB is another effective way of communicating colors. LAB is a color space with a much richer and vast color spectrum when compared with other color palettes like RGB, BGR or YCbCr. ‘L’ stands for lightness in the range of 0 to 100. A and B traverse all the colors from red to green and yellow to blue respectively. In the context of this project, color transformation has been applied to add a new flavor to the process of extracting features from the image. Additionally, fire and smoke being the vital higher level features for this project, can be detected only by studying the color changes throughout the image. As the spectrum of LAB captures much more minute details in the color patterns of the image, its use will only enhance the capabilities of the classifier.
![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/image28.png)

*Fig. RGB to LAB color transformation*

### 4. Thresholding :
Thresholding is a technique used for image segmentation and is achieved by varying pixels. This helps to better analyze the image. Particularly for this dataset, thresholding helps in segmenting  fire/smoke and forest background. This will help the model learn to differentiate the details better from the image. 
Here two types of thresholding are applied to images. The first is binary thresholding which outputs a binary image and it can be applied to both gray scale and colored images. The second is Adaptive thresholding which is subclassified into Adaptive Gaussian, Adaptive Mean thresholding. It is observed that binary thresholding has given better image segmentation for analysis.
![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/image29.png)

*Fig. Thresholding*

### 5. Edge Detection:
Edge detection here is being done to check if the edges can separate the fire entity in the image. Through edge detection the discontinuity in the brightness is checked when one image section changes to another. It is understood that the image section with the fire must have a sudden change in the intensity of brightness and the change in edge should give some indication of the image segment with fire in it. The images were not grayscale before applying edge detection techniques as it is also necessary that the hue is detected in the image as hue is an important factor to detect fire in the image. First the horizontal and the vertical edges were checked to see if significant segments of images are detected through these edges. Further, the vertical edges were also checked to identify similar edges. Then further the vertical and horizontal edges were combined to see if any significant difference is observed in the images with and without fire where the images with fire have distinct characteristic differences compared to the ones without fire.
A Sobel filter has been used to identify the edges. Sobel filter calculates the gradient of the intensity of each pixel. Using sobel filter the intensity changes in the image was identified
![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/image3.png)


*Fig : Horizontal Edge Detection*
                                               
![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/image20.png)


*Fig : Vertical Edge Detection*
                                                
![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/image6.png)


*Fig : Horizontal and Vertical Edges Combined* 


### 6. ORB Keypoint Detection:
ORB(Oriented FAST and Rotated BRIEF) is an amalgamation of two different Image Processing Techniques. FAST (Features from Accelerated Segment Testing) which is an open source adaptation of a keypoint detector like SIFT or SURF. BRIEF (Binary Robust Independent Elementary Features) is an effective keypoint descriptor algorithm which is usually used in tandem with a keypoint extractor. The purpose of ORB in the project is to detect sudden shifts in the pixel intensities, which can be looked upon as detecting corners in the image. Output generated in the notebook file suggests that the algorithm detects corners of the trees, sharp rocky edges,etc. in the images. A set of descriptors generated by the BRIEF part in ORB is beneficial in matching the key points of a certain image with any other image. This comes in handy for the classifier model to train itself with this information and further predict an outcome based on the learnt keypoint descriptors.

![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/image22.png)
                                        
 
 *Fig. Oriented FAST and Rotated BRIEF keypoint detection*

### 7. Image Orientation
Orientation is varying the angle of the image to get a desired real time scenario than an inverted or tilted image. This would ensure images of similar class and pattern are learnt as the same category by the model which avoids confusion while the model is learning by updating weights.

![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/image31.png)


*Fig. Image Orientation*


### 8. Censure and Corner Detection
Censure feature detector detects the significant features and is highly used in real time images to extract important details. Harris Corner detector as name suggests helps in locating the corners and helps in inferring image features.
![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/image30.png)


*Fig. Images after censure and corner detection*

## Feature Engineering and Model Training

After trying out multiple combinations of features, the following two combinations have proven to be most effective.
After a good amount of  research on the suitable metrics for the context of this project, the metrics we will be using to evaluate our models are Accuracy Score, Confusion Matrix, Precision-Recall-F1 Score.

### 1. Gaussian Noise Filter + Segmentation + LAB color space :-
Herein, we stack the feature maps extracted from the Gaussian Noise filter, Segmentation filter and color components from LAB space feature extraction techniques and train our CNN architecture using this map. 
Following are the results obtained :-

**Loss and Accuracy curves**

![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/accuracy-curve-comb1.png)

*Fiq. Accuracy curve for combination 1*

![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/loss-curve-comb1.png)

*Fig. Loss curve for combination 1*

As usual, we can observe that the validation accuracy goes increasing as the network trains in identifying fire from the images.

**Accuracy Score :- **
The optimum accuracy score obtained for this combination after fine tuning the number of epochs, on an unseen test data set is 77.23%

**Confusion Matrix :-**
Confusion Matrix for the three target labels in the context of this project is as follows,

![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/confusion-matrix-comb1.png)


As we can see from the above matrix, from a dataset of 694, most of the test instances are classified correctly. 286 of the fire images were classified as fire, 224 of the non-fire forest images were classified under the no-fire category, while the remaining 26 of the images which displayed initial stages of fire, were classified as start fire.
Precision and Recall:-
Precision and Recall values estimated for the model trained using this combination of features are 70% and 71% respectively, which is good enough compared to other combinations.

### 2. Edge Detection Sobel Filters + Gaussian Noise Filter + ORB Keypoins :-
The next set of features we stack up are edges in the images using the sobel filters for horizontal and vertical edge detection in the images, gaussian noise filters to get rid of any noisy pixels in the images and smoothen them. The last feature we pick is the ORB keypoints detected in the images as these key points may carry vital information about the nature of fire or forest vegetation in the images.
Following results are obtained :-

**Loss and Accuracy curves**

![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/acc-curve-comb1.png)
 
*Fig. Accuracy curve for combination 2*

![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/loss-comb2.png)

*Fig. Loss curve for combination2*

We can observe the sharp transitions in accuracy and loss for both the training and validation curves. Accuracy goes on significantly increasing while the loss gradually decreases.

**Accuracy Score :-**
We obtained an accuracy score of 77% using this model to make predictions on the same unseen data set.

**Confusion Matrix :-**
Following confusion matrix is obtained,

![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/confusion-comb2.png)

From the above matrix we can say that the model has improved in its performance over the Fire and No Fire classes but has completely failed in identifying the start fire category. One of the reasons for this may be that the features involved in this combination mostly deal with the physical components of the image like shapes and structures. Out of 694 test instances, 283 of the fire images were classified as fire images and 252 of the non-fire images were classified under the no fire category. The third category, which is the start fire category, has failed miserably.

Precision and Recall:-
Precision and Recall values obtained for this model are 51% and 55% which is poor compared to the earlier combination, but holds a vital insight which will be discussed further in this report.

### 3. All Features combined :-
Observing the results from both set of features, some of the conclusions we can draw are
First set of features performs well on all three classes. The precision and recall values for the model convey the same story.
Second set of features gives exceptionally good results on the Fire and No Fire classes but fails to predict any of the Start Fire instances.
Combination of all the features would be a good option to go ahead with as it will combine the positives from both the models.
Hence, the final model we train combines all the extracted features that are mentioned above in this report.
Following are the results obtained for this final model,

**Loss and Accuracy curves :-**

![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/acc-comb3.png)

*Fig. Accuracy curve*

![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/loss-comb3.png)

*Fig. Loss curve*
                                                           
We can observe from the above curves that there is a significant increase in the accuracy values compared to the previous models.

**Accuracy Score :-**
Based on the predictions of this model on an unseen test dataset, estimated accuracy of the model comes out to be 92%, which supports the hypothesis that the combination of all the features proves to be more effective when detecting all the fire patterns in images.

**Confusion Matrix :-**
Following is the confusion matrix obtained for this model’s predictions,

![alt text](https://github.com/Shreets/Forest-Fire-Detection/blob/main/images/confusion-matrix-comb3.png)

As we can see, the diagonal entries of the above matrix look very promising. Out of 694 test data samples, 309 which consisted of fire were predicted as Fire, 299 of no-fire images were predicted as no-fire and 25 of the images which displayed the initial stages of fire were classified under the Start Fire category.

**Precision and Recall :-**
Precision and Recall values for the model’s predictions also show a significant change from the earlier combinations. We achieve a precision value of 90% while a recall value of 81% for this model.


