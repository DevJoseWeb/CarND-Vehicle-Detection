# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation. 

Woohoo :)

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it -- I hope you enjoy it!

### Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell of the IPython notebook (P5.ipynb) under the header "Histogram of Gradients, Parameter Tuning and Feature Extraction". 

I started by reading in all the `vehicle` and `non-vehicle` images provided in lecture. Due to the size of the datasets, they are stored locally on my machine. Here are links to download them from Udacity:  . Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Adapting the HOG feature extraction function provided in lecture, I extracted and plotted HOG feature vectors for random images of my dataset (comprising of the GTI and KITTI datasets) using different color spaces such as YUV, HSV, HLS, LUV, Lab, and YCrCb. I repeated this multiple times on various combinations of color spaces and HOG parameters. 

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried multiple combinations of color spaces and HOG parameters, eventually settling on a combination using YCrCb color space, HOG parameters (orientation=8, pix_per_cell=8, and cell_per_block=2). I settled on this combination as a result of the next step, which involved training a linear SVM classifier. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using YCrCb color, orientation=8, pix_per_cell=8 and cell_per_block=2. I used all HOG channels. In addition, I spatially binned each image. Relevant parameters include hist_bins=32 and spatial_size=(32,32). I tried to decrease spatial_size to (16, 16), but I saw significant drops in my test accuracy. 

I decided to use a linear SVM because it is generally fast to train and yields high accuracy. From past experience (having taken the Udacity Machine Learning course as well as personal projects), I hypothesized that a linear SVM would be more than sufficient for this project. This assumption was supported by the recommendations of many students in the online forums and discussions.

Here are my experiments:

  __Experiment 1: Hist + HOG__
  * Color Space: HLS
  * Orient = 11
  * Pix per Cell = 8
  * Cell per Block = 2
  * HOG Channel = 2
  * Hist Bins = 32
  * Y Min = image_shape[1]//2
  * Y Max = image_shape[1] - 100
  * __RESULTS__: Feature vector length: 2252 | 5.3 Seconds to train SVC... | Test Accuracy of SVC =  0.9738

  __Experiment 2: Hist + Spatial + HOG__
  * Color Space: YCrCb
  * Orient = 9
  * Pix per Cell = 8
  * Cell per Block = 2
  * HOG Channel = "ALL"
  * Spatial Size = 32, 32
  * Hist Bins = 32
  * Y Min = image_shape[1]//2
  * Y Max = image_shape[1] - 100
  * __RESULTS__: Feature vector length: 8460 | 26.11 Seconds to train SVC... | Test Accuracy of SVC =  0.9896
  * __RESULTS__: Feature vector length: 8460 | 23.99 Seconds to train SVC... | Test Accuracy of SVC =  0.9893
  * __RESULTS__: Feature vector length: 8460 | 22.89 Seconds to train SVC... | Test Accuracy of SVC =  0.987
  * __RESULTS__: Feature vector length: 8460 | 24.52 Seconds to train SVC... |Test Accuracy of SVC =  0.9882
  * __RESULTS__: Feature vector length: 8460 | 8.54 Seconds to train SVC... | Test Accuracy of SVC =  0.9876
  
  __Experiment 3: HOG (test purposes only)__
   * Color Space: HLS
  * Orient = 11
  * Pix per Cell = 8
  * Cell per Block = 2
  * HOG Channel = 2
  * Hist Bins = 32
  * __RESULTS__: Feature vector length: 5292 | 7.25 Seconds to train SVC... | Test Accuracy of SVC =  0.9783

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I restricted the search space of the sliding window search to be the bottom half of the image. As recommended, it seemed illogical to search for objects in the sky. I stuck with a 50% overlap window because I found it to be fairly accurate in discovering cars. However, it did take some nudging to ignore false positives. Here is an example output of the sliding window search: 

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

However, I am not a fan of the vanilla Sliding Window approach. Instead I used the HOG Sub-sampling Window Search. The beauty of this approach lies in the fact that a part of the original image is sampled and can immediately make a prediction within a restricted area. Knowing that the same car in a moving video stream can appear at different scales, this allows us to explicitly specify intervals of the image where we'd like the classifier to search. I specified approximately 7 different scales ranging between y-values of [400, 700] -- from the middle to the bottom of the image, or rather viewable components of the road. 

In the end, I searched on 7 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. This came after a lot of experimentation with parameter tuning. Please refer to my experiment log included above. Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_with_detection.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To combine overlapping bounding boxes, I implemented the heatmap feature. Effectively, I keep track of all rectangular detections, and increment the heatmap for each new detection. If multiple rectangles on the heatmap cluster around a given area forming a "blob", I can safely assume that "blob" is perhaps a car. Using scipy.ndimage.measurements.label(), I can aggregate those rectangles into a single labeled location around which I can draw a close bounding box. For a single image, if multiple detections have identified a single car, it most likely has a "hot" (or high) detection value. I filter out false positives by applying a threshold to the heatmap, i.e., a "hot" value of at least 5 constitutes a valid detection of a car. 

In addition, I borrowed an approach that was recommend by Ryan in this really helpful video [here](https://www.youtube.com/watch?v=P2zwrTM8ueA&list=PLAwxTw4SYaPkz3HerxrHlu1Seq8ZA7-5P&index=5), and drew inspiration from Vivek Yadav and Jeremy Shannon, two veteran SDCND students. Basically, I created a class called Tracker() that keeps track of the last 10 frames of the video input, and stores vehicle detections when they pop up. For each new detection, I aggregate all detected rectangles, and average the last 10 discovered heatmaps. I update this average with the newest heatmap that was just discovered. The intuition here is that I am using previously discovered locations so that I can smoothen the jumpiness of the bounding box as the car moves along its trajectory. Furthermore, I apply a threshold to the heatmap, which is essentially a median value of all previously detected cars within the last 10 frames. This does a great job dropping false positives, in addition to maintaining fairly stable and smooth bounding boxes around detected objects. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I really enjoyed working on this project, as I did with Project 1, primarily because I felt like I had control over tuning the model's parameters with immediate feedback. One of the main difficulties was establishing a proper set of scales with which to optimize the classifier's search over the image for vehicle detections. Sometimes, I would make adjustments and nothing would happen, so it would be increasingly frustrating to debug and fix. 

One of the other issues that I faced was coming up with a suitable approach to track and optimize vehicle detection in the moving video stream. I had to rely on helpful explanations and approaches from Jeremy Shannon and Vivek Yadav to establish a proper working Tracker class. To improve this pipeline, I would like to re-design the Tracker class to track more meaningful information like rectangle dimensions, xy-coordinates and centroid displacement to predict where a car might go based on its previous trajectory. 

I think one of the current short-comings of my pipeline is not having video data of another car moving directly into the car's driving lane. I'm curious to see if my pipeline will be able to properly identify the front car in this situation. Similarly, how would my pipeline behave in an urban street with pedestrians? I hypothesize that I will have many false positives being detected, in addition to lacking processing for traffic/movement along the median of the image moving horizontally rather than vertically. 
