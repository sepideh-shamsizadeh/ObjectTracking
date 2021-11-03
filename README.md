# ObjectTracking
Get as input some pictures of objects and the video that want track objects in it.
In the first image of the video we find the location of objects of interest, which get an example image as input. For doing this we used the SIFT features, and the cv::BFMatcher class as feature matcher. Show the keypoints of different objects with different colors.

To discard outliers we used the RANSAC algorithm provided inside the findHomography() function. Then we start only to track features.

Draw a rectangle around each located object. Track (and show) the features of each object using the pyramid Lukas-Kanade tracker implemented in OpenCV 

Update at each new image the rectangles around each located object, e.g., by estimating the translation and rotation of the objects by using the optical flow of the related points.
A sample video and a set of four example objects are provided, by build our own dataset with your objects, e.g. by using your smartphone.
