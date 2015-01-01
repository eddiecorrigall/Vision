# Unrectified Examples

For each subfolder in Examples/unrectified there contains:

* Let image1.? be the __unrectified left image__
* Let image2.? be the __unrectified right image__

* Let rectified1.? be the __rectified left image__
* Let rectified2.? be the __rectified right image__

* Let gcs1.? be the __left gcs disparity__
* Let gcs2.? be the __right gcs disparity__

* Let block1.? be the __left block matching disparity__
* Let block2.? be the __right block matching disparity__

* Let K.npy be the __camera matrix__
* Let d.npy be the __distortion parameters__ in OpenCV format
* Let x1.npy be __feature points__ in image1
* Let x2.npy be __feature points__ in image2

## Loading npy Files ##

Load a *.npy file using numpy.load(..)
For example: K = numpy.load("Examples/girl/K.npy")
