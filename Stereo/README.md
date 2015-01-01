# Stereo Vision

__Features__
* Uncalibrated/calibrated stereo rectication
* Horizontal Baseline Block Matching
	* Censoring (filter areas of low texture)
	* Progress animation
	* Consistency checking

## Examples

* python rectify.py Examples/girl
* python matching.py Examples/star/image1.jpg Exmaples/star/image2.jpg

__Note:__ In the depth map, lighter shades are closer, darker shades are farther, black can mean no information.
