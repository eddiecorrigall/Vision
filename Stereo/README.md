# Stereo Vision

__Features__
* Growing Correspondence Seeds
* Uncalibrated/calibrated stereo rectication
* Horizontal Baseline Block Matching
	* Censoring (filter areas of low texture)
	* Progress animation
	* Consistency checking

## Examples

* python rectify.py Examples/girl
* python matching.py Examples/star/rectified1.jpg Examples/star/rectified2.jpg
* python gcs.py Examples/martin/rectified1.png Examples/martin/rectified2.png 64 3
