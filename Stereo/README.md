# Stereo Vision

__Features__
* Horizontal Baseline Block Matching
	* Censoring (filter areas of low texture)
	* Progress animation
* Consistency checking

## Examples

__Build Depth Map from Stereo Images__

* python matching.py Examples/star_L.jpg Exmaples/star_R.jpg
* python matching.py Examples/pm_L.tif Exmaples/pm_R.tif
* python matching.py Examples/synth_L.tif Exmaples/synth_R.tif

In the depth map, lighter shades are closer, darker shades are farther, black can mean no information.

__Star__

* [Star Left](Examples/star_L.jpg?raw=Ture)
* [Star Right](Examples/star_R.jpg?raw=Ture)
* [Star Left Disparity Map](Examples/star_L_disparity.png?raw=Ture)
* [Star Right Disparity Map](Examples/star_R_disparity.png?raw=Ture)
* [Star Final Depth Map](Examples/star_depth.png?raw=Ture)

__Parking Meter__

* [PM Left](Examples/pm_L.tif?raw=Ture)
* [PM Right](Examples/pm_R.tif?raw=Ture)
* [PM Left Disparity Map](Examples/pm_L_disparity.png?raw=Ture)
* [PM Right Disparity Map](Examples/pm_R_disparity.png?raw=Ture)
* [PM Final Depth Map](Examples/pm_depth.png?raw=Ture)

__Synthetic Image__

* [Synth Left](Examples/synth_L.tif?raw=Ture)
* [Synth Right](Examples/synth_R.tif?raw=Ture)
* [Synth Left Disparity Map](Examples/synth_L_disparity.png?raw=Ture)
* [Synth Right Disparity Map](Examples/synth_R_disparity.png?raw=Ture)
* [Synth Final Depth Map](Examples/synth_depth.png?raw=Ture)
