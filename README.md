# Vision
__Features__
* block matching for baseline stero images
* block maching censoring to remove low texture areas in scanline
* consistency checking of left, right disparity
__TODO__
* Finish stereo Growing Correspondence Seeds (GCS)
* Move AutoKMeans here
* Move DetectCircles here

## Build depth map from stereo images:
* cd ./Stereo/
* python matching.py pm_L.py pm_R.py

__Stereo Example__
* [PM Left](Stereo/pm_L.tif)
* [PM Right](Stereo/pm_R.tif)
* [PM Left Disparity Map](Stereo/pm_L_disparity.png)
* [PM Right Disparity Map](Stereo/pm_R_disparity.png)
* [PM Final Depth Map](Stereo/pm_depth.png)

__Stereo Example__
* [Synth Left](Stereo/synth_L.tif)
* [Synth Right](Stereo/synth_R.tif)
* [Synth Left Disparity Map](Stereo/synth_L_disparity.png)
* [Synth Right Disparity Map](Stereo/synth_R_disparity.png)
* [Synth Final Depth Map](Stereo/synth_depth.png)

__Stereo Example__
* [Star Left](Stereo/star_L.jpg)
* [Star Right](Stereo/star_R.jpg)
* [Star Left Disparity Map](Stereo/star_L_disparity.png)
* [Star Right Disparity Map](Stereo/star_R_disparity.png)
* [Star Final Depth Map](Stereo/star_depth.png)
