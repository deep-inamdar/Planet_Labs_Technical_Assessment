# PSScene4Band Imagery Time Series Analyzer
## Program Description

The PSScene4Band Imagery Time Series Analyzer is designed to analyze a set of
PS2 images (item_type:PSScene4Band). The analysis quantifies the rate of change
from green vegetation to bare soil over the time period represented by the 
image series.


# Instructions

## Download Repository to Local Folder
1) Open link to GitHub repository
2) Click the green " <> code" button 
3) Download ZIP to desired location 
4) Unzip folder to dersired location

## Install Required Libraries
1) Open Command Prompt (Windows) or Terminal (macOS)
2) Change directory to unzipped folder ("Planet_Labs_Technical_Assessment-main")
3) Run "pip install -r requirements.txt"

## Running Progam
1) Change directory to unzipped folder ("Planet_Labs_Technical_Assessment-main")
2) Run the code: "python main.py"
3) When prompted, select to the data folder
   (.../Planet_Labs_Technical_Assessment-main/Data") in the pop-up prompt.
4) The results will be printed in the console. Go to the output directory in 
   (.../Planet_Labs_Technical_Assessment-main/Data/Output") to see additional 
   data figures.

# Additional Details About Program

## High-level flowchart
1) Extract file names of images, UDM and xml files.
2) Extract relevant meta data from xml files (including TOA-Coefficients)
3) Import imagery and apply TOA-Coefficients
4) Calculate NDVI for imagery time series.
5) Import masks from UDM files and merge into a single mask. 
6) Apply mask to NDVI imagery time series
7) Thresold NDVI to classify imagery into vegetation and soil. Perform change
   detection on a pixel-by-pixel basis between sequential data aquisitions.
8) Calculate rate of change from vegetation class to soil class between
   sequential data aquistions
9) Plot NDVI images, classification images, and change detection images

## Theoretical approach

The devleoped function produces an approximate measure indicating the rate of 
change from "green vegetation" to "bare soil" over the time period represented 
by the image series. To acomplish this task, the imagery first needed to be 
classified into vegetation and soil classes. Spectrally, vegetation and soil are
quite distinct. Where soild generally increases from the blue to the NIR, 
vegetation is typically characterized by a chlorophyll absorption feature that
results in poor reflectance in the red portion of the reflectance. These key
spectral differences are captured by the normalized difference vegetation index
(NDVI), which looks at the normalized difference between the NIR and red band:

$$
NDVI = (\rho _{NIR}-\rho_{red})/(\rho _{NIR} +\rho_{red})
$$

where $\rho _{NIR}$ and $\rho _{red}$ rerpresent reflectance in the red and NIR 
bands, respectively.

Due to the chlorophyll absorption feature, large NDVI values are assoicated
with vegetation. Lower values that approach zero are typically associated with
soil (due to the gradually increasing reflectance spectrum from the red to the
NIR). As such, NDVI can be thresholded to classify pixels as either vegetation
or soil. 

$$

\left\{ 
  \begin{array}{ c l }
    \frac{x^2 - x}{2} & \quad \textrm{if } x \geq 1 \\
    0                 & \quad \textrm{otherwise}
  \end{array}
\right.

$$
Various thesholds were tested (Threshold=0.1,0.2,0.3,0.4,0.5,0.6).


After classifying the imagery from each date, the rate of change
between sequential images can be calculated. Specifically, the rate of 
change from "green vegetation" to "bare soil" between time sequential images was
calculated using the following expression:

$$
R_{veg,soil}= N_{pix}(class_{veg,t_1},class_{veg,t_2})*Res^2/(t_2-t_1)
$$

where $R_{veg,soil}$ represents the rate of change (m^2/day) from vegetation
to soil, $N_pix(class_{veg,t_1},class_{veg,t_2})$ represents the number of pixels
that were classified as vegetation at time 1 ($t_1$) and soil at time 2 ($t_2$),
and $Res$ represents the spatial resolution of each pixel (3 m). It is important
to note that the rate of change calculation is directional (i.e., R_{veg_to_soil}≠ -R_{soil_to_veg}.

 
2) The Term "green vegetation" was difficult to interprete (i.e., what is considered green vegetation?). This challenge is handled by defined green vegetation based on the green absoption feature.



9) only data files can be located in folder location... otherwise program fails. this can create issues when analyzing data with other files stored in same directory as imaging datasets

## Challenges and Potential Future Improvements

1) The NDVI threshold 

1) Automatic image mosaicing is not comprehensive and only merges images 
   collected on the same day(if one half of a data aquistion was collected at 
   11:59 PM on the 1st, and the second half was collected at 12:00 am on the 2nd, 
   they would be treated as different time series entries). This issue could be 
   eliminate by merging files based on the time difference between sequential 
   images (reported in img_2_img_time_diff variable). This process would need 
   to be repeated while there is atleast one element in the img_2_img_time_diff
   variable that was less than a given time difference threshold (e.g., 1 hour). 
2) Image mosaicing occurs before the Top-of-atmosphere coefficents are applied.
   As such the TOA-coefficents from the first image are applied to both
   mosaiced images. Although this may be an issue that could be mitigated by
   moving the image normalization before image mosaicing, coefficents from 
   images separated by 1 second should vary insignificantly. If they do vary 
   drastically, it may not be ideal to use the imagery given the unstable 
   atmospheric conditions.
3) All functions were designed to be generalizable. In this task, I was
   was assigned to calculate the rate of change from vegetation to soil. Using
   the devleoped functions, the program can easily be modified to look at 
   different types of changes (e.g., vegetation to soil). The function were also
   designed to handle multiple classes that can be differentiated by 
   thresholding NDVI. 
4) The time series mask could be expanded to remove water from the scene using 
   a spectral indices that exploits information in the blue end of the spectrum.
   In this analysis, NDVI<0 were unclassified. Given that water typically has a 
   negative NDVI (as water attenuates electromagnetic radiation strongly at 
   longer wavelengths), the unclassified class likely represented water. In the
   analyzed scene, there was only one easily visible water body. Given that
   that this body changed to soil, throughout the time series, this cloud imply
   that the body of water was shallow and dried up in the summer.
5) The python script was developed and tested on a windows 10 machine. While
   testing the program in macOS, various bugs appeared. Although these bugs were
   resolved, the program should be stress tested on a variety of operating 
   systems. 
6) No additional data files or folders (other than the output
   folder that is generated while running the program) can be present in the 
   data folder otherwise the program will fail. This is due to how the program 
   parses the file names. This issue could be resolved by requiring users to
   manually input the name of the files to be analzyed instead of automatically
   extracting them. 

