# PSScene4Band Imagery Time Series Analyzer
##Program Description

The PSScene4Band Imagery Time Series Analyzer is designed to analyze a set of
PS2 images (item_type:PSScene4Band). The analysis quantifies the rate of change
from green vegetation to bare soil over the time period represented by the 
image series.


# Instructions

## Download repository to local folder
1) Open directory
2)

## Install required library
1) Navigate to the folder location using the command line
2) Run "-m pip install -r requirements.txt"

## Running Progam
1) Navigate to the folder location using the command line
2) Run main.py
3) When prompted, navigate to the "Data" folder in the downloaded directory
4) The results will be printed in the console. Go to the indentified directories
to see the auxilary data files. 

# Additional Details About Program





Soil can be differentiated from green vegetaion via the chloropyl absoption feature and the red edge. As such NVDI and Greenness spectral indices should be able to differentiate between the two materials.

The histogram can be used to establish a threshold value for the classification. 

Band 1 of the UDM file was used as the image mask since we were not interested in snow, shadows, haze(light or heavy) or clouds. 

% Challenges

1) This was my first remote sensing project using Python and Github. As such, it took some time to shift gears from MATLAB to Python Syntax. 
2) The Term "green vegetation" was difficult to interprete (i.e., what is considered green vegetation?). This challenge is handled by defined green vegetation based on the green absoption feature.
3) Image 2 was broken into two halves. need to merge them
4) It was unclear if the time was on a 24 hour clock or not.
5) Automatic extraction of images is not comprehensive (if one half of a data aquistion was collected at 11:59PM on the 1st, and the second half was collected at 12:00am on the 2nd, they would be treated as different time series entries)
6) There were slightly different atmospheric coeffiencts between the two halves of the 20200727 imagery. I decided to use the coefficents form the first image on both under the assumption that the atmospheric differences between the two images were negligble (which they should be if data was collected under stable illumination connditions).
7) Since all of the images are from the same size, we dont need to spatially subset the data...it would have been nice to add image check function to see if all images have the same transform
8) It was unclear 
9) only data files can be located in folder location... otherwise program fails. this can create issues when analyzing data with other files stored in same directory as imaging datasets



# overveiw 

#running



#extensions

