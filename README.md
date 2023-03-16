# Planet_Labs_Technical_Assessment
Take Home Test for Planet Labs Interview

# Task

The data folder includes GeoTIFFs and metadata files for a number of PlanetScope acquisitions over Sumatra between June and September, 2020. The AnalyticMS files are analytic assets from PSScene4Band imagery; the udm2 files are Usable Data Masks; and the metadata.json files include the full imagery footprint, acquisition times, and additional metadata. All raster data is clipped to the same extent (which is different than the full imagery footprints).

Your task is to write a program that analyzes the imagery and produces an approximate measure indicating the rate of change from green vegetation to bare soil over the time period represented by the image series.

In submitting your task, please provide instructions on running your program and include any additional detail describing the choices you made or issues you encountered.

# Instructions

# Additional Details About Program
Inputs: folder location of images, masks and 



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
7) Since all of the images are from the same 