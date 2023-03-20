"""
This program is used to analyze PSScene4Band imagery time series. The program
quantifies the rate of change from green vegetation to soil over the time
period represented by the image series. Additional figures are generated to
cacluate NDVI and classify each image. Furthermore, a change detection image is
generated for each time sequential pair of images.

Author: Deep Inamdar
"""



#Load required libraries
from tkinter.filedialog import askdirectory
import numpy as np
from function_definitions import prep_4_import, xml_info, import_imgry_norm, calc_ndvi, import_mask, apply_mask, plot_ndvi_ts, ts_analysis, plot_classification, calc_rate_of_change, plot_class_diff

# Start program
print('---PSScene4Band Imagery Time Series Analyzer----')
print()
print('Select the folder location of the time series data using the pop-up dialog')

# Grab folder directory
folder_loc=askdirectory()
print()

# Extract list of files sorted by date and mosaic images from the same day
print('Preparing and extracting file information...')
img_name_list, mask_name_list, xml_name_list,folder_loc=prep_4_import(folder_loc)

# Extract TOA reflectance coefficients, resolution and time for each dataset
print('Extracting relevant meta data from xml file...')
coeffs, times, resolution=xml_info(folder_loc,xml_name_list)

# Import imagery and apply TOA reflectance coefficients
print('Improrting imagery as time series and applying Top-of-atmosphere coefficients...')
img_ts_ref, easting_vec,northing_vec=import_imgry_norm (folder_loc,img_name_list,coeffs)

# Calculate NDVI
print('Calculating NDVI...')
ndvi_ts=calc_ndvi(img_ts_ref)

# Import time series masks and merge into a single mask
print('Extracting time series mask from UDM2 files...')
mask_ts=import_mask(folder_loc,mask_name_list)

# Apply time series mask to data
print('Applying generated mask...')
ndvi_ts_masked=apply_mask(mask_ts,ndvi_ts)

# Carry out time series analysis
print('Conducting change detection analysis...')
threshold_val=.40
class_threshold= np.array([[0, threshold_val], [threshold_val, np.max(ndvi_ts_masked)]])  #find way to automate calculation
img_2_img_time_diff, class_ts, delta_class_ts=ts_analysis(ndvi_ts_masked,times,class_threshold,mask_ts)

# Vegetation = class 2, soil = class 1
start_class=2
end_class=1
delta_GV_SO= calc_rate_of_change(img_2_img_time_diff,delta_class_ts,start_class,end_class,resolution)

# Plot NDVI
print('Generating NDVI figures in '+folder_loc+'Output/NDVI/ ...')
min_col=0
max_col=1
color_mapping='YlGn'
plot_ndvi_ts(ndvi_ts_masked,times,easting_vec,northing_vec, min_col,max_col,color_mapping,folder_loc)

# Plot classification
print('Generating classification figures in '+folder_loc+'Output/Classification/ ...')
class_names=['Unclassified', 'Soil', 'Vegetation']
plot_classification(class_ts,times,easting_vec,northing_vec,folder_loc,class_names)

# Plot change detection
print('Generating change detection figures in '+folder_loc+'Output/Change_Detection/ ...')
plot_class_diff(delta_class_ts,times,easting_vec,northing_vec,folder_loc,class_names,mask_ts)

# Print rate of change from vegetation to soil
print()
print('---Program complete---')
print()
print('---Output---')
print()
print('--Date of time series entries--')
for i in range(len(times)):
    d=times[i][0].strftime("%Y-%m-%dT%H:%M:%S")
    print('Date '+str(i+1)+': '+d)
print()
print('--Rate of change from green vegetation to soil between sequential dates--')
for i in range(len(times)-1):
    print('Date '+str(i+1)+' to Date '+str(i+2)+": "+ str(round(delta_GV_SO[i]))+ " m^2/day" )
