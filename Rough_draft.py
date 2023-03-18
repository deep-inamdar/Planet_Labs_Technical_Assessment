#Required libraries
from Function_Definitions import prep_4_import, xml_info, import_imgry_norm, calc_ndvi, import_mask, apply_mask, plot_ndvi_ts, ts_analysis, plot_classification
import numpy as np

#input parameters
folder_loc='D:\Planet_Techical_Assignment\Assessment\data'

print('---PSScene4Band Imagery Time Series Analyzer----')
print()
print('Preparing and extracting file information...')
#Extract list of files sorted by date and mosiac images from the same day
img_name_list, mask_name_list, xml_name_list,folder_loc=prep_4_import(folder_loc)


print('Extracting relevant meta data from xml file...')
#Extract TOA reflectance coefficients and time for each dataset
coeffs, times=xml_info(folder_loc,xml_name_list)

print('Improrting imagery as time series and applying Top-of-atmosphere coefficients...')
#Import imagery, apply TOA reflectance coefficients 
img_ts_ref, easting_vec,northing_vec=import_imgry_norm (folder_loc,img_name_list,coeffs)


print('Calculating NDVI...')
#Calculate NDVI
ndvi_ts=calc_ndvi(img_ts_ref)


print('Extracting time series mask from UDM2 files...')
#Input time series masks and merge into a single mask
mask_ts=import_mask(folder_loc,mask_name_list)


print('Applying generated mask...')
#Apply time series mask to data 
ndvi_ts_masked=apply_mask(mask_ts,ndvi_ts)


print('Calculating rate of change between green vegetation and bare soil...')
# Carry out time series analysis
class_threshold= np.array([[0, 0.4], [0.5, np.max(ndvi_ts_masked)]])  #find way to automate calculation
img_2_img_time_diff, class_ts, delta_class_ts, delta_GV_SO=ts_analysis(ndvi_ts_masked, times,class_threshold, mask_ts)

print('Generating NDVI figures in '+folder_loc+'Output/NDVI/ ...')
#Plot NDVI
min_col=0
max_col=1
color_mapping='YlGn'
#plot_ndvi_ts(ndvi_ts_masked, times, easting_vec,northing_vec, min_col,max_col,color_mapping,folder_loc)

print('Generating classification figures in '+folder_loc+'Output/Classification/ ...')
#Plot classification
class_names=['Unclassified', 'Soil', 'Vegetation','Mask']
plot_classification(class_ts, times, easting_vec,northing_vec,folder_loc,class_names)

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
print('--Rate of change from green vegetation to baren soil between sequential dates--')
for i in range(len(times)-1):
    print('Date '+str(i+1)+' to Date '+str(i+2)+": "+ str(round(delta_GV_SO[i]))+ " m^2/day" )











