#Required libraries
import os
import fnmatch
from Function_Definitions import prep_4_import, xml_info, import_imgry_norm, calc_ndvi, import_mask, apply_mask, plot_ndvi_ts
import shutil
import numpy as np
import matplotlib.pyplot as plt
#input parameters
folder_loc='D:\Planet_Techical_Assignment\Assessment\data'


#Extract list of files sorted by date and mosiac images from the same day
img_name_list, mask_name_list, xml_name_list,folder_loc=prep_4_import(folder_loc)

#Extract TOA reflectance coefficients and time for each dataset
coeffs, times=xml_info(folder_loc,xml_name_list)


#Import imagery, applyTOA reflectance coefficients 
img_ts_ref, easting_vec,northing_vec=import_imgry_norm (folder_loc,img_name_list,coeffs)

#Calculate NDVI
ndvi_ts=calc_ndvi(img_ts_ref)

#Input time series masks and merge into a single mask
mask_ts=import_mask(folder_loc,mask_name_list)

#Apply time series mask to data 

ndvi_ts_masked=apply_mask(mask_ts,ndvi_ts)

#define threshold values
class_threshold= np.array([[0, 0.3], [0.3, np.max(ndvi_ts_masked)]])


img_2_img_time_diff, class_ts, delta_class_ts, delta_GV_SO=(ndvi_ts_masked, times,class_threshold, mask_ts)

#plot_ts
min_col=.3
max_col=1
color_mapping='YlGn'

plot_ndvi_ts(ndvi_ts_masked, times, easting_vec,northing_vec, min_col,max_col,color_mapping)



min_col=0
max_col=1
color_mapping='YlGn'

plot_ndvi_ts((delta_class_ts==302), times[:3], easting_vec,northing_vec, min_col,max_col,color_mapping)





plt.hist(ndvi_ts_masked.compressed())
plt.hist(ndvi_ts.flatten())

np.nanmax(ndvi_ts_masked)


class_2_transition_code(start_class,end_class)
