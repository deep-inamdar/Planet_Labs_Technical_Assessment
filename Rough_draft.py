#Required libraries
import os
import fnmatch
from Function_Definitions import prep_4_import, xml_info, import_imgry_norm, calc_ndvi, import_mask, apply_mask
import shutil

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





















