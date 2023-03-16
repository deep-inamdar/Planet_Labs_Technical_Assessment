#Required Libraries
import rasterio
from rasterio.merge import merge
import os
import fnmatch
import shutil
from xml.dom import minidom
import numpy as np
from datetime import datetime

# Merge Rasters
def merge_rasters(folder_loc,raster_names):    
    raster_to_mosiac = []
    for i in range(len(raster_names)):
        raster=rasterio.open(folder_loc+raster_names[i])
        raster_to_mosiac.append(raster)
    mosaic, output = merge(raster_to_mosiac)
    
    output_meta = raster.meta.copy()
    output_meta.update(
        {"driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": output,
        }
    )
    mos_name="Mosaic/" + raster_names[0][:-4]+"_mos.tif"
    #write out image
    with rasterio.open(folder_loc+mos_name, "w", **output_meta) as m:
        m.write(mosaic)
    mos_name=[mos_name]
    return  mos_name


def prep_4_import(folder_loc):
    folder_loc=folder_loc.replace(os.sep, '/')
    if folder_loc[-1]!='/':
        folder_loc+='/'
        
    #Create folder for output
    isExist = os.path.exists(folder_loc+'Output')
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(folder_loc+'Output')
    #Create folder for potential Mosaic
    isExist = os.path.exists(folder_loc+'Mosaic')
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(folder_loc+'Mosaic')

    #Obtain a list of files in directory
    folder_dir=os.listdir(folder_loc)


    folder_dir_dates=[(i[:8]) for i in folder_dir[:-2]]
    unq_dates=list(set(folder_dir_dates))
    unq_dates.sort(key = int)

    #Identify images needing to be merged

    img_str='_3B_AnalyticMS_clip.tif'
    xml_str='_AnalyticMS_metadata_clip.xml'
    udm2_str='_3B_udm2_clip.tif'


    #Identify images to be analyzed and mosaic images from the same day
    img_name_list=[]

    for i in range(len(unq_dates)):
        pattern = unq_dates[i]+'*'+img_str
        matching = fnmatch.filter(folder_dir, pattern)
        if len(matching)>1:
            matching=merge_rasters(folder_loc,matching)
        img_name_list.append(matching)

    mask_name_list=[]

    for i in range(len(unq_dates)):
        pattern = unq_dates[i]+'*'+udm2_str
        matching = fnmatch.filter(folder_dir, pattern)
        if len(matching)>1:
            matching=merge_rasters(folder_loc,matching)
        mask_name_list.append(matching)

    xml_name_list=[]

    for i in range(len(unq_dates)):
        pattern = unq_dates[i]+'*'+xml_str
        matching = fnmatch.filter(folder_dir, pattern)
        if len(matching)>1:
            shutil.copy(folder_loc+matching[0],folder_loc+ "Mosaic/" + matching[0][:-4]+"_mos.xml")
            matching="Mosaic/" + matching[0][:-4]+"_mos.xml"
        xml_name_list.append(matching)
        
    return img_name_list, mask_name_list, xml_name_list, folder_loc
    
    
def xml_info(folder_loc,xml_name_list):
    #predefine output variables
    coeffs=np.zeros([len(xml_name_list), 4])
    times=np.zeros([len(xml_name_list),1]).astype(datetime)
    
    
    for i in range(len(xml_name_list)):
        xmldoc = minidom.parse(folder_loc+''.join(xml_name_list[i]))
        nodes_coef = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")
        nodes_time=xmldoc.getElementsByTagName("ps:Acquisition")
        
        #Extract TOA reflectance coefficients
        for node in nodes_coef:
            bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
            if bn in ['1', '2', '3', '4']:
                j = int(bn)-1
                value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
                coeffs[i,j] = float(value)
                
        date_time=nodes_time[0].getElementsByTagName("ps:acquisitionDateTime")[0].firstChild.data
        times[i] = datetime.strptime(date_time[0:19], "%Y-%m-%dT%H:%M:%S")
    return coeffs, times
    
    
    
def import_imgry_norm(folder_loc,img_name_list,coeffs):
    
    #import meta data
    meta_data=[]
    for i in range(len(img_name_list)):
        img_meta=rasterio.open(folder_loc+''.join(img_name_list[i])).meta
        meta_data.append(img_meta)
    
    #check spatial dimensions 
    spatial_extent_chck= len(set([(j["transform"]) for j in meta_data[:]]))
    if    spatial_extent_chck!=1:
        raise Exception("Image spatial exentents do not match each other. Please check input data")
    
    #calculate easting and northing for each column and row, respectively
    map_info=img_meta.get("transform")
    easting_vec=np.arange(map_info[2],map_info[2]+map_info[0]*(img_meta['width']),map_info[0])
    northing_vec=np.arange(map_info[5],map_info[5]+map_info[4]*(img_meta['height']),map_info[4])
    
    #Import imagery
    img_ts=np.zeros([len(img_name_list),img_meta['count'],img_meta['height'],img_meta['width']])
    for i in range(len(img_name_list)):
        img_ts[i,:,:,:]=rasterio.open(folder_loc+''.join(img_name_list[i])).read()
    
    img_ts_ref=np.zeros([len(img_name_list),img_meta['count'],img_meta['height'],img_meta['width']])
    
    #normalize imagery using TOA reflectance coefficients
    for i in range(len(img_name_list)):
        for j in range(img_meta['count']):
            img_ts_ref[i,j,:,:] = np.multiply(img_ts[i,j,:,:],coeffs[i,j])
    
    return img_ts_ref, easting_vec,northing_vec

def calc_ndvi(img_ts_ref):
    
    #set up calculations
    img_dim=np.shape(img_ts_ref)
    np.seterr(divide='ignore', invalid='ignore')
    ndvi_ts=np.zeros([img_dim[0],img_dim[2],img_dim[3]])
    
    #generate ndvi time series
    for i in range(img_dim[0]):
        ndvi_ts[i,:,:]= (img_ts_ref[i,3,:,:].astype(float)-img_ts_ref[i,2,:,:].astype(float))/(img_ts_ref[i,3,:,:].astype(float)+img_ts_ref[i,2,:,:].astype(float))
    
    return ndvi_ts

def import_mask(folder_loc,mask_name_list):
    meta_data=[]
    for i in range(len(mask_name_list)):
        img_meta=rasterio.open(folder_loc+''.join(mask_name_list[i])).meta
        meta_data.append(img_meta)
    
    #check spatial dimensions 
    spatial_extent_chck= len(set([(j["transform"]) for j in meta_data[:]]))
    if    spatial_extent_chck!=1:
        raise Exception("Mask spatial exentents do not match each other. Please check input data")
    
    
    #combine all time series mask into 1
    mask_ts=np.ones([img_meta['height'],img_meta['width']])
    for i in range(len(mask_name_list)):
        temp_mask=rasterio.open(folder_loc+''.join(mask_name_list[i])).read(1).astype(bool)
        mask_ts=np.logical_and(mask_ts,temp_mask)
    #reverse band 1 to generate mask    
    mask_ts=np.invert(mask_ts)
    return mask_ts


def apply_mask(mask_ts,img_ts):
    mask_data= np.broadcast_to(mask_ts, img_ts.shape)
    imgs_ts_masked=np.ma.array(img_ts, mask=mask_data)

    return imgs_ts_masked


