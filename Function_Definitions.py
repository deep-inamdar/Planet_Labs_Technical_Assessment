"""
A collection of python functions used to analyze PSScene4Band imagery
clipped to the same extent.The functions are used in main.py
to quantify the rate of change from green vegetation to bare soil over the time
period represented by the image series. 

Author: Deep Inamdar
"""



#Required Libraries
import os
import fnmatch
import shutil
from xml.dom import minidom
from datetime import datetime
import rasterio
from rasterio.merge import merge
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches



def merge_rasters(folder_loc,raster_names):
    """
    Mosaics together images, writing out mosaiced image in the "Output/Mosaic/" 
    subfolder. The mosaiced file is in *.tif format with "_mos" appended to the
    first listed file name.

    Parameters
    ----------
    folder_loc : str
        The location of the folder that contains the PSScene4Band time series 
        data to be mosaiced. 
    raster_names : list of strings, size= n
        List of strings containing the file names of the n images to be 
        mosaiced.

    Returns
    -------
    mos_name : list of single string, size= 1
        A list containing a string with the name of the mosaiced file. 
    """
    
    #Create folder for mosaiced image
    is_exist = os.path.exists(folder_loc+'Output/Mosaic')
    if not is_exist:
        os.makedirs(folder_loc+'Output/Mosaic')

    #Mosaic all files on raster_names
    raster_to_mosiac = []
    for i in range(len(raster_names)):
        raster = rasterio.open(folder_loc+raster_names[i])
        raster_to_mosiac.append(raster)
    mosaic, output = merge(raster_to_mosiac)

    #Generate meta data for mosaiced image
    output_meta = raster.meta.copy()
    output_meta.update(
        {"driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": output,
        }
    )
    mos_name = "Output/Mosaic/" + raster_names[0][:-4]+"_mos.tif"

    #Write out image
    with rasterio.open(folder_loc+mos_name, "w", **output_meta) as m:
        m.write(mosaic)
    mos_name = [mos_name]

    return  mos_name



def prep_4_import(folder_loc):
    """
    Prepares folder with time series imagery for data analysis. The preparation
    identifies the names of the image files, along with the names of the 
    ancillary UDM and xml files. It also creates a folder for outputs from the 
    analysis.The function mosaics all imagery (including UDM files) collected 
    on the same day. 

    Parameters
    ----------
    folder_loc : str
        The location of the folder that contains the PSScene4Band time series 
        data to be merged. 

    Returns
    -------
    img_name_list : list of strings, size=n
        A list of n strings with the names of the time series images to be 
        analyzed. 
    mask_name_list: list of strings, size=n
        A list of n strings with the names of UDM files associated with time
        series images to be analyzed. 
    xml_name_list: list of strings, size=n
        A list of n strings with the names of the UDM files associated with 
        time series images to be analyzed.   
    folder_loc: str
        The location of the folder that contains the PSScene4Band time series 
        data to be merged (forward slashes only, ends in forward slash). 
    """

    #Ensure that '\' are replaced with '/' and folder directory ends in '/'
    folder_loc=folder_loc.replace(os.sep, '/')
    if folder_loc[-1] != '/':
        folder_loc += '/'

    #Create folder for output from workflow
    is_exist = os.path.exists(folder_loc+'Output')
    if not is_exist:
        os.makedirs(folder_loc+'Output')

    # Obtain a list of files in directory
    folder_dir = os.listdir(folder_loc)

    # Extract dates from file names
    # First 8 digits represent the date
    folder_dir_dates = [(i[:8]) for i in folder_dir[:-1]]
    unq_dates = list(set(folder_dir_dates))
    unq_dates.sort(key = int)

    #Define unique ending string of images, UDM files, and xml files
    img_str='_3B_AnalyticMS_clip.tif'
    xml_str='_AnalyticMS_metadata_clip.xml'
    udm2_str='_3B_udm2_clip.tif'

    #Identify images to be analyzed and mosaic images from the same day
    img_name_list = []
    for i in range(len(unq_dates)):
        pattern = unq_dates[i]+'*'+img_str
        matching = fnmatch.filter(folder_dir, pattern)

        #Mosaic images from same day
        if len(matching) > 1:
            matching=merge_rasters(folder_loc,matching)
        img_name_list.append(matching)

    #Identify images to be analyzed and mosaic images from the same day
    mask_name_list = []
    for i in range(len(unq_dates)):
        pattern = unq_dates[i]+'*'+udm2_str
        matching = fnmatch.filter(folder_dir, pattern)
        if len(matching) > 1:
            matching = merge_rasters(folder_loc,matching)
        mask_name_list.append(matching)

    #Identify xml files to be analyzed
    xml_name_list = []
    for i in range(len(unq_dates)):
        pattern = unq_dates[i]+'*'+xml_str
        matching = fnmatch.filter(folder_dir, pattern)
        if len(matching)>1:
            shutil.copy(folder_loc+matching[0],folder_loc+ "Output/Mosaic/" + matching[0][:-4]+"_mos.xml")
            matching = "Output/Mosaic/" + matching[0][:-4]+"_mos.xml"
        xml_name_list.append(matching)

    return img_name_list, mask_name_list, xml_name_list, folder_loc



def xml_info(folder_loc,xml_name_list):
    """
    Extracts relevant information (Top-of-atmosphere(TOA) coefficents, and data
    aquistion times) from xml files. 

    Parameters
    ----------
    folder_loc : str
        The location of the folder that contains the PSScene4Band time series 
        data to be merged (forward slashes only, ends in forward slash). 
    xml_name_list, : list of strings, size=n
        A list of n strings with the names of the UDM files associated with 
        time series images to be analyzed.  
        
    Returns
    -------
    coeffs: Array of float64, size= (n,b)
        An array of the top of Top-of-atmosphere(TOA) coefficents for band, b,
        and image n. Each row corresponds with a different image while each 
        column corresponds with a coefficent for a different band. 
    times: array of datetime objects, size=(n,1)
        An array of the dates/times associated with each time series image.
    """

    #Predefine output variables
    coeffs=np.zeros([len(xml_name_list), 4])
    times=np.zeros([len(xml_name_list),1]).astype(datetime)

    #Extract information from xml
    for i in range(len(xml_name_list)):
        xmldoc = minidom.parse(folder_loc+''.join(xml_name_list[i]))
        nodes_coef = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")
        nodes_time = xmldoc.getElementsByTagName("ps:Acquisition")
        for node in nodes_coef:
            bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
            if bn in ['1', '2', '3', '4']:
                j = int(bn)-1
                value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
                coeffs[i,j] = float(value)
        date_time = nodes_time[0].getElementsByTagName("ps:acquisitionDateTime")[0].firstChild.data
        times[i] = datetime.strptime(date_time[0:19], "%Y-%m-%dT%H:%M:%S")
    resolution= float(xmldoc.getElementsByTagName("eop:resolution")[0].firstChild.data)
    
    return coeffs, times, resolution



def import_imgry_norm(folder_loc,img_name_list,coeffs):
    """
    Imports imagery from img_name_list, applying Top-of-atmosphere coefficents
    and extracting the postion of each pixel column and row. 
    
    Parameters
    ----------
    folder_loc : str
        The location of the folder that contains the PSScene4Band time series 
        data to be merged (forward slashes only, ends in forward slash). 
    img_name_list : list of strings, size=n
        A list of n strings with the names of the time series images to be 
        analyzed. 
    coeffs: Array of float64, size= (n,b)
        An array of the top of Top-of-atmosphere(TOA) coefficents for band, b,
        and image n. Each row corresponds with a different image while each 
        column corresponds with a coefficent for a different band. 
        
    Returns
    -------
     
    img_ts_ref: Array of float64, size=(t,b,y,x)
        Time series data in units of reflectance. Dimension t,b,y and x 
        correspond to the time, spectral band, pixel row and pixel column, 
        respectively.  
    easting_vec: Array of float64, size=(x,)
        Easting postion (m) of each column of the imagery.  
    northing_vec: Array of float64, size=(x,)
        Northing postion (m) of each row of the imagery. 
    """

    #Import meta data as a list
    meta_data = []
    for i in range(len(img_name_list)):
        img_meta = rasterio.open(folder_loc+''.join(img_name_list[i])).meta
        meta_data.append(img_meta)

    #Check spatial dimensions to see if all images are the same size
    spatial_extent_chck = len(set([(j["transform"]) for j in meta_data[:]]))
    if    spatial_extent_chck != 1:
        raise Exception("Image spatial exentents do not match each other. Please check input data")

    #Calculate easting and northing for each column and row, respectively
    map_info = img_meta.get("transform")
    easting_vec = np.arange(map_info[2],map_info[2]+map_info[0]*(img_meta['width']),map_info[0])
    northing_vec = np.arange(map_info[5],map_info[5]+map_info[4]*(img_meta['height']),map_info[4])

    #Import imagery
    img_ts = np.zeros([len(img_name_list),img_meta['count'],img_meta['height'],img_meta['width']])
    for i in range(len(img_name_list)):
        img_ts[i,:,:,:] = rasterio.open(folder_loc+''.join(img_name_list[i])).read()

    #Normalize imagery using TOA reflectance coefficients
    img_ts_ref = np.zeros([len(img_name_list),img_meta['count'],img_meta['height'],img_meta['width']])
    for i in range(len(img_name_list)):
        for j in range(img_meta['count']):
            img_ts_ref[i,j,:,:] = np.multiply(img_ts[i,j,:,:],coeffs[i,j])

    return img_ts_ref, easting_vec,northing_vec



def calc_ndvi(img_ts_ref):
    """
    Calculates NDVI from img_ts_ref time series.
    
    Parameters
    ----------
    img_ts_ref: Array of float64, size=(t,b,y,x)
        Time series data in units of reflectance. Dimension t,b,y and x 
        correspond to the time, spectral band, pixel row and pixel column, 
        respectively.
        
    Returns
    -------
    ndvi_ts: Array of float64, size=(t,y,x)
        Time series NDVI data. Dimension t,y and x 
        correspond to the time, pixel row and pixel column, 
        respectively.  
    """

    #Set up calculations
    img_dim = np.shape(img_ts_ref)
    np.seterr(divide = 'ignore', invalid = 'ignore')
    ndvi_ts = np.zeros([img_dim[0],img_dim[2],img_dim[3]])

    #Generate ndvi time series
    for i in range(img_dim[0]):
        ndvi_ts[i,:,:] = (img_ts_ref[i,3,:,:].astype(float)-img_ts_ref[i,2,:,:].astype(float))/(img_ts_ref[i,3,:,:].astype(float)+img_ts_ref[i,2,:,:].astype(float))

    return ndvi_ts




def import_mask(folder_loc,mask_name_list):
    """
    Extracts information from UDM files and generates a single mask for the 
    time series analysis. The mask eliminates pixels that are not present in 
    ALL of the time series images. 

    Parameters
    ----------
    folder_loc : str
        The location of the folder that contains the PSScene4Band time series 
        data to be merged (forward slashes only, ends in forward slash).
    mask_name_list: list of strings, size=n
        A list of n strings with the names of UDM files associated with time
        series images to be analyzed.

    Returns
    -------
    mask_ts: Array of bool, size=(y,x)
        Time series mask. The mask is TRUE for all spatial pixels that are not
        present in ALL of the time series images.
    """

    #Import meta data as a list
    meta_data = []
    for i in range(len(mask_name_list)):
        img_meta = rasterio.open(folder_loc+''.join(mask_name_list[i])).meta
        meta_data.append(img_meta)

    #Check spatial dimensions to see if images are all the same size
    spatial_extent_chck = len(set([(j["transform"]) for j in meta_data[:]]))
    if    spatial_extent_chck != 1:
        raise Exception("Mask spatial exentents do not match each other. Please check input data")

    #Combine all time series masks from UDM files into a single mask
    mask_ts = np.ones([img_meta['height'],img_meta['width']])
    for i in range(len(mask_name_list)):
        temp_mask = rasterio.open(folder_loc+''.join(mask_name_list[i])).read(1).astype(bool)
        mask_ts = np.logical_and(mask_ts,temp_mask)
    #reverse band 1 to generate mask
    mask_ts = np.invert(mask_ts)

    return mask_ts



def apply_mask(mask_ts,img_ts):
    """
    Extracts information from UDM files and generates a single mask for the 
    time series analysis. The mask eliminates pixels that are not present in 
    ALL of the time series images. 

    Parameters
    ----------
    mask_ts: Array of bool, size=(y,x)
        Time series mask. The mask is TRUE for all spatial pixels that are not
        present in ALL of the time series images.
    img_ts: Array of float64
        Time series data that needs to be masked. Size of time series is 
        variable, so long as the spatial dimensions matchup with mask_ts. 

    Returns
    -------
    img_ts_masked: Array of float64
        Masked time series. Size is equal to the size of img_ts.
    """
    mask_data = np.broadcast_to(mask_ts, img_ts.shape)
    imgs_ts_masked = np.ma.array(img_ts, mask = mask_data)

    return imgs_ts_masked



def ts_analysis(ndvi_ts_masked, times,class_threshold, mask_ts):
    """
    Extracts information from UDM files and generates a single mask for the 
    time series analysis. The mask eliminates pixels that are not present in 
    ALL of the time series images. 

    Parameters
    ----------
    ndvi_ts_masked: Masked Array of float64, size=(t,y,x)
        Masked time series NDVI data. Dimension t,y and x 
        correspond to the time, pixel row and pixel column, 
        respectively.  
    times: array of datetime objects, size=(n,1)
        An array of the dates/times associated with each time series image.
    class_threshold: Array of float64, size(j,k)
        Lower and upper NDVI threshold for class j (e.g., 
        np.array([[lower_thresh_class_1, upper_class_1], [lower_thresh_class_2, upper_class_1]])) 
    mask_ts: Array of bool, size=(y,x)
        Time series mask. The mask is TRUE for all spatial pixels that are not
        present in ALL of the time series images. 

    Returns
    -------
    img_2_img_time_diff: Array of float64, size= (t-1,)
        Time difference(days) between the t sequential time series images.
    class_ts: Array of float64, size= (t,y,x)
        Classifed time series data. Value of 1:j represent classes of 1:j, 
        respectively. A value of zero indicates the pixel does not fall in any
        of these classes. 
    delta_class_ts: Transition image between sequential time series images.
        Each value represent a distinct transition from one class to another.
        Please use transition_code_2_class to identify which codes correspond 
        to which transitions. 
    delta_GV_SO: Array of float64, size= (t-1,) 
        Rate of change (m^2/day) from green vegetation to barren soil
        between sequential time series images. 
    """

    #Get dimensions of ndvi time series and class threshold
    thresh_dims = np.shape(class_threshold)
    img_dim = np.shape(ndvi_ts_masked)

    #Predefine output variables
    class_ts = np.zeros(img_dim)
    class_ts = apply_mask(mask_ts,class_ts)
    tot_pixels = np.zeros([img_dim[0],thresh_dims[0]+1])
    img_2_img_time_diff = np.zeros([len(times)-1])
    delta_class_ts = np.zeros([len(times)-1,img_dim[1],img_dim[2]])
    
    #Calculate days between images
    for i in range(len(times)-1):
        delta = times[i+1]-times [i]
        img_2_img_time_diff[i] = delta[0].total_seconds()/60/60/24

    #Generate classififed time series data cube and count number of pixels in each class
    for i in range(thresh_dims[0]):
        indx = np.logical_and(ndvi_ts_masked > class_threshold[i,0],ndvi_ts_masked <= class_threshold[i,1])
        class_ts[indx] = i+1

    #Calculate number of pixels in each class
    for i in range(img_dim[0]):
        for j in range(thresh_dims[0]+1):
            tot_pixels[i,j] = np.sum(class_ts[i,:,:] == (j+1))
            if j == thresh_dims[0]:
                tot_pixels[i,j] = np.sum(class_ts[i] == 0)

    #Generate change map between time sequential images
    for i in range(img_dim[0]-1):
        delta_class_ts[i,:,:] = (class_ts[i,:,:]+1)*100+(class_ts[i+1,:,:]+1)
    delta_class_ts = apply_mask(mask_ts,delta_class_ts)
    
    return img_2_img_time_diff, class_ts, delta_class_ts



def calc_rate_of_change(img_2_img_time_diff,delta_class_ts, start_class, end_class, resolution):
    """
    Calculate rate of change from starting class to ending class. 

    Parameters
    ----------
    img_2_img_time_diff: Array of float64, size= (t-1,)
        Time difference(days) between the t sequential time series images.

    delta_class_ts: Transition image between sequential time series images.
        Each value represent a distinct transition from one class to another.
        Please use transition_code_2_class to identify which codes correspond 
        to which transitions. 
    start_class: float
        Number of starting class
    end_class: float
        Number of ending class
    Returns
    -------
    delta_GV_SO: Array of float64, size= (t-1,) 
        Rate of change (m^2/day) from class 1 to class 2
    """
    # Number of pixels that tranistioned from vegetation at t=1 to soil at t=2
    # Value is multiplied by the resolution squared to convert pixels to m^2
    # Value is then normalized by days between aquisitions to obtain rate 
    delta_change = np.zeros(len(img_2_img_time_diff))
    for i in range(len(img_2_img_time_diff)):
        delta_change[i] = np.sum(delta_class_ts[i,:,:] == class_2_transition_code(start_class,end_class))*resolution*resolution/img_2_img_time_diff[i]
    
    return delta_change



def class_2_transition_code(start_class,end_class):
    """
    Identify value of pixel from delta_class_ts at which pixel transitions from 
    the starting class to the ending class.

    Parameters
    ----------
    start_class: float
        Number of starting class
    end_class: float
        Number of ending class
    Returns
    -------
    Transition_code: float
        Identify starting class and ending class assoicated with the values 
        from delta_class_ts. 
    """

    transition_code = (start_class+1)*100+(end_class+1)
    
    return transition_code

def transition_code_2_class_code(transition_code):
    """
    Identify value of pixel from delta_class_ts at which pixel transitions from 
    the starting class to the ending class.

    Parameters
    ----------
    Transition_code: float
        Value of pixel from delta_class_ts at which pixel transitions from 
        the starting class to the ending class. 
    
    Returns
    -------
    start_class: float
        Value of starting class
    end_class: float
        Value of ending class
    """
    
    end_class = (transition_code % 100)-1
    start_class = (transition_code-end_class-1)/100-1
    
    return start_class, end_class



def plot_ndvi_ts(ndvi_ts, times, easting_vec,northing_vec, min_col,max_col,color_mapping,folder_loc):
    """
    Plot ndvi_ts data. 

    Parameters
    ----------
    ndvi_ts: Array of float64, size=(t,y,x)
        Time series NDVI data. Dimension t,y and x 
        correspond to the time, pixel row and pixel column, 
        respectively.
    times: array of datetime objects, size=(n,1)
        An array of the dates/times associated with each time series image.
    easting_vec: Array of float64, size=(x,)
        Easting postion (m) of each column of the imagery.  
    northing_vec: Array of float64, size=(x,)
        Northing postion (m) of each row of the imagery.
    min_col: float
        Minimum value of color bar.
    max_col: float
        Maximum value of color bar.
    color_mapping: str
        Color map str.
    folder_loc : str
        The location of the folder that contains the PSScene4Band time series 
        data to be merged (forward slashes only, ends in forward slash).
    """

    img_dim = np.shape(ndvi_ts)
    is_exist = os.path.exists(folder_loc+'Output/NDVI/')
    if not is_exist:
        os.makedirs(folder_loc+'Output/NDVI/')
    for i in range(img_dim[0]):
        temp_img = ndvi_ts[i,:,:]
        x = easting_vec
        y = northing_vec
        date_time = times[i][0].strftime("%m/%d/%Y, %H:%M:%S")
        plt.imshow(temp_img,interpolation='none',vmin = min_col, vmax = max_col, cmap = color_mapping,extent = [x.min(), x.max(), y.min(), y.max()])
        plt.title(date_time)
        plt.xlabel("Easting (m)- WGS 84 / UTM zone 48S")
        plt.xticks(rotation = 45)
        plt.ylabel("Northing (m)- WGS 84 / UTM zone 48S")
        plt.colorbar()
        plt.savefig(folder_loc+'Output/NDVI/'+'NDVI_Date_'+str(i+1), dpi=200, bbox_inches='tight', pad_inches=0.7)
        plt.close('all')
        #plt.show()



def plot_classification(class_ts, times, easting_vec,northing_vec,folder_loc,class_names):
    """
    Plot classified time series data 

    Parameters
    ----------
    class_ts: Array of float64, size= (t,y,x)
    Classifed time series data. Value of 1:j represent classes of 1:j, 
    respectively. A value of zero indicates the pixel does not fall in any
    of these classes. 
    times: array of datetime objects, size=(n,1)
        An array of the dates/times associated with each time series image.
    easting_vec: Array of float64, size=(x,)
        Easting postion (m) of each column of the imagery.  
    northing_vec: Array of float64, size=(x,)
        Northing postion (m) of each row of the imagery.
    folder_loc : str
        The location of the folder that contains the PSScene4Band time series 
        data to be merged (forward slashes only, ends in forward slash).
    class_names= list of strings size=3
        List of strings containing the names of the classes
    """

    img_dim = np.shape(class_ts)
    new_colors = np.array([[0.7, 0.75, .75, 1],[0.9176,0.8667,0.7922, 1],[0,0.7059,0.3412,1]])
    newcmp = ListedColormap(new_colors)
    values = np.unique(class_ts.ravel())
    is_exist = os.path.exists(folder_loc+'Output/Classification/')
    if not is_exist:
        os.makedirs(folder_loc+'Output/Classification/')

    for i in range(img_dim[0]):
        temp_img = class_ts[i,:,:]
        x = easting_vec
        y = northing_vec
        date_time = times[i][0].strftime("%m/%d/%Y, %H:%M:%S")
        im=plt.imshow(temp_img,interpolation='none',vmin = 0, vmax = 3, cmap = newcmp,extent = [x.min(), x.max(), y.min(), y.max()])
        plt.title(date_time)
        plt.xlabel("Easting (m)- WGS 84 / UTM zone 48S")
        plt.xticks(rotation = 45)
        plt.ylabel("Northing (m)- WGS 84 / UTM zone 48S")

        # get the colors of the values, according to the unique values
        # colormap used by imshow
        colors = [ im.cmap(im.norm(value)) for value in values[range(len(values)-1)]]

        # create a patch (proxy artist) for every color
        patches = [ mpatches.Patch(color=colors[j], label = class_names[j]) for j in range(len(values)-1) ]

        # put those patched as legend-handles into the legend
        plt.legend(handles = patches, bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0. )
        plt.savefig(folder_loc+'Output/Classification/'+'Classification_Date_'+str(i+1), dpi=200, bbox_inches='tight', pad_inches=0.7)
        plt.close('all')
        #plt.show()

def plot_class_diff(delta_class_ts, times, easting_vec,northing_vec,folder_loc,class_names,mask_ts):
    """
    Plot classified time series data 

    Parameters
    ----------
    class_ts: Array of float64, size= (t,y,x)
    Classifed time series data. Value of 1:j represent classes of 1:j, 
    respectively. A value of zero indicates the pixel does not fall in any
    of these classes. 
    times: array of datetime objects, size=(n,1)
        An array of the dates/times associated with each time series image.
    easting_vec: Array of float64, size=(x,)
        Easting postion (m) of each column of the imagery.  
    northing_vec: Array of float64, size=(x,)
        Northing postion (m) of each row of the imagery.
    folder_loc : str
        The location of the folder that contains the PSScene4Band time series 
        data to be merged (forward slashes only, ends in forward slash).
    class_names= list of strings size=3
        List of strings containing the names of the classes
    """

    img_dim = np.shape(delta_class_ts)
    is_exist = os.path.exists(folder_loc+'Output/Change_Detection/')
    if not is_exist:
        os.makedirs(folder_loc+'Output/Change_Detection/')   
        
    permutations_classes = np.array(np.meshgrid(range(len(class_names)), range(len(class_names)))).T.reshape(-1,2)
    class_names2 = []
    for i in range(len(permutations_classes)):
        temp_str = class_names[permutations_classes[i,0]]+' to '+class_names[permutations_classes[i,1]]
        class_names2.append(temp_str)
        
    #generate change detection images with int values incrementing by 1
    img_dim2=np.shape(mask_ts)
    delta_class_ts2 = np.zeros([len(times)-1,img_dim2[0],img_dim2[1]])    
    for i in range(len(permutations_classes)):
        indx = delta_class_ts == class_2_transition_code(permutations_classes[i,0],permutations_classes[i,1])
        delta_class_ts2[indx] = i
    delta_class_ts2 = apply_mask(mask_ts,delta_class_ts2)
    values = range(len(permutations_classes))
    
    
    for i in range(img_dim[0]):
        temp_img = delta_class_ts2[i,:,:]
        x = easting_vec
        y = northing_vec
        date_time = times[i][0].strftime("%m/%d/%Y, %H:%M:%S")
        date_time2 = times[i+1][0].strftime("%m/%d/%Y, %H:%M:%S")
        im = plt.imshow(temp_img,interpolation = 'none',vmin = 0, vmax = np.max(values), cmap = 'tab20c',extent = [x.min(), x.max(), y.min(), y.max()])
        plt.title("Date 1: "+ date_time+'\n Date 2: ' +date_time2)
        plt.xlabel("Easting (m)- WGS 84 / UTM zone 48S")
        plt.xticks(rotation = 45)
        plt.ylabel("Northing (m)- WGS 84 / UTM zone 48S")

        # get the colors of the values, according to the unique values
        # colormap used by imshow
        colors = [ im.cmap(im.norm(value)) for value in values]

        # create a patch (proxy artist) for every color
        patches = [ mpatches.Patch(color=colors[j], label= class_names2[j]) for j in range(len(values)) ]

        # put those patched as legend-handles into the legend
        plt.legend(handles = patches, bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0. )
        plt.savefig(folder_loc+'Output/Change_Detection/'+'Classification_Date_'+str(i+1)+'_to_Date_'+str(i+2), dpi=200, bbox_inches='tight', pad_inches=0.7)
        plt.close('all')
        #plt.show()
        