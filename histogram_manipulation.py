# ### __Histogram Modifications__

# #### __Description__
# * __application of__ the basic histogram-modification techniques (__stretching, equalization or CLAHE__)
# * __applied on__ a __single image__ (_'histogram_manipulation_single_frame'_) __or batch__ of images (_'histogram_manipulation_batch'_) 
# * __maintains__ the input-image(s) __format__

# #### __Input/Outut__
# * __Input:__  __pathway to single image__ ('Single Image' section) __or folder__ ('Multiple Images' section) with multiple images (__8- or 16-bit image(s)__)
# * __Output:__ __single processed image__ (labeled by prefix and original name, within specified folder) __or batch of processed inamges__ in a new folder (creatd within specified folder)

# #### __Arguments__
# * __path_to_image/path_to_image_folder__: pathway to the single image or folder with multiple images
# * __histogram_manipulation__: type of histogram manipulation (_default= 'stretch'_)
# >- __'stretch'__: histogram stretching (supports both 8- and 16-bit images, generally the MOST SUITABLE)
# >- __'equalize'__: histogram equalization (supports only 8-bit images, 16-bit input converted to 8-bit, NOT RECOMMENDED)
# >- __'clahe'__: __C__ ontrast __L__ imited __A__ daptive __H__ istogram __E__ qualization (supports only 8-bit images, 16-bit input converted to 8-bit,)
# * __clahe_clip_limit__: int, contrast clipping, see docs (_default= '2_)
# * __clahe_grid_size__: int, tile size, see docs (default= 15) 
# * __export__: True/False (default= False), always __keep as False for 'histogram_manipulation_batch' function__ (automatic iterative export)
# >- __'True'__: exports the processed image (to the specified folder, with modification-based prefix)
# >- __'False'__: returns an image-array

# #### __Histogram Manipulations__
# ![Alt text](histogram_manipulations.png)

# #### __Libraries__
import cv2 as cv
import numpy as np
import os
from pystackreg import StackReg
import matplotlib.pyplot as plt
from pathlib import Path


# #### __Functions__
#single image
def histogram_manipulation_single_frame(path_to_image,
                                        histogram_manipulation= 'stretch',
                                        clahe_clip_limit= 2,
                                        clahe_grid_size= 15,
                                        export= False):
    
    #check input path
    if not os.path.exists(path_to_image):
        raise FileNotFoundError(f"input path does NOT exist: {path_to_image}")
        
    #generate and output path and folder (from the input path with file-name prefix 'stretch_', 'equalize_' or 'clahe_')
    #preserves the input format (usually .TIFF)
    try:
        folder = os.path.dirname(path_to_image)   #input folder
        filename = os.path.basename(path_to_image)  #input file name (original image)
        output_filename= f'{histogram_manipulation}_' + filename #output filename
        output_path= os.path.join(folder, output_filename) #output path
    except Exception as ex:
        raise RuntimeError(f"failed to create output path '{output_path}': {ex}")
    
    #load the single image
    image= cv.imread(path_to_image, 
                     cv.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError('failed to load image')
    
    #identify the pixel depth through dtype (8-bit: np.uint8 and 16-bit: np.uint16)
    img_dtype= image.dtype
    if img_dtype not in (np.uint8, np.uint16):
        raise ValueError(f'unsupported pixel depth: only uint8 and uint16 supported') 
    
    #get the general dtype parameters (min, max and dtype), specific for np.unit8 and np.uint16 
    #accessible by dtype_info.min/dtype_info.max and dtype_info.dtype
    dtype_info= np.iinfo(img_dtype)
    
    ##histogram manipulation
    #stretching
    if histogram_manipulation== 'stretch':
        #image min/max
        img_min= image.min()
        img_max= image.max()
        #avoid division by 0
        if img_max == img_min:
            raise ValueError('unable to stretch the histogram: flat image, no intensity range')
        #convert to float
        image_float= image.astype(np.float32)
        #stretch the histogram (based on dtype_info)   
        try:
            modified_img= np.clip((image_float - img_min) / (img_max - img_min) * dtype_info.max, dtype_info.min, dtype_info.max).astype(dtype_info.dtype)
        except Exception as ex:
            raise ValueError(f"failed to stretch the hitogram: {ex}") 
    #equalization
    elif histogram_manipulation== 'equalize':
        if img_dtype== np.uint8:
            try:
                modified_img= cv.equalizeHist(image)
            except Exception as ex:
                raise ValueError(f'failed to equalize histogram of the 8-bit image: {ex}')
        else:
            try:
                image8 = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
                modified_img = cv.equalizeHist(image8)
                print('16-bit images not supported for histogram equalization, converted to 8-bit and equalized')
            except Exception as ex:
                raise ValueError(f'failed to equalize histogram of the 16-bit image: {ex}')        
    #CLAHE
    elif histogram_manipulation== 'clahe':
        clahe = cv.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_grid_size, clahe_grid_size))
        if img_dtype== np.uint8:
            try:
                modified_img = clahe.apply(image)
            except Exception as ex:
                raise ValueError(f'CLAHE failed (8-bit input): {ex}')
        else:
            try:
                image8 = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
                modified_img = clahe.apply(image8)
                print('16-bit images not supported for CLAHE, converted to 8-bit and CLAHE applied')
            except Exception as ex:
                raise ValueError(f'CLAHE failed (8-bit input): {ex}')
            
    #export or return image
    if export== True:
        #generate and output path and folder (from the input path with file-name prefix 'stretch_', 'equalize_' or 'clahe_')
        #preserves the input format (usually .TIFF)
        try:
            folder = os.path.dirname(path_to_image)   #input folder
            filename = os.path.basename(path_to_image)  #input file name (original image)
            output_filename= f'{histogram_manipulation}_' + filename #output filename
            output_path= os.path.join(folder, output_filename) #output path
            cv.imwrite(output_path, modified_img)
        except Exception as ex:
            raise RuntimeError(f"failed to create output path and export the modified image'{output_path}': {ex}")
    else:
        return modified_img
    
#batch of images (folder)
def histogram_manipulation_batch(path_to_image_folder,
                                 histogram_manipulation= 'stretch',
                                 clahe_clip_limit= 2,
                                 clahe_grid_size= 15,
                                 export= False):
    
    #check input path
    if not os.path.exists(path_to_image_folder):
        raise FileNotFoundError(f"input path does NOT exist: {path_to_image_folder}")
        
    #create an output folder within the input folder (prefix according to the histogram modification)
    try:
        output_pathfolder= os.path.join(path_to_image_folder, f'{histogram_manipulation}_output_folder')
        os.makedirs(output_pathfolder, exist_ok=True)
    except Exception as ex:
        raise RuntimeError(f"failed to createthe output folder '{output_path}': {ex}")
        
    for image in os.listdir(path_to_image_folder):
        #filter to only image files
        if Path(image).suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
            continue
        #create path for each image
        path_to_image_file= os.path.join(path_to_image_folder, image)
        #modify histogram for each image and export to the output file (within input file)
        try:
            modified_img= histogram_manipulation_single_frame(path_to_image= path_to_image_file,
                                                              histogram_manipulation= histogram_manipulation,
                                                              clahe_clip_limit= clahe_clip_limit,
                                                              clahe_grid_size= clahe_grid_size,
                                                              export= export)
            output_path= os.path.join(output_pathfolder, image) #full output path for specific image
            cv.imwrite(output_path, modified_img)
            
        except Exception as ex:
            print(f'file {image} skipped, error: {ex}')


# #### __Single Image__
# histogram_manipulation_single_frame(path_to_image= r"",
#                                     histogram_manipulation= 'stretch',
#                                     clahe_clip_limit= 2,
#                                     clahe_grid_size= 15,
#                                     export= True)


# #### __Multiple Images__
# histogram_manipulation_batch(r"",
#                              histogram_manipulation= 'stretch',
#                              clahe_clip_limit= 2,
#                              clahe_grid_size= 15,)
