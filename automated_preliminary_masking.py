# ## __Preliminary Cell Segmentation__
# * __transforms microscopy image (single channel, 16-bit) into 8-bit, single channel image plus mask for each object (cell)__
# * __masking achieved by intensity-based thresholding, object-center identification and watershed-based object separation__
# * __for single image:__
# >- define __'path_to_image_gfp'__ and use __'single_image_processing'__ function
# * __for batch of multiple images:__
# >- define __'path_to_folder_images_gfp'__ and use __'batch_processing'__ function


# #### __Libraries__
import cv2 as cv
import numpy as np
import pandas as pd
import os
import zipfile 
import matplotlib.pyplot as plt
import shutil


# #### __Inputs__
# * __input pathways__
# >- pathway to a __single image or__ to a __folder__ containing multiple images for batch processing
path_to_image_gfp= r"C:\Users\Jakub\Desktop\CNN_training_images\class3_CAAC\16bit_for_training\20241205_A04_w2TimePoint_55.TIF"
path_to_folder_images_gfp= r"C:\Users\Jakub\Desktop\test_input"

# * __output pathway__
output_path= r"C:\Users\Jakub\Desktop"


# * __image-processing parameters__
# >- used as a __default__ arguments for __individual__ image-processing and object-separating __functions__
# >- __also__ applied on the __compiled function__
#dimension of a kernel for uneven-background correction
kernel_for_background_correction= 151

#alpha for contrast adjustment
contrast= 2

#beta for brightness adjutment
brightness= 51

#dimension of a kernel for gaussian denoising
kernel_for_gaussian_blur_denoising= 51

#threshold for image binarisation as a percentile of all intensities
percentile_for_binarisation= 75

#kernel dimension for erode cleaning and dilate filling
erode_cleaning_kernel_dimension= 5
dilate_filling_kernel_dimension= 20

#threshold for center areas of objects (proportion of max distance from background)
threshold_for_cell_center_areas= 0.5


# #### __Functions__
# * __load and 16- to 8-bit conversion__
#input: pathway to a single image (single channel, grayscale, 16-bit)
#output: grayscale, 8-bit image
def img_load_and_16_to_8_bit_conversion(path):
    try:
        #load
        img16= cv.imread(path, cv.IMREAD_UNCHANGED)
        #0-255 rescaling + 8-bit unsined format conversion
        min_max_norm = cv.normalize(img16, None, 0, 255, cv.NORM_MINMAX)
        img8 = min_max_norm.astype('uint8')
        return img8
    except Exception as ex:
        raise RuntimeError(f"Failed to load and/or convert the image to 8-bit. Original error: {ex}")

# * __image processing__
#input: grayscale or binary image
#output: processed grayscale or binary image
def contrast_brightness_adjustment(image, alpha= contrast, beta= brightness):
    try:
        #modify contrast and brightness
        img = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
        return img
    except Exception as ex:
        raise RuntimeError(f"Adjustments failed. Original error: {ex}")
        
def background_correction(image, kernel_dim= kernel_for_background_correction):
    try:
        background= cv.GaussianBlur(image, (kernel_dim, kernel_dim), 0)
        image= cv.subtract(image, background)
        return image
    except Exception as ex:
        raise RuntimeError(f"Background correction failed. Original error: {ex}")        
        
def gaussian_denoising(image, kernel_dim= kernel_for_gaussian_blur_denoising):
    try:
        image= cv.GaussianBlur(image, (kernel_dim, kernel_dim), 0)
        return image
    except Exception as ex:
        raise RuntimeError(f"De-noising failed. Original error: {ex}") 
        
def otsu_thresholding(image):
    try:
        thr, mask = cv.threshold(image, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
        print(f'threshold: {thr}')
        return mask
    except Exception as ex:
        raise RuntimeError(f"Otsu's binarisation failed. Original error: {ex}") 
        
def manual_thresholding(image, thr= percentile_for_binarisation):
    try:
        _, mask = cv.threshold(image, np.percentile(image, thr), 1, cv.THRESH_BINARY)
        return mask
    except Exception as ex:
        raise RuntimeError(f"Manual binarisation failed. Original error: {ex}") 
        
def erode_clean(image, kernel_dim= erode_cleaning_kernel_dimension):
    try:
        kernel= np.ones((kernel_dim, kernel_dim), np.uint8)
        cleaned = cv.erode(image, kernel)
        return cleaned
    except Exception as ex:
        raise RuntimeError(f"Erode cleaning failed. Original error: {ex}") 
        
def dilate_fill(image, kernel_dim= dilate_filling_kernel_dimension):
    try:
        kernel= np.ones((kernel_dim, kernel_dim), np.uint8)
        filled = cv.dilate(image, kernel)
        return filled
    except Exception as ex:
        raise RuntimeError(f"Dilate filling failed. Original error: {ex}") 

# * __object separation__
#input: binary image
#output: binary image with separated cells
def cell_separation(image, threshold_fraction_of_maximum= threshold_for_cell_center_areas):
    try:
        #distance-from-background matrix
        dist_matrix = cv.distanceTransform(image, cv.DIST_L2, 0)
        #center area 
        #area where value/distance-from-background is a set proportion of center (max) distance from the background
        center_areas = (dist_matrix > threshold_fraction_of_maximum * dist_matrix.max()).astype(np.uint8)
        #label each center area by a number- the entire blob is a area of pixels of a certain value 1, 2, 3 etc...background is 0
        count, labels = cv.connectedComponents(center_areas)
        #add +1, background 1, center areas 2,3,4...
        labels = labels + 1
        #reverse background around center areas to 0
        labels[center_areas == 0] = 0
        # convert mask to RGB
        color_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        #watershed
        #outputs 2Darray (not rgb image)
        cell_masks = cv.watershed(color_image, labels)
        #back to 0/1 binary matrix
        #background also treated as a one big blob by the watershed...does not remain 0 (defined by the input_image)
        #connection lines between cells or between cells and background are -1
        cell_masks_binary= np.uint8((cell_masks > 1) & (image > 0))
        return cell_masks_binary
    except Exception as ex:
        raise RuntimeError(f"Cell separation FAILED. Original error: {ex}")

# * __compiled function: input image to original image and binary mask__
# >- __input:__ path to a __single image__ (16-bit)
# >- __output:__ __original image__ (8-bit) plus __segmented image__ (masks for cells)
# >- __used for both__ the __single-image__ and __batch__ processing
def preliminary_cell_segmentation(path_to_image,
                                  binarisation= 'percentile'):
    try:
        #load and 16- to 8-bit convert
        input_image_8bit= img_load_and_16_to_8_bit_conversion(path_to_image)
        #background correction
        mask= background_correction(input_image_8bit)
        #contrast and brightness adjustment (optional, default: no adjustment)
        mask= contrast_brightness_adjustment(mask)
        #gaussian blur denoising
        mask= gaussian_denoising(mask)
        #binarisation
        if binarisation== 'percentile':
                mask= manual_thresholding(mask)
        elif binarisation== 'otsu':
               mask= otsu_thresholding(mask)
        else:
            raise ValueError(f"Invalid binarisation input: '{binarisation}'. Expected: 'percentile', or 'otsu'.")
        #erode cleaning
        mask= erode_clean(mask)
        #dilate filling
        mask= dilate_fill(mask)
        #cell separation
        mask= cell_separation(mask)
        
        return input_image_8bit, mask
    except Exception as ex:
        raise RuntimeError(f"Preliminary cell segmentation FAILED. Original error: {ex}")

# * __segmentation and export for single image__
# >- __input: single image__
# >- __output:__ folder with __original image__ (now 8-bit) __and__ a separate __mask for each object__
def single_image_processing(path_to_image,
                            output_pathway,
                            folder_name= 'test_folder',
                            image_name= 'Image1'):
    
    if not os.path.exists(path_to_image):
        raise FileNotFoundError(f"Path does not exist: {path_to_image}")

    if not os.path.exists(output_pathway):
        raise FileNotFoundError(f"Path does not exist: {output_pathway}")
        
    original_image, masks= preliminary_cell_segmentation(path_to_image)    

    output_folder= os.path.join(output_pathway, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    object_count, individual_objects = cv.connectedComponents(masks)

    image_dir= os.path.join(output_folder, 'images')
    masks_dir= os.path.join(output_folder, 'masks')
    masks_subdir= os.path.join(masks_dir, image_name)

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_subdir, exist_ok=True)

    cv.imwrite(os.path.join(image_dir, f'{image_name}.png'), contrast_brightness_adjustment(original_image, 3.5, 25))

    for object_id in range(1, object_count):
        try:
            individual_object= (individual_objects==object_id).astype(np.uint8)*255
            file_name= f'{object_id:04d}.png'
            cv.imwrite(os.path.join(masks_subdir, file_name), individual_object)
        except Exception as ex:
            print(f'object {object_id} SKIPPED, error: {ex}')
                
    zip_path = os.path.join(output_pathway, f'{folder_name}.zip') 
    with zipfile.ZipFile(zip_path, 'w') as zipf: 
        for root, _, files in os.walk(output_folder): 
            for file in files: 
                full_path = os.path.join(root, file) 
                arcname = os.path.relpath(full_path, output_folder) 
                zipf.write(full_path, arcname)
   
    print('single-image processing finished')    

# * __segmentation and export for batch processing__
# >- __folder__ with __multiple images__ to be segmented and exported
def batch_processing(path_to_input_folder,
                     output_path,
                     output_folder_name= 'output_folder',
                     _zip= True):
    
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Path does not exist: {output_path}")
    output_folder_path= os.path.join(output_path, output_folder_name)
    os.makedirs(output_folder_path, exist_ok=True)
    
    image_dir= os.path.join(output_folder_path, 'images')
    masks_dir= os.path.join(output_folder_path, 'masks')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    for file in os.listdir(path_to_input_folder):
        img_path = os.path.join(path_to_input_folder, file)
        filename_stripped = os.path.splitext(file)[0]
        masks_subdir= os.path.join(masks_dir, filename_stripped)
        os.makedirs(masks_subdir, exist_ok=True)
        
        try:
            image, mask= preliminary_cell_segmentation(img_path)
            
            cv.imwrite(os.path.join(image_dir, f'{filename_stripped}.png'), contrast_brightness_adjustment(image, 2, 25))
            
            object_count, individual_objects =cv.connectedComponents(mask)
            for object_id in range(1, object_count):
                try:
                    individual_object= (individual_objects==object_id).astype(np.uint8)*255
                    file_name= f'{object_id:04d}.png'
                    cv.imwrite(os.path.join(masks_subdir, file_name), individual_object)
                except Exception as ex:
                    print(f'object {object_id} SKIPPED, error: {ex}')
            
        except Exception as ex:
            print(f'file {file} SKIPPED: {ex}')
            
    if _zip:
        zip_path = os.path.join(output_path, output_folder_name)
        shutil.make_archive(zip_path, 'zip', output_folder_path)
        
    print('batch processing finished')


# ## __Single-image segmentation__
# * __test visual (optional)__
# original_image, mask= preliminary_cell_segmentation(path_to_image_gfp)

# fig, ax= plt.subplots(1, 2, figsize= (30, 15))

# ax[0].imshow(original_image,
#              cmap= 'gray')
# ax[0].set_title('original image', weight= 'bold', fontsize= 16)

# ax[1].imshow(mask,
#              cmap= 'gray')
# ax[1].set_title('segmented cells', weight= 'bold', fontsize= 16)

# * __segmentation__
single_image_processing(path_to_image_gfp,
                        output_path)


# ## __Batch processing__
batch_processing(path_to_folder_images_gfp,
                 output_path)
