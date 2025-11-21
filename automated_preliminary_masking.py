# ## __Preliminary Cell Segmentation__

# #### __Libraries and Image Path__

# * __Libraries__
import cv2 as cv
import numpy as np
import pandas as pd
import os
from zipfile import ZipFile
import matplotlib.pyplot as plt

# * __image path__
path_to_image_gfp= r""

# #### __Functions__

# * __individual image-processing functions__
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
        
def contrast_brightness_adjustment(image, alpha= 1, beta= 0):
    try:
        #modify contrast and brightness
        img = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
        return img
    except Exception as ex:
        raise RuntimeError(f"Adjustments failed. Original error: {ex}")
        
def background_correction(image, kernel_dim=151):
    try:
        background= cv.GaussianBlur(image, (kernel_dim, kernel_dim), 0)
        image= cv.subtract(image, background)
        return image
    except Exception as ex:
        raise RuntimeError(f"Background correction failed. Original error: {ex}")        
        
def gaussian_denoising(image, kernel_dim=9):
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
        
def manual_thresholding(image, thr):
    try:
        _, mask = cv.threshold(image, thr, 1, cv.THRESH_BINARY)
        return mask
    except Exception as ex:
        raise RuntimeError(f"Manual binarisation failed. Original error: {ex}") 
        
def erode_clean(image, kernel_dim):
    try:
        kernel= np.ones((kernel_dim, kernel_dim), np.uint8)
        cleaned = cv.erode(image, kernel)
        return cleaned
    except Exception as ex:
        raise RuntimeError(f"Erode cleaning failed. Original error: {ex}") 
        
def dilate_fill(image, kernel_dim):
    try:
        kernel= np.ones((kernel_dim, kernel_dim), np.uint8)
        filled = cv.dilate(image, kernel)
        return filled
    except Exception as ex:
        raise RuntimeError(f"Dilate filling failed. Original error: {ex}") 
        
def cell_separation(image, threshold_fraction_of_maximum= 0.5):
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
def preliminary_cell_segmentation(path_to_image, 
                                  kernel_for_background_correction= 151, 
                                  contrast= 2, 
                                  brightness= 200, 
                                  kernel_for_gaussian_blur_denoising= 25, 
                                  binarisation= 'percentile', 
                                  percentile_for_binarisation= 80,
                                  erode_cleaning_kernel_dimension= 5,
                                  dilate_filling_kernel_dimension= 10,
                                  threshold_for_cell_center_areas= 0.5):
    try:
        #load and 16- to 8-bit convert
        input_image_8bit= img_load_and_16_to_8_bit_conversion(path_to_image)
        #background correction
        mask= background_correction(input_image_8bit, kernel_for_background_correction)
        #contrast and brightness adjustment (optional, default: no adjustment)
        mask= contrast_brightness_adjustment(mask, contrast, brightness)
        #gaussian blur denoising
        mask= gaussian_denoising(mask, kernel_for_gaussian_blur_denoising)
        #binarisation
        if binarisation== 'percentile':
                mask= manual_thresholding(mask, np.percentile(mask, percentile_for_binarisation))
        elif binarisation== 'otsu':
               mask= otsu_thresholding(mask)
        else:
            raise ValueError(f"Invalid binarisation input: '{binarisation}'. Expected: 'percentile', or 'otsu'.")
        #erode cleaning
        mask= erode_clean(mask, erode_cleaning_kernel_dimension)
        #dilate filling
        mask= dilate_fill(mask, dilate_filling_kernel_dimension)
        #cell separation
        mask= cell_separation(mask, threshold_for_cell_center_areas)
        
        return input_image_8bit, mask
    except Exception as ex:
        raise RuntimeError(f"Preliminary cell segmentation FAILED. Original error: {ex}")


# * __export for CVAT: original image plus individual-object masks__
def export_for_CVAT(output_pathway= r"",
                    folder_name= 'test_folder',
                    image_name= 'Image1',
                    original_image= original_image,
                    masks= mask):

    if not os.path.exists(output_pathway):
        raise FileNotFoundError(f"Path does not exist: {output_pathway}")

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
    with ZipFile(zip_path, 'w') as zipf: 
        for root, _, files in os.walk(output_folder): 
            for file in files: 
                full_path = os.path.join(root, file) 
                arcname = os.path.relpath(full_path, output_folder) 
                zipf.write(full_path, arcname)


# ## __Single-image segmentation__

# * __preliminary segmentation__
original_image, mask= preliminary_cell_segmentation(path_to_image_gfp)

# * __test visual__
fig, ax= plt.subplots(1, 2, figsize= (30, 15))

ax[0].imshow(contrast_brightness_adjustment(original_image, 2, 100),
             cmap= 'gray')
ax[0].set_title('original image', weight= 'bold', fontsize= 16)

ax[1].imshow(mask,
             cmap= 'gray')
ax[1].set_title('segmented cells', weight= 'bold', fontsize= 16)

# * __export for CVAT__
export_for_CVAT()

