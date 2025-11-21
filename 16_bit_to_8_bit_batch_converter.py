# ### __Training dataset batch pre-processing (1)__

# #### __conversion from 16-bit to 8-bit images__
# * __high-content microscope outputs 16-bit images__
# >- __not suitable__ __for__ cell annotations in __cvat.ai__ 
# * conversion done __in batches__
# * __define source__ and __output folders__

# #### __libraries__
import cv2 as cv
import numpy as np
import pandas as pd
import os

# #### __inputs__
#16-bit images (to be converted)
source_folder= r""

#storage for 8-bit images (converted)
output_folder= r""

# #### __function__
def _16_to_8_bit_batch_converter(input_folder, output_folder, alpha= 1, beta= 0):
    
    for file in os.listdir(input_folder):
        try:
            #load the 16-bit image
            path_to_image= os.path.join(input_folder, file)
            img16= cv.imread(path_to_image, cv.IMREAD_UNCHANGED)
            
            #0-255 rescaling + 8-bit unsined format conversion
            min_max_norm = cv.normalize(img16, None, 0, 255, cv.NORM_MINMAX)
            img8 = min_max_norm.astype('uint8')
            
            #modify contrast and brightness
            img8 = cv.convertScaleAbs(img8, alpha=alpha, beta=beta)
            
            path_to_output= os.path.join(output_folder, file)
            cv.imwrite(path_to_output, img8)
            print(f'file {file} processed')
        
        except Exception as ex:
            print(f'file {file} skipped: {ex}')


# #### __batch conversion__
# _16_to_8_bit_batch_converter(source_folder,
#                              output_folder,
#                              2,
#                              100)







