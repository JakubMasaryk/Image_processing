#!/usr/bin/env python
# coding: utf-8

# ## __Microscopy image watermarking__
# 
# * __labels microscopy image or registered video with GU logo__
# * __Inputs:__ 
# >- __single image:__ pathway to single grayscale microscopy image (either 8- or 16-bit)
# >- __video:__ pathway to registered video (see https://github.com/JakubMasaryk/Image_processing_for_YOLO/blob/main/video_stackreg.py )
# >- __both:__ pathway to logo (RGB image)
# * __Other Inputs:__ set to default, modify if needed

# #### __Libraries__

# In[1]:


import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### __Single Image__

# * __inputs__

# In[75]:


image_path= r"C:\Users\Jakub\Desktop\CNN_training_images\class2_individual_aggregates\8bit_for_cvat_annotation\20240502_D17_s1_w2.TTimePoint_19.TIF"
logo_path= r"C:\Users\Jakub\Documents\opencv_course_imgs\university_of_gothenburg_logo.jpg"


# * __function__

# In[77]:


def single_image_watermark(image_pathway, #path to input image
                           logo_pathway, #path to logo
                           logo_dimensions_x_y= (200, 200), #imprinted logo dimensions
                           logo_intensity= 200, #intensity on a grayscale 0-255
                           logo_position= 'upper right', #logo positioning within image
                           invert_logo= False, #invert, applicable if original background white and logo dark
                           export= True):
    
    #check the paths
    if not os.path.exists(image_pathway):
        raise FileNotFoundError(f"input-image file does not exist: {image_pathway}")    
    if not os.path.exists(logo_pathway):
        raise FileNotFoundError(f"input-logo file does not exist: {logo_pathway}")
        
    #generate and export path
    input_directory= os.path.dirname(image_pathway)
    input_file= os.path.basename(image_pathway)
    output_file= 'watermarked_' + input_file
    export_pathway= os.path.join(input_directory, output_file)
    
    #load the image  
    image= cv.imread(image_pathway,
                     cv.IMREAD_UNCHANGED)
    #convert image to 8-bit (if needed)
    if image.dtype == np.uint16:
        image = (image >> 8).astype(np.uint8)
    
    #load the logo
    logo= cv.imread(logo_pathway,
                    cv.IMREAD_GRAYSCALE)
    #resize the logo
    logo= cv.resize(logo, (logo_dimensions_x_y[0], logo_dimensions_x_y[1]))
    
    #logo positioning
    if logo_position== 'upper left':
        x_offset= 0
        y_offset= 0
        x_end= x_offset + logo.shape[1]
        y_end= y_offset + logo.shape[0]
    
    elif logo_position== 'upper right':
        x_offset= image.shape[1] - logo.shape[1]
        y_offset= 0
        x_end= x_offset + logo.shape[1]
        y_end= y_offset + logo.shape[0]
        
    elif logo_position== 'lower left':
        x_offset= 0
        y_offset= image.shape[0] - logo.shape[0]
        x_end= x_offset + logo.shape[1]
        y_end= y_offset + logo.shape[0]
        
    elif logo_position== 'lower right':
        x_offset= image.shape[1] - logo.shape[1]
        y_offset= image.shape[0] - logo.shape[0]
        x_end= x_offset + logo.shape[1]
        y_end= y_offset + logo.shape[0]
    else:
        raise ValueError(f"Invalid logo_position argument: '{logo_position}'. Valid: 'upper left'/'upper right'/'lower left' or 'lower right'.")   
    
    try:
        #define ROI
        roi= image[y_offset:y_end, x_offset:x_end]

        #mask the logo by Otsu's
        thr, logo_mask = cv.threshold(logo, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        #invert the logo mask (if needed)
        if invert_logo== True:
            logo_mask= 255-logo_mask
            
        #imprint logo on roi, based on the logo mask and intensity (0-255)
        roi[logo_mask==255]= logo_intensity
        
        #put into original image (not needed, roi corrsponding section in image updated automatically)
        # image[y_offset:y_end, x_offset:x_end]= roi
        
    except Exception as ex:
        raise RuntimeError(f"Image watermarking FAILED. Original error: {ex}")
    
    if export==True:
        #export
        cv.imwrite(export_pathway,
                   image)
    else:
        return image


# * __single-image watermarking__

# In[79]:


single_image_watermark(image_path, logo_path)


# #### __Video__

# * __inputs__

# In[67]:


video_path= r"C:\Users\Jakub\Documents\opencv_course_imgs\registered_test_video.avi"
logo_path= r"C:\Users\Jakub\Documents\opencv_course_imgs\university_of_gothenburg_logo.jpg"


# * __function__

# In[90]:


def video_watermark(video_pathway,
                    logo_pathway, #path to logo
                    logo_dimensions_x_y= (150, 150), #imprinted logo dimensions
                    logo_intensity= 50, #intensity on a grayscale 0-255
                    logo_position= 'upper right', #logo positioning within image
                    invert_logo= False, #invert, applicable if original background white and logo dark
                    export= True,
                    fps= 10):
    
    #check the paths
    if not os.path.exists(video_pathway):
        raise FileNotFoundError(f"input-image file does not exist: {video_pathway}")    
    if not os.path.exists(logo_pathway):
        raise FileNotFoundError(f"input-logo file does not exist: {logo_pathway}")
        
    #generate and export path
    input_directory= os.path.dirname(video_pathway)
    input_file= os.path.basename(video_pathway)
    output_file= 'watermarked_' + input_file
    export_pathway= os.path.join(input_directory, output_file)
    
        
    #load the logo
    logo= cv.imread(logo_pathway,
                    cv.IMREAD_GRAYSCALE)
    #resize the logo
    logo= cv.resize(logo, (logo_dimensions_x_y[0], logo_dimensions_x_y[1]), interpolation=cv.INTER_AREA)
    
    #mask the logo by Otsu's
    thr, logo_mask = cv.threshold(logo, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    #invert the logo mask (if needed)
    if invert_logo== True:
        logo_mask= 255-logo_mask
        
    #load up the video
    vid = cv.VideoCapture(video_pathway)
    if not vid.isOpened():
        raise ValueError(f"falied to open the video: {video_pathway}")
    
    #read the first frame of the input video
    ret, frame = vid.read()
    if not ret:
        raise RuntimeError("failed to read the first frame")
    
    #get the dimensions
    width, height= frame.shape[1], frame.shape[0]

    #logo positioning
    if logo_position== 'upper left':
        x_offset= 0
        y_offset= 0
        x_end= x_offset + logo.shape[1]
        y_end= y_offset + logo.shape[0]
    
    elif logo_position== 'upper right':
        x_offset= frame.shape[1] - logo.shape[1]
        y_offset= 0
        x_end= x_offset + logo.shape[1]
        y_end= y_offset + logo.shape[0]
        
    elif logo_position== 'lower left':
        x_offset= 0
        y_offset= frame.shape[0] - logo.shape[0]
        x_end= x_offset + logo.shape[1]
        y_end= y_offset + logo.shape[0]
        
    elif logo_position== 'lower right':
        x_offset= frame.shape[1] - logo.shape[1]
        y_offset= frame.shape[0] - logo.shape[0]
        x_end= x_offset + logo.shape[1]
        y_end= y_offset + logo.shape[0]
    
    # rewind to start
    vid.set(cv.CAP_PROP_POS_FRAMES, 0)
    
    # codec + videowriter
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    video_output = cv.VideoWriter(export_pathway, fourcc, fps, (width, height), isColor=False)
    if not video_output.isOpened():
        raise ValueError(f"failed to access the ouput path {export_pathway}")

    #for each frame convert to grayscale and add logo
    frame_index= 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        try:
            frame_index += 1
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            #define ROI
            roi= frame[y_offset:y_end, x_offset:x_end]

            #imprint logo on roi, based on the logo mask and intensity (0-255)
            roi[logo_mask==255]= logo_intensity

            #frame output
            video_output.write(frame)
    
        except Exception as ex:
            print(f"watermarking failed on frame {frame_index}, error: {ex}")
    
    vid.release()
    video_output.release()


# * __video watermarking__

# In[64]:


video_watermark(video_path,
                logo_path)


# In[ ]:




