# ## __StackReg for timelapse fluorescent microscopy (2)__ 

# #### __Description__
# * __registration__ (alignment or stabilisation) of the __live-cell timelapse images__ to a common reference
# * __correcting the frame-to-frame drift__ caused by an inaccurate stage repositioning of the high-content microscope
# * __input__ is a __folder with time-lapse microscopy images__, __output__ is a __folder with registered time-lapse microscopy images__ of the .TIFF format

# #### __Inputs__
# * __input_path__: pathway to the folder with a time-lapse microscopy images (to be registered)
# >- mind the __image labelling__...add __suffix '-timepoint_01', '-timepoint02'__ etc...
# * __reg_type__: type of registration (_default= 'rigid_body'_)
# >- __'translation'__: translation
# >- __'rigid_body'__: rigid body (translation + rotation)
# >- __'scaled_rotation'__: scaled rotation (translation + rotation + scaling)
# >- __'affine'__: affine (translation + rotation + scaling + shearing)
# >- __'bilinear'__: bilinear (non-linear transformation; does not preserve straight lines)
# * __reference__: reference for the alignment (_default= 'previous'_)
# >- __'first'__
# >- __'previous'__
# >- __'mean'__
# * __histogram_stretching__: True/False (default= False) 
# >- __True__: intensities rescaled 0-65535
# >- __False__: no stretching
# >- histogram already altered by the registration!
# * __rescale__: (default= 'single-frame')
# >- __'single-frame'__: min and max computed for each frame separately, every frame uses full intensity range, does NOT maintain temporal intensity changes
# >- __'global'__: mn and max computed for the whole stack of frames, some frames within a narrow intensity range, preserves temporal changes in intensity

# #### __Libraries__
import cv2 as cv
import numpy as np
import os
from pystackreg import StackReg


# --------------------------------------------------------------------------------

# #### __StackReg Function__
def stack_reg_folder(input_path,
                     reg_type= 'rigid_body',
                     reference= 'previous',
                     histogram_stretching= False,
                     rescale= 'global'):
    
    #generate and output path and folder (from the input path with prefix 'registered_')
    try:
        common_path, folder= os.path.split(input_path)
        output_folder= 'registered_' + folder
        output_path= os.path.join(common_path, output_folder)
        os.makedirs(output_path, exist_ok=True)
    except Exception as ex:
        raise RuntimeError(f"failed to create folder '{output_path}': {ex}")
    
    #select the registration type
    stack_reg_map = {'translation': StackReg.TRANSLATION,
                     'rigid_body': StackReg.RIGID_BODY,
                     'scaled_rotation': StackReg.SCALED_ROTATION,
                     'affine': StackReg.AFFINE,
                     'bilinear': StackReg.BILINEAR}
    if reg_type not in stack_reg_map.keys():
        raise ValueError(f"invalid 'reg_type' argument: '{reg_type}'; available: {list(stack_reg_map.keys())}")
        
    #check the reference argumnent
    if reference not in ['first', 'previous', 'mean']:
        raise ValueError(f"invalid 'reference' argument: '{reference}'; available: {['first', 'previous', 'mean']}")
    
    #check input path
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input path does NOT exist: {input_path}")
        
    #list to store individual frames (2D matrices)
    frames = []
    
    #iterate over images in folder
    for file in sorted(os.listdir(input_path)):
        path_to_file= os.path.join(input_path, file)
        
        try:
            frame= cv.imread(path_to_file, cv.IMREAD_UNCHANGED)
            if frame is None:
                continue
            # Convert to grayscale ONLY if needed
            if frame.ndim == 3:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frames.append(frame)
        except Exception as ex:
            print(f'file {file} skipped, error: {ex}')

    if len(frames) == 0:
        raise ValueError("folder contains NO readable frames")

    #stack individual frames into a 3D array (frame x height x width)
    video_array=  np.stack(frames, axis=0)
    
    #scale down to app. 0-1 range (actual range narrower) and convert to float32 (better accuracy and performance)
    video_array = video_array/65535
    video_array = video_array.astype(np.float32)
    
    #initialize selected registration type (based on stack_reg_map and reg_type argument)
    sr= StackReg(stack_reg_map[reg_type])
        
    #register frames 
    registered= sr.register_transform_stack(video_array, reference= reference)
    
    ###back to 16-bit range + export
    ##with histogram stretching
    if histogram_stretching== True:
        try:
            #frame-wise rescaling
            if rescale== 'single-frame':
                for i, frame in enumerate(registered):
                    frame_min = frame.min()
                    frame_max = frame.max()
                    frame= np.clip((frame - frame_min) / (frame_max - frame_min) * 65535, 0, 65535).astype(np.uint16)
                    path = os.path.join(output_path, f"frame_{i:04d}.tif")
                    cv.imwrite(path, frame)
            #global rescaling
            elif rescale== 'global':
                global_min = registered.min()
                global_max = registered.max()
                registered = np.clip((registered - global_min) / (global_max - global_min) * 65535, 0, 65535).astype(np.uint16)
                for i, frame in enumerate(registered):
                    path = os.path.join(output_path, f"frame_{i:04d}.tif")
                    cv.imwrite(path, frame)
            else:
                raise ValueError(f"Invalid rescale argument: '{rescale}'. Expected: 'single-frame' or 'global'.")
        except Exception as ex:
            raise ValueError(f"histogram stretching failed: {ex}")
    ##without histogram stretching
    else:
        try:
            registered = np.clip(registered * 65535, 0, 65535).astype(np.uint16)
            #export each frame
            for i, frame in enumerate(registered):
                path = os.path.join(output_path, f"frame_{i:04d}.tif")
                cv.imwrite(path, frame)
        except Exception as ex:
            raise ValueError(f"rescaling to 16-bit failed: {ex}")


# --------------------------------------------------------------------------------

# #### __StackReg__
# * with __default params__
# stack_reg_folder(input_path= r"")
