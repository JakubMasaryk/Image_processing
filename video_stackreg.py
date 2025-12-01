# ## __StackReg for timelapse fluorescent microscopy__ 

# #### __Description__
# * __registration__ (alignment or stabilisation) of the __live-cell timelapse images__ to a common reference
# * __correcting the frame-to-frame drift__ caused by an inaccurate stage repositioning of the high-content microscope
# * __input__ is a __video__ in common format (.avi), __outputs__ a __registered video__ of the same format

# #### __Inputs__
# * __input_path__: pathway to the video (to be registered)
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
# * __fps__: frames-per-second for the output video (_default= 10_)

# #### __Libraries__
import cv2 as cv
import numpy as np
import os
from pystackreg import StackReg

# --------------------------------------------------------------------------------

# #### __StackReg Function__
def stack_reg_video(input_path,
                    reg_type= 'rigid_body',
                    reference= 'previous',
                    fps= 10):
    
    #generate and output path
    input_directory= os.path.dirname(input_path)
    input_file= os.path.basename(input_path)
    output_file= 'registered_' + input_file
    output_path= os.path.join(input_directory, output_file)
    
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
    
    #load up the video
    vid = cv.VideoCapture(input_path)
    if not vid.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    #list to store individual frame (2D matrix)
    frames = []

    #for each frame convert to grayscale and add to the frames-list
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames.append(frame)
    vid.release()

    if len(frames) == 0:
        raise ValueError("video contains NO readable frames")

    #stack individual frame into a 3D array (frame x height x width)
    video_array=  np.stack(frames, axis=0)
    
    # Normalize across the whole stack
    min_int = video_array.min()
    max_int = video_array.max()
    video_array = (video_array - min_int) / (max_int - min_int) * 255
    video_array = video_array.astype(np.uint8)
    
    #initialize selected registration type (based on stack_reg_map and reg_type argument)
    sr= StackReg(stack_reg_map[reg_type])
        
    #register frames (+ clip and covert)
    registered= sr.register_transform_stack(video_array, reference= reference)
    registered = np.clip(registered, 0, 255).astype(np.uint8)
    
    #get frame dimensions
    _, height, width= registered.shape
    
    # codec + videowriter
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    registered_stack_output = cv.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    if not registered_stack_output.isOpened():
        raise ValueError(f"failed to access the ouput path {output_path}")

    #frame-by-frame writing
    for frame in registered:
        registered_stack_output.write(frame) 
    
    #release
    registered_stack_output.release()

# --------------------------------------------------------------------------------

# #### __StackReg__
# * with __default params__
stack_reg_video(input_path= r"C:\Users\Jakub\Desktop\test_video.avi")

