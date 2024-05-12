Module src.detector
===================
This module implements an algorithm of object detection of the 9 different classes of vehicles. It also makes preprocessing and postprocessing of an input image.

Classes
-------

`VehicleDetector(model_path, size)`
:   A class to vehicle detection. 
    
    Attributes:
        size (int): Size of an input image.
    
    Methods:
        draw_box(img): Draws bounding boxes on the input image.
        video_run(cap): Processes video frames.
    
    Initializes the VehicleDetector object.
    
    Args:
        model_path (str): Path to a model file.
        size (tuple): Size of an input image.
    
    Returns:
        None

    ### Methods

    `draw_box(self, img)`
    :   Draws bounding boxes on the input image.
        
        Args:
            img (numpy.ndarray): Input image.
        
        Returns:
            numpy.ndarray: Image with drawn bounding boxes.

    `video_run(self, cap)`
    :   Processes video frames.
        
        Args:
            cap (VideoCapture): Video capture object.
        
        Returns:
            io.BytesIO: In-memory file containing the processed video.