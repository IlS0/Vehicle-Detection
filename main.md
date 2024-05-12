Module src.main
===============
This module handles the interaction between the `detector` and  `loader` modules, and saves the detection results in the same format as the input (picture or video).

Classes
-------

`Main()`
:   A class for program modules management.
    
    Attributes:
        loader (Loader): Command line arguments loader.
        input (MatLike | VideoCapture):  Input.
        model_path (str): Path to a model file.
        isVideo(bool): Input type flag.
        img_size(int): Size of an input image.
        detector (Detector): Object detector.
    
    Initializes the Main object.
    
    Args:
       None
    
    Returns:
        None