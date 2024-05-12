Module src.loader
=================
This module is responsible for correct parsing of the command line. It saves arguments necessary for correct launch and running of the `detector`.

Classes
-------

`Loader()`
:   A class to parse and save command line arguments.
    
    Attributes:
        model_path (str): Path to a model file.
        inp_path (str): Path to an input.
        input (MatLike | VideoCapture): Input.
        parser (ArgumentParser): Command line parser.
        isVideo (bool): Input type flag.
        img_size (int): Size of an input image.
    
    Initializes the Loader object.
    
    Args:
       None
    
    Returns:
        None

    ### Methods

    `parse(self)`
    :   Parses and saves command line arguments.
        
        Args:
            None
        
        Returns:
            None