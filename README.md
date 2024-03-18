# Vehicle-Detection

In video monitoring systems for traffic situation - on streets, in yards, in parking lots - it is often necessary to determine the type of vehicle.
This is a __Python3__ application that uses a convolutional neural network at its core to detect 9 different classes of vehicles. The software module is capable of processing both single images and video streams.

## Files description

1. `src/main.py`: code of the main module managing other modules of the app; Program entry point;
2. `src/loader.py`: code of the loader module;
3. `src/detector.py`: detection implementation code. The module processing video stream and a picture.

## Build

### Cloning

Clone the repository from github by running the command in the terminal:

```
git clone https://github.com/IlS0/Vehicle-Detection.git
```

### Requirements

Unzip the archive, navigate to the directory and install the requirements by running the following command:

```
pip install -r requirements.txt
```

## Run

To the run you have to go to the `/src` directory. Start the application by running the command in the terminal:

```
python main.py <path to model> <input size> (-v | -i) <path to input>
``` 

__IMPORTANT:__ your Python version must be greater than or equal to `3.10`. Also, the model that you pass to the application has to be in `.onnx` format only!

## Inference results

![Inference](https://raw.githubusercontent.com/IlS0/Vehicle-Detection/main/images/infrenece.gif)

## Metrics

![Train-Valid](https://raw.githubusercontent.com/IlS0/Vehicle-Detection/main/images/results.png)

### PR-curve

![PR-curve](https://raw.githubusercontent.com/IlS0/Vehicle-Detection/main/images/PR_curve.png)

### Recall

![Recall](https://raw.githubusercontent.com/IlS0/Vehicle-Detection/main/images/R_curve.png)

### Confusion matrix

![Confusion matrix](https://raw.githubusercontent.com/IlS0/Vehicle-Detection/main/images/confusion_matrix.png)


## Documentation

You can find the documentation for all of the application's basic modules [here](https://github.com/IlS0/Vehicle-Detection/tree/main/docs).

## License

The app is released under the [Apache 2.0 license](https://github.com/IlS0/Vehicle-Detection/blob/main/LICENSE).
