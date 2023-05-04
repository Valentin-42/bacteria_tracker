<div align="center">
  <h1>Bacteria Tracker</h1>
<!-- Badges -->
<p>
  <a href="https://github.com/Valentin-42/bacteria_tracker/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/Valentin-42/bacteria_tracker" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/Valentin-42/bacteria_tracker" alt="last update" />
  </a>
</p>
 
</div>

<br />

<!-- Table of Contents -->
# :notebook_with_decorative_cover: Table of Contents

- [About the Project](#star2-about-the-project)
- [Architecture](#📐-Architecture) 
- [Getting Started](#Getting-Started)
- [Warning for all contributors](#Warning-for-all-contributors)
- [Roadmap](#Roadmap)
- [Contact](#Contact)

  

<!-- About the Project -->
## :star2: About the Project


<!-- Screenshots -->
### :camera: Screenshots

<div align="center"> 
  <img src="https://github.com/Valentin-42/bacteria_tracker/blob/main/img50.jpg" alt="screenshot" />
</div>


<!-- TechStack -->
### 👋 Project objective and context

This project takes place in the context of a collaboration with the microbiologists with whom we are intending to develop a pipeline to detect and track bacteria attaching themselves on a glass substrate. In this experimental setup, a flow of water loaded with bio-luminescent bacteria is passing a glass substrate while being imaged with a microscope at 0.5Hz. Some bacteria attach to the surface, sometimes detaching after some time. The objective of the project is to build a detector and tracker to follow these events and provide statistics on the adherence properties of the bacteria in various conditions, as a function of their angles with respect to the flow in particular.

<!-- Features -->
### :dart: Current implemented features

- Detection
- Orientation
- Tracking
- Statistics

See slides with videos : [here](https://docs.google.com/presentation/d/1IzDbCYl1f1OtAy6J5RWzvNzrpVyTDHdpT4xsQvKNCBk/edit?usp=sharing)

<!-- Architecture -->
## 	📐  Architecture

- Dataset folder : Last created dataset ready to use for training yolo model.
- Labelbox : Codes to interact with the labelbox images and tools to convert annotation files. 
- yolov5 : Forked from official yolov5 Ultralytics repo. Below is the list of added codes :
	
   * 📁 ws/ : 
	   * color_filter.py :  
	   This script can be used to quickly and easily filter images based on their color content, which can be useful in a variety of computer vision applications such as object detection and tracking.
	   * DatasetCreator.py : 
This code defines a class called `DatasetCreator` which has several methods to create a custom dataset from a set of images, including methods to label the images, create the folder architecture of the dataset, clean up the dataset and a method to check a threshold for filtering images based on their color.
	   * images_to_video.py : Create a avi video from a given folder of images where names of images are set to imgX.jpg (X=1,2,...). 
	   * interpreter.py : 
	   The code defines two functions: 'statistics_from_csv' to extract statistics and plot a bar graph and 'create_illustration_video' to create a video illustration from the CSV data.
	   * kalman.py :
	   Computes tracking from a set of detections using a Kalman filter. Create the `Bacteria` class.
	   * normalize.py : Normalize pixels of an image (cool code but not really used for now).
   * mydetect.sh : Bash script to run detect.py with custom args. Compute detections from a set of images.
   * mytrain.sh : Bash script to run train.py with custom args.  Train a model given dataset path.


<!-- Getting Started -->
## 	🧰  Getting Started


<!-- Installation -->
### :gear: Installation

Clone the project 

```bash
   clone 
```
   

<!-- Run Locally -->
### 🖥️  Running pipeline from scratch


Go to the workspace directory
```bash
  cd bacteria_tracker/yolov5/ws/
```
Open the pipeline bash script
```bash
   vim run_pipeline.sh -> Change paths to yours
```
 Run 
```bash
   run run_pipeline.sh
```
<!-- Warning -->
## ⚠️ Warning for all contributors

- Make sure to change paths that are defined in each code. We tried to make all code reusable but some may require you to change hard coded paths, especially if not running on Linux. No environment variable is being used in any custom codes.  

<!-- Roadmap -->
## 🧭 Roadmap for future contributions

* [ ] Rework & clean statistics codes 
* [ ] Improve model with more annotated images from LabelBox

<!-- License -->
## :warning: License

Distributed under the no License. See LICENSE.txt for more information.


<!-- Contact -->
## :handshake: Contact

Dream Lab   - https://dream.georgiatech-metz.fr/ 

Georgia Tech - https://europe.gatech.edu/

Alvaro - https://github.com/hopett11
Valentin - https://github.com/Valentin-42

Prof. - https://github.com/cedricpradalier




