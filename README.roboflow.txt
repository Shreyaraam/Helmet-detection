
Helmet Detection - v3 2026-02-17 1:12pm
==============================

This dataset was exported via roboflow.com on February 17, 2026 at 7:42 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 1598 images.
Helmet are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Random shear of between -10° to +10° horizontally and -12° to +12° vertically
* Random brigthness adjustment of between 0 and +15 percent
* Random Gaussian blur of between 0 and 0.1 pixels
* Salt and pepper noise was applied to 0.1 percent of pixels


