[![Maintenance](https://img.shields.io/badge/Maintained%3F-No-red.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![Generic badge](https://img.shields.io/badge/python-3.7%20|%203.8-blue.svg)](https://shields.io/) [![Generic badge](https://img.shields.io/badge/version-v1.0-cc.svg)](https://shields.io/)

# P-NeRF

##### :speech_balloon: A report explaining the project can be found here: [report.pdf](report.pdf) :speech_balloon:

This is the project for the Computer Vision course at the University of Trento (Italy).
The objective of this project is to create a wrapper around pixelnerf, which was developed by Yu et al. and is available at this [link](https://github.com/sxyu/pixel-nerf).
In our specific case, we want to automate the training of a NeRF model (the pixelnerf variation which requires less views).

### Example of 3D scene reconstruction
Starting from 8 views of the same scene, the model reconstructs the scene:

![image](https://media.giphy.com/media/WrxRcc5mnexksBeHyd/giphy.gif)

## Set Up

**installing detectron2**

detectron2 is used in preprocess.py to remove the images background, the parameter --coco_class specifies which object type to segment (0-->person,  2-->car, (default) 56-->chair)

```sh
pip install -U torch torchvision
pip install git+https://github.com/facebookresearch/fvcore.git
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
```


**installing required python packages**

some python packages required by the code can by installed from `requirements.txt`, inside the repository.


```sh
git clone https://github.com/thomasreolon/P-NeRF.git
cd P-NeRF
pip install -r requirements.txt
```

## How To Run:

The script `run.py` inside the `src` folder coordinates three steps:
- preprocessing: which extract the frames from the video, removes the backgound and build a custom dataset
- training: load the custom step and train the NeRF model for a predefined number of epochs
- video generation: use the model trained in the previous step, plus some views, to generate the final video.

A simple demo for running the code: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oi_CgZw2H8RixG2UxxzqCEOOSbVRtfaH?usp=sharing)


### Example:

```sh
python src/run.py -n my_model -e 20000 -m chair --gen_video --preprocess
```

We create a new model, called `my_model` which load the pretrained `chair` model and is trained for `20000` epochs.  
`--preprocess` is a flag that tells the script to create a new dataset, while `--gen_video` is a flag that tells the script to create a video after the training.  
In order to train a new model from scratch one can omit `-m chair`.


## Project Structure

    EVOLUTIVE-NAS
    ├── src
    |    ├── run.py                [main script]
    |    ├── nerf                  [implementation of NeRF]
    |    |    ├── data               [dataset declaration]
    |    |    ├── model              [MLP model & encoder]
    |    |    ├── render             [IMPORTANT: use the rays to sample points, then ask to the MLP to predict the rgb/density; computes the fiinal rgb color]
    |    |    ├── util               [utility funcions]
    |    |    |    ├── args            [some settings of the ]
    |    |    |    ├── recon           [class that implements crossover between 2 genotypes]
    |    |    |    ├── util            [class that implements crossover between 2 genotypes]
    |    |
    |    ├── scripts                [steps of the algorithm]
    |    |    ├── detectron2           [model that does segmentation]
    |    |    ├── trainlib             [class that handles the training]
    |    |    ├── gen_video            [use a finetuned model to produce the video]
    |    |    ├── gray2rgb             [change images from grayscale to rgb]
    |    |    ├── preproc              [create the custom dataset from the videos inside /input]
    |    |    ├── train.py             [trains the model extending trainlib class]

## Output example
Example of 3D scene reconstruction:  

![output](https://media.giphy.com/media/GpLiNp6mzroNzz7oU1/giphy.gif)
