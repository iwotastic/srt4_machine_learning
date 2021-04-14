# SRT IV Machine Learning
The machine learning pre-processing and training code for my SRT IV project.

## Repo layout
This repository is structured a bit differently than my other `srt4_...` repos, it is organized into folders that perform different stages of the machine learning process and different models. The purposes of the folders are as follows:
* [`download_and_prepare`](/download_and_prepare): Downloads the bot data from the PostgreSQL database used in [iwotastic/srt_data_collector](https://github.com/iwotastic/srt_data_collector) and puts it into JSON files.
* [`preprocess`](/preprocess): Processes the data from `download_and_prepare` to be used for training the ML models.
* [`train`](/train): This foldercontains files for each ML model that must be run seprately.
  * The models currently are hard-coded to expect the data storage device to be at `/Volumes/SRT4Data`.
  * ⚠️ **Warning:** Keyboard and browser data are relatively kind to low-powered computers. Mouse data is not, it uses over 300% CPU (on my MacBook Pro), over 16GB of RAM, and obscene amoust of disk read. In hind sight, I should have run this on a better computer, maybe an M1 Mac with Metal acceleration or better yet, a gaming PC with CUDA graphics.
* [`evaluate`](/evaluate): Uses data downloaded using `download_and_prepare` to evaluate a specfic "round" of data collection.
* [`model`](/model): This folder contains the models in TensorFlow 2 format, some notes:
  * The models with no `_v#` at the ends of their names are the originals
  * The ones with `_v2` at the end are the second version, but due to an error I made when training them the definitions of bot and human are inverted, this will need to be kept in mind if using this model.

## Requirements
* *Exactly* Python 3.8
* The packages in `requirements.txt` obtained using the below instructions

## Venv Instructions
```bash
# Initialize the venv (replace python with the name of your system's Python 3.8 binary)
python -m venv venv

# Activate the venv
source venv/bin/activate

# Install requirements with pip
pip install -r ./requirements.txt
```