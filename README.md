# SRT IV Machine Learning
The machine learning pre-processing and training code for my SRT IV project.

## Repo layout
This repository is structured a bit differently than my other `srt4_...` repos, it is organized into folders that perform different stages of the machine learning process and different models. The purposes of the folders are as follows:
* [`download_and_prepare`](/download_and_prepare) 📦: downloads the bot data from the PostgreSQL database used in [iwotastic/srt_data_collector](https://github.com/iwotastic/srt_data_collector) and puts it into JSON files.

## Requirements
* Python 3.9 or later
* For all the folders above marked with a 📦, install requirements from the requirements.txt at the root of this repo using the [Venv Instructions](#venv-instructions).
* For all the folders above marked with a 🐍, install and activate a conda environment, it'll have what you need.

## Venv Instructions
```bash
# Initialize the venv
python -m venv venv

# Activate the venv
source venv/bin/activate

# Install requirements with pip
pip install -r ./requirements.txt
```