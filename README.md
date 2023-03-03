# Video Feature Extractor

## Description

This feature extracts video features via an action recognition model in [GluonCV](https://github.com/dmlc/gluon-cv), audio featues usins [panns_interfernce](https://github.com/qiuqiangkong/panns_inference), Mae features [VideoMAE](https://github.com/MCG-NJU/VideoMAE), ACR features using [Whisper](https://github.com/openai/whisper) for transcriptions and [Bert](https://github.com/google-research/bert) for tokenization, and CLIP features using using [CLIP](https://github.com/openai/CLIP). These features will be used to train an audio descriptive captioning moddel.

## Setup

The following step will require Conda installed. Run the following to create the Conda environment with all the dependencies:

    conda env create -f ENV.yml 

## How to use

Run the below command to run the extractor with the **videos** set to the directory path of of the videos and **output** to where you want the features stored

    python full_extraction.py --videos=/data/msr_vtt/train_val_videos --output=/data/msr_vtt_test/feat_ext
