# **Wav2Lip**: *Accurately Lip-syncing Videos In The Wild* 
### A commercial version of Wav2Lip can be directly accessed at https://sync.so
Are you looking to integrate this into a product? We have a turn-key hosted API with new and improved lip-syncing models here: https://sync.so/
For any other commercial / enterprise requests, please contact us at pavan@sync.so and prady@sync.so
To reach out to the authors directly you can reach us at prajwal@sync.so, rudrabha@sync.so.
This code is part of the paper: _A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild_ published at ACM Multimedia 2020. 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-lip-sync-expert-is-all-you-need-for-speech/lip-sync-on-lrs2)](https://paperswithcode.com/sota/lip-sync-on-lrs2?p=a-lip-sync-expert-is-all-you-need-for-speech)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-lip-sync-expert-is-all-you-need-for-speech/lip-sync-on-lrs3)](https://paperswithcode.com/sota/lip-sync-on-lrs3?p=a-lip-sync-expert-is-all-you-need-for-speech)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-lip-sync-expert-is-all-you-need-for-speech/lip-sync-on-lrw)](https://paperswithcode.com/sota/lip-sync-on-lrw?p=a-lip-sync-expert-is-all-you-need-for-speech)
|📑 Original Paper|📰 Project Page|🌀 Demo|⚡ Live Testing|📔 Colab Notebook
|:-:|:-:|:-:|:-:|:-:|
[Paper](http://arxiv.org/abs/2008.10010) | [Project Page](http://cvit.iiit.ac.in/research/projects/cvit-projects/a-lip-sync-expert-is-all-you-need-for-speech-to-lip-generation-in-the-wild/) | [Demo Video](https://youtu.be/0fXaDCZNOJc) | [Interactive Demo](https://synclabs.so/) | [Colab Notebook](https://colab.research.google.com/drive/1tZpDWXz49W6wDcTprANRGLo2D_EbD5J8?usp=sharing) /[Updated Collab Notebook](https://colab.research.google.com/drive/1IjFW1cLevs6Ouyu4Yht4mnR4yeuMqO7Y#scrollTo=MH1m608OymLH)
 
![Logo](https://drive.google.com/uc?export=view&id=1Wn0hPmpo4GRbCIJR8Tf20Akzdi1qjjG9)
----------
**Highlights**
----------
 - Weights of the visual quality disc has been updated in readme!
 - Lip-sync videos to any target speech with high accuracy :100:. Try our [interactive demo](https://sync.so/).
 - :sparkles: Works for any identity, voice, and language. Also works for CGI faces and synthetic voices.
 - Complete training code, inference code, and pretrained models are available :boom:
 - Or, quick-start with the Google Colab Notebook: [Link](https://colab.research.google.com/drive/1tZpDWXz49W6wDcTprANRGLo2D_EbD5J8?usp=sharing). Checkpoints and samples are available in a Google Drive [folder](https://drive.google.com/drive/folders/1I-0dNLfFOSFwrfqjNa-SXuwaURHE5K4k?usp=sharing) as well. There is also a [tutorial video](https://www.youtube.com/watch?v=Ic0TBhfuOrA) on this, courtesy of [What Make Art](https://www.youtube.com/channel/UCmGXH-jy0o2CuhqtpxbaQgA). Also, thanks to [Eyal Gruss](https://eyalgruss.com), there is a more accessible [Google Colab notebook](https://j.mp/wav2lip) with more useful features. A tutorial collab notebook is present at this [link](https://colab.research.google.com/drive/1IjFW1cLevs6Ouyu4Yht4mnR4yeuMqO7Y#scrollTo=MH1m608OymLH).  
 - :fire: :fire: Several new, reliable evaluation benchmarks and metrics [[`evaluation/` folder of this repo]](https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation) released. Instructions to calculate the metrics reported in the paper are also present.
--------
**Disclaimer**
--------
All results from this open-source code or our [demo website](https://bhaasha.iiit.ac.in/lipsync) should only be used for research/academic/personal purposes only. As the models are trained on the <a href="http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html">LRS2 dataset</a>, any form of commercial use is strictly prohibited. For commercial requests please contact us directly!

# Mouth Sync

A Python implementation of lip-sync technology based on the Wav2Lip architecture. This project provides tools for synchronizing lip movements in videos with audio input.

## Features

- Lip-sync videos to any target speech with high accuracy
- Works with any identity, voice, and language
- Supports both real and CGI faces
- Complete training and inference code included
- Pre-trained models available

## Quick Start

### Prerequisites

- Python 3.6+
- ffmpeg: `sudo apt-get install ffmpeg`
- Required Python packages: `pip install -r requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/melisay/mouth-sync.git
cd mouth-sync
```

2. Download the face detection pre-trained model:
```bash
# Download to face_detection/detection/sfd/s3fd.pth
wget https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth -O face_detection/detection/sfd/s3fd.pth
```

3. Download the pre-trained Wav2Lip model:
```bash
# Download from Google Drive and place in checkpoints/
# Link: https://drive.google.com/drive/folders/153HLrqlBNxzZcHi17PEvP09kkAfzRshM
```

### Usage

#### Inference

To lip-sync a video with audio:

```bash
python inference.py --checkpoint_path checkpoints/wav2lip.pth --face <video.mp4> --audio <audio.wav>
```

The result will be saved in `results/result_voice.mp4`.

#### Tips for Better Results

- Adjust face detection padding with `--pads` (e.g., `--pads 0 20 0 0`)
- Use `--nosmooth` if you see mouth dislocation artifacts
- Try different `--resize_factor` values for better results
- For 720p videos, you might get better results than 1080p

### Training

1. Prepare your dataset following the LRS2 structure
2. Preprocess the dataset:
```bash
python preprocess.py --data_root data_root/main --preprocessed_root lrs2_preprocessed/
```

3. Train the expert discriminator:
```bash
python color_syncnet_train.py --data_root lrs2_preprocessed/ --checkpoint_dir checkpoints/
```

4. Train the Wav2Lip model:
```bash
python wav2lip_train.py --data_root lrs2_preprocessed/ --checkpoint_dir checkpoints/ --syncnet_checkpoint_path checkpoints/syncnet.pth
```

## License

This project is for research and personal use only. Commercial use requires explicit permission from the authors.

## Citation

If you use this code, please cite the original Wav2Lip paper:

```bibtex
@inproceedings{10.1145/3394171.3413532,
author = {Prajwal, K R and Mukhopadhyay, Rudrabha and Namboodiri, Vinay P. and Jawahar, C.V.},
title = {A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild},
year = {2020},
booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
pages = {484–492},
doi = {10.1145/3394171.3413532}
}
```

## Acknowledgments

- Original Wav2Lip implementation by [Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
- Face detection code from [face_alignment](https://github.com/1adrianb/face-alignment)
- Code structure inspired by [deepvoice3_pytorch](https://github.com/r9y9/deepvoice3_pytorch)
