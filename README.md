# HashNeRF Accelerating NeRF Training with Multiresolution Hash Encoding
This is the final project for COMS W4732 Computer Vision II. Report is available [here](https://github.com/NingHsia/HashNeRF-Accelerating-NeRF-Training-with-Multiresolution-Hash-Encoding/blob/main/report.pdf).

This project is built on top of [hashNeRF-pytorch](https://github.com/NingHsia/HashNeRF-Accelerating-NeRF-Training-with-Multiresolution-Hash-Encoding) implementation.

## Getting Started
### Requirements
Install all required packages:
```
pip install -r requirements.txt
```
### Data
Download data from here: [Google Drive](https://drive.google.com/file/d/1jdqTZigCFbPz0-r2FVRDVDHioGTtQRcA/view?usp=share_link).
Unzip it, and put folder data into the repo folder.

### Training HashNeRF
### HashNeRF
To train a `fern` HashNeRF model:
```
python run_nerf.py --config configs/fern.txt
```
To train for other objects, replace `configs/fern.txt` with `configs/{object}.txt`. You can choose from fern, room, orchids, and leaves.
### Variation of HashneRF
To train a `fern` HashNeRF-3RGB model mentioned in the report experiment section:
```
python run_nerf.py --config configs/fern.txt --variation 3RGB
```
To train a `fern` HashNeRF-DRGB model mentioned in the report experiment section:
```
python run_nerf.py --config configs/fern.txt --variation DRGB
```
