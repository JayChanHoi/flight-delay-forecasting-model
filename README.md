# flight-delay-forecasting-model
This is the repository of flight delay forecasting model for HK Express. The purpose of this project is to 
demonstrating coding test by implementing a data science solution for modeling a task of sparse feature data.

## Solution
### Train
![Screenshot](https://github.com/JayChanHoi/flight-delay-forecasting-model/blob/main/doc/flight-forecasting-model-train.png)
### Inference
![Screenshot](https://github.com/JayChanHoi/flight-delay-forecasting-model/blob/main/doc/flight_delay_forecasting.png)

The solution here is constructed with metric learning. The feature transformation network is the only module that needed to train.
Here we train it with unsupervised algorithms with autoencoder and random projection layer.

For further exploration, the core feature transformation network can be consider to train by supervised or other metric learning algorithms, like those using 
constractive loss, triplet loss, arcface loss.

The raw feature are selected from the raw data columns and has been processed by discretization and feature scaling.

As this is only the demonstrating a coding test, the infer module is not implemented as it only
valid to use when it's ready for deployment.

## Installation
Please setup environment with python 3.7 and install the dependencies by the following command.
```
pip install -r requirement.txt
```

## Getting Started
### Specific GPU
```
CUDA_VISIBLE_DEVICES=0,1 python -m src.train --batch_size 128 --test_interval 20 --num_nearest_neighbors 3 --embedding_dim 128 --num_epochs 2000 --model_version 1_2_4 --gaussian_kernel_k 0.8
```
### All GPU
```
python -m src.train --batch_size 128 --test_interval 20 --num_nearest_neighbors 3 --embedding_dim 128 --num_epochs 2000 --model_version 1_2_4 --gaussian_kernel_k 0.8
```
using the same command above with no GPU device will automatically switch to CPU mode.

## License
Distributed under the MIT License. See LICENSE for more information