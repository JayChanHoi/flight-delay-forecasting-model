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

As this is only the demonstrating a coding test, the infer module is not implemented as it only
valid to use when it's ready for deployment.

## Features
The raw feature are selected from the raw data columns and has been processed by discretization and feature scaling.

- aircraft name -> discretization
- aircraft type -> discretization
- route -> discretization
- weekday of flight -> discretization
- schedule departure time -> feature scaling
- schedule arrival time -> feature scaling
- pax adult -> feature scaling
- pax inf -> feature scaling
- seating capacity -> feature scaling
- baggage weight -> feature sacling

The features mentioned above will concatenate to form a feature vector with dimension of 122 as the input for feature 
transformation network.

## Installation
Please setup environment with python 3.7 and install the dependencies by the following command. Here I use pytorch=1.5 as
the deep learning framework which can provides dynamic tensor computation and is pythonic compared to other frameworks.
```
pip install -r requirement.txt
```

## Getting Started
### Specific GPU
if want to use metric learning model 
```
CUDA_VISIBLE_DEVICES=0,1 python -m src.train --batch_size 128 --test_interval 20 --num_nearest_neighbors 3 --embedding_dim 128 --num_epochs 2000 --model_version 1_2_4 --gaussian_kernel_k 0.8 --metric_learning
```
if want to use supervised PredNetwork model
```
CUDA_VISIBLE_DEVICES=0,1 python -m src.train --batch_size 128 --test_interval 20 --embedding_dim 128 --num_epochs 2000 --model_version 1_2_4 --smoothing_factor 0.05
```

### All GPU
if want to use metric learning model 
```
python -m src.train --batch_size 128 --test_interval 20 --num_nearest_neighbors 3 --embedding_dim 128 --num_epochs 2000 --model_version 1_2_4 --gaussian_kernel_k 0.8 --metric_learning
```
if want to use supervised PredNetwork model
```
python -m src.train --batch_size 128 --test_interval 20 --embedding_dim 128 --num_epochs 2000 --model_version 1_2_4 --smoothing_factor 0.05
```

using the same command above with no GPU device will automatically switch to CPU mode.

## License
Distributed under the MIT License. See LICENSE for more information