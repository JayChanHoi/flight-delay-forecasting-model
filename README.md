# flight-delay-forecasting-model
This is the repository of flight delay forecasting model for HK Express. The purpose of this project is to 
demonstrating coding test by implementing a data science solution for modeling a task of sparse feature data.

## Repository Structure
```
.
├── LICENSE
├── README.md
├── data
│   ├── DelayReason.csv
│   └── FlightSchedule.csv
├── doc
│   ├── Data\ Dictionary.xlsx
│   ├── Presentation\ Questions.pdf
│   ├── flight-delay-forecasting-prednet.png
│   ├── flight-forecasting-model-train.png
│   └── flight_delay_forecasting.png
├── requirement.txt
└── src
    ├── data_handler
    │   └── data_processor.py
    ├── model
    │   ├── embedding_network.py
    │   ├── focal_loss.py
    │   ├── knn.py
    │   ├── prediction_network.py
    │   └── self_smoothing_operator.py
    ├── train.py
    └── utils.py
```

## Solution Architecture
### Metric Learning Model
![Screenshot](https://github.com/JayChanHoi/flight-delay-forecasting-model/blob/main/doc/flight_delay_forecasting.png)

This solution here is constructed with metric learning. The feature transformation network is the only module that needed to train.
Here we train it with unsupervised algorithms with autoencoder and random projection layer.

For further exploration, the core feature transformation network can be consider to train by supervised or other metric learning algorithms, like those using 
constractive loss, triplet loss, arcface loss.

### End to End Prediction Model
![Screenshot](https://github.com/JayChanHoi/flight-delay-forecasting-model/blob/main/doc/flight-delay-forecasting-prednet.png)

This solution is constructed with multiple feed-forward neural network. Each of them serve different functionality.

 - Discrete feature FFN -> transform discrete feature 
 - continuous feature FFN -> transform continuous feature  
 - fusion feature FFN -> used to fuse refined discrete feature and continuous feature
 
Also, in order to reduce over-fitting and data set imbalance, label smoothing and focal loss are induced with this end to 
end model. 

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
CUDA_VISIBLE_DEVICES=0,1 python -m src.train --batch_size 256 --test_interval 20 --embedding_dim 32 --num_epochs 2000 --model_version 1_3_5 --weight_decay 0.00005 --dropout_p 0.2 --smoothing_factor 0.1
```

### All GPU
if want to use metric learning model 
```
python -m src.train --batch_size 128 --test_interval 20 --num_nearest_neighbors 3 --embedding_dim 128 --num_epochs 2000 --model_version 1_2_4 --gaussian_kernel_k 0.8 --metric_learning
```
if want to use supervised PredNetwork model
```
python -m src.train --batch_size 256 --test_interval 20 --embedding_dim 32 --num_epochs 2000 --model_version 1_3_5 --weight_decay 0.00005 --dropout_p 0.2 --smoothing_factor 0.1
```

using the same command above with no GPU device will automatically switch to CPU mode.

## License
Distributed under the MIT License. See LICENSE for more information