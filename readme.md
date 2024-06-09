# UP2ME: <u>U</u>nivariate <u>P</u>re-training to <u>M</u>ultivariate Fin<u>e</u>-tuning as a General-purpose Framework for Multivariate Time Series Analysis (ICML 2024)

This is the origin Pytorch implementation of “[UP2ME: Univariate Pre-training to Multivariate Fine-tuning as a General-purpose Framework for Multivariate Time Series Analysis (ICML 2024)](https://openreview.net/pdf?id=aR3uxWlZhX)”

## Workflow
UP2ME is a general-purpose framework for Multivariate Time Series Analysis. It conducts taskagnostic pre-training when downstream tasks are unspecified. Once the task and setting (e.g. forecasting length) are determined, it gives sensible solutions with frozen pre-trained parameters. Further accuracy is achieved through multivariate fine-tuning.

<p align="center">
<img src=".\pic\workflow.png" width = "550" align=center />
</p>

## Keypoints

<p align="center">
<img src=".\pic\framework.png" width = "650" alt="" align=center />

Overview of UP2ME framework. <b>Left: Univariate Pre-training.</b> Univariate instances are generated using variable window length and channel decoupling. Generated instances are fed into the encoder and decoder for Masked AutoEnncoder (MAE) pre-training. Formulating downstream tasks as specific mask-reconstruction problems, UP2ME can give sensible solutions without parameter modification (right part without TC layers). <b>Right: Multivariate Fine-tuning</b> (forecasting in this example).  The pre-trained frozen encoder encodes a multivariate series into latent tokens.
The tokens are used to construct a dependency graph among channels. Learnable Temporal-Channel (TC) layers which take constructed graph as input, are inserted before the frozen decoder for fine-tuning.
</p>

### Univariate Pretraining

1. **Variable Window Length**: To meet the uncertain requirements for window length, for each pre-training step, we randomly sample a window length $L$ then generate a batch of instances with this length.

2. **Channel Decoupling**: To generate an instance of length $L$, instead of extracting a multivariate sub-series, we independently sample a time span and a channel index to generate a univariate sub-series.

3. **Immediate Reaction Mode**: After pre-training, UP2ME can perform
immediate forecasting, anomaly detection and imputation with frozen parameters by formulating them into specific mask-reconstruction problems:

    3.1. **Forecasting**: Past series are viewed as unmasked patches and future series are viewed as masked patches to reconstruct.
    
    3.2.  **Imputation**: Fully-observed patches are viewed as unmasked patches and patches containing at least one missing point are viewed as masked.

    3.3 **Anomaly Detection**: Iteratively mask each patch and use other unmasked patches to reconstruct it. Difference between reconstructed series and original series is used as the anomaly score.

### Multivariate Fine-tuning

1. **Sparse Dependency Graph Construction**: A sparse dependency graph is constructed using representations output by the pre-trained encoder to guide cross-channel dependency capturing.

2. **Temporal-Channel (TC) layer**: We freeze parameters of the pre-trained encoder and decoder while inserting learnable Temporal-Channel (TC) layers between them to capture cross-channel dependency and adjust temporal dependency. With few inductive biases, our TC layer contains a standard Transformer layer and a standard
Graph Transformer layer. The Graph Transformer layer takes the constructed dependency graph as input.

## Reproducibility
1. Install requirements by:

    ```
    pip install -r requirements.txt
    ```


2. Download the datasets from [UP2ME-datasets](https://drive.google.com/file/d/1oLYcQa7NJcMDSP_rYSkP5hQHzXL2rpZM/view?usp=drive_link) and unzip it into the folder `datasets` in the root folder. The struture should be like:

    ```
    datasets
    ├── ETT
    │   └── ETTm1.csv
    ├── same for csv format datasets: weather, ECL(Electricity) and traffic
    ├── SMD
    │   ├── SMD_train.npy
    │   ├── SMD_test.npy
    │   └── SMD_test_label.npy
    └── same for npy format datasets: PSM, SWaT and NIPS_Water(GECCO)
    ```



3. We have already put the pre-trained model for each dataset in `./pretrain-library`. To get forecasting results on ETTm1, run:
    ```
    bash scripts/forecast_scripts/ETTm1.sh
    ```
    the immediate reaction(UP2ME(IR)) and fine-tuning(UP2ME(FT)) modes  will be tested on 4 different forecasting lengths (96, 192, 336, 720) and results will be saved in a new folder `./forecast_results`.

4. To reproduce results for all 3 tasks on all 8 datasets, run other scripts in `./scripts`.  

## Citation
If you find this repository useful in your research, please cite:
```
@inproceedings{zhang2024upme,
title={{UP}2{ME}:  Univariate Pre-training to Multivariate Fine-tuning as a General-purpose Framework for Multivariate Time Series Analysis},
author={Yunhao Zhang and Minghao Liu and Shengyang Zhou and Junchi Yan},
booktitle={International Conference on Machine Learning (ICML)},
year={2024}
}
```

## Acknowledgement
We appreciate the following works for their valuable code and data:

https://github.com/thuml/Time-Series-Library

https://github.com/DAMO-DI-ML/KDD2023-DCdetector

https://github.com/zezhishao/STEP

https://github.com/facebookresearch/mae
