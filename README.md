# [ACCV 2024] ELLAR: An Action Recognition Dataset for Extremely Low-Light Conditions with Dual Gamma Adaptive Modulation

By [Minse Ha](https://github.com/haminse/)\*, [Wan-Gi Bae](https://github.com/wangiid_TBA)\*, [Geunyoung Bae](https://github.com/flora101), and [Jong Taek Lee](https://scholar.google.com/citations?hl=en&user=NZ55Q-AAAAAJ).


This repo is the official implementation of ["ELLAR: An Action Recognition Dataset for Extremely Low-Light Conditions with Dual Gamma Adaptive Modulation"](paper_link_TBA). It is based on [mmaction2](https://github.com/open-mmlab/mmaction2) and [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer).



## Updates
***10/03/2024*** Initial commits | Project page is now available at [here](https://sites.google.com/view/knu-ellar/).

## About
In this research, we address the challenging problem of action recognition in extremely low-light environments. Currently, available datasets built under low-light settings are not truly representative of  extremely dark conditions because they have a sufficient signal-to-noise ratio, making them visible with simple low-light image enhancement methods. Due to the lack of datasets captured under extremely low-light conditions, we present a new dataset with more than 12K video samples, named Extremely Low-Light condition Action Recognition (ELLAR). This dataset is constructed to reflect the characteristics of extremely low-light conditions where the visibility of videos is corrupted by overwhelming noise and blurs. ELLAR also covers a diverse range of dark settings within the scope of extremely low-light conditions. Furthermore, we propose a simple yet strong baseline method, leveraging a Mixture of Experts in gamma intensity correction, which enables models to be flexible and adaptive to a range of low illuminance levels. Our approach significantly surpasses state-of-the-art results by 3.39% top-1 accuracy on ELLAR dataset. 


## ELLAR Dataset

You can download the ELLAR dataset from [here](http://gofile.me/7cPY4/TVPps3XL8), both videos and annotation files.

<img width="1151" alt="fig2" src="https://github.com/user-attachments/assets/cb30b116-e344-4df4-9e10-9d944aa9c5e1">


## Model and experimental results

The core idea of **DGAM(Dual Gamma Adaptive Modulation)** is its dual Mixture of Experts structure. This structure first identifies the characteristics of each sample and performs adaptive image enhancement that is optimal for action recognition. To this end, we designed a gating network named Adaptive Gamma Correction (AGC) that selects and applies the optimal GIC based on the illuminance information of the samples. Moreover, the matching classification head called Adaptive Head Selection (AHS) is selected to recognize features specific to the enhanced inputs. This dual mixture of expert systems allows the action recognition model to dynamically respond to inputs from diverse dark settings. 

<img width="1087" alt="main" src="https://github.com/user-attachments/assets/12c0a843-138f-49b5-9526-40ef4814e04b">

### Results

Comparison on ELLAR Dataset. Our method is pretrained by Kinetics400, and finetuned by ELLAR dataset. You can download the checkpoint pth file in [here](http://gofile.me/7cPY4/yu2u18Etb). 

The config file format is following mmaction2. The config file for DGAM is already located in `./configs/recognition/swin/hydra_config.py`.    


| Model          | Pretrained | Input Size      | Top-1  | Top-5  |
|----------------|------------|-----------------|--------|--------|
| ResNet101      | K700       | 3×16×112²       | 10.46  | 45.69  |
| ResNeXt101     | K400       | 3×16×112²       | 9.63   | 39.37  |
| DarkLight      | IG-65M     | 3×64×112²       | 28.58  | 64.31  |
| TimeSformer    | K400       | 3×96×224²       | 15.51  | 55.96  |
| Video-Swin-B   | K400       | 3×32×224²       | 35.03  | 68.87  |
| **DGAM (Ours)**| K400       | 3×32×224²       | **38.42** | **74.44** |


## Usage

####  Installation

Please refer to [mmaction2](https://github.com/open-mmlab/mmaction2) and [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) setup for installation.


####  Data Preparation

Expected Data Directory Structure:
```
.data/
|-- Other dataset/
|-- Another dataset/
|-- ELLAR/
|------- ELLAR_label_train.txt
|------- ELLAR_label_val.txt
|------- ELLAR_label_test.txt
|------- videos/
|------------- Walking/
|------------- Running/
|------------- Stertching/
|------------- ...
                
```


### Inference

```
python tools/test.py configs/recognition/swin/hydra_config.py work_dirs/hydra_den/DGAM_ELLAR.pth --eval top_k_accuracy
```

### Training

```
python tools/train.py configs/recognition/swin/hydra_config.py --cfg-options load_from=workdirs/hydra_den/DGAM_ELLAR.pth model.backbone.use_checkpoint=True --validate
```
