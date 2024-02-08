# Sensor-Fusion

This repository contains the code for the implementation of a novel cross-attention-based pedestrian visual-inertial odometry model as described in the paper titled ["A Novel Cross-Attention-Based Pedestrian Visual–Inertial Odometry With Analyses Demonstrating Challenges in Dense Optical Flow"](https://ieeexplore.ieee.org/abstract/document/10363184).

## Installation

Clone the repository:

```bash
git clone https://github.com/helsinki-sda-group/Sensor-Fusion.git
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Download KITTI
To download and format the KITTI dataset run:

```bash
  $cd data
  $source data_prep.sh
```

## Training and Testing

To train the model, run `train_CrossVIO.py` with appropriate arguments. An example provided:

```bash
python train_CrossVIO.py \
                        --model_type 'DeepCrossVIO' \
                        --data_dir './data/' \
                        --workers 8 \
                        --skip_frames 1 \
                        --save_dir './training_results' \
                        --pretrain_flownet './pretrain_models/flownets_bn_EPE2.459.pth.tar'\
                        --batch_size 16 \
                        --img_w 1241 \
                        --img_h 376 \
                        --v_f_len 512 \
                        --i_f_len 512 \
                        --rnn_hidden_size 512 \
                        --optimizer 'AdamW' \
                        --epochs_warmup 40 \
                        --epochs_joint 40 \
                        --epochs_fine 21 \
                        --lr_warmup 5e-4 \
                        --lr_joint 5e-5 \
                        --lr_fine 1e-6 \
                        --experiment_name 'DeepCrossVIO-STL' \
                        --print_frequency 20
```

To test the model, run `test_CrossVIO.py` with appropriate arguments. [Pretrained weights](https://drive.google.com/file/d/1K9si144jhC9fmmEIxuJQ-p3EkGs4h6r9/view?usp=drive_link) finetuned on KITTI are provided, remember to edit the path for the --pretrain argument.

```bash
python test_CrossVIO.py \
                        --model_type 'DeepCrossVIO' \
                        --data_dir './data/' \
                        --pretrain './results/IMU-queries-int-multiloss-2/checkpoints/best_5.50.pth' \
                        --batch_size 32 \
                        --v_f_len 512 \
                        --i_f_len 512 \
                        --rnn_hidden_size 512 \
                        --optimizer 'AdamW' \
                        --epochs_warmup 50 \
                        --epochs_joint 100 \
                        --epochs_fine 51 \
                        --lr_warmup 1e-3 \
                        --lr_joint 5e-5 \
                        --lr_fine 1e-6 \
                        --experiment_name 'BEST'
```

## Acknowledgements

To the authors of ["Efficient Deep Visual and Inertial Odometry with Adaptive Visual Modality Selection"](https://arxiv.org/pdf/2205.06187.pdf) for their work and publishing their code to the public.


