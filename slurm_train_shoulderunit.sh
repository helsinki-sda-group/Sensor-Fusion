#!/bin/bash
#SBATCH --job-name=Cross-Fusion
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --exclusive
#SBATCH --partition standard-g
#SBATCH --account=project_462000238
#SBATCH --gpus-per-node=8

module load LUMI/22.08 partition/G
module load singularity-bindings
module load aws-ofi-rccl

export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1
export SINGULARITYENV_LD_LIBRARY_PATH=/opt/ompi/lib:${EBROOTAWSMINOFIMINRCCL}/lib:/opt/cray/xpmem/2.4.4-2.3_9.1__gff0e1d9.shasta/lib64:${SINGULARITYENV_LD_LIBRARY_PATH}

srun singularity exec -B /scratch/project_462000238/sensor-fusion/Sensor-Fusion:/scratch/project_462000238/sensor-fusion/Sensor-Fusion \
                        pytorch_rocm5.4.1_ubuntu20.04_py3.7_pytorch_1.12.1.sif \
                        bash -c ". ~/pt_rocm5.4.1_env/bin/activate && python train_CrossVIO.py \
                        --model_type 'DeepCrossVIO' \
                        --data_dir './ADVIO/' \
                        --train_seq 01 02 03 04 05 06 07 09 10 11 13 14 16 17 18 19 21 22 23 \
                        --val_seq 20 15 12 08 \
                        --workers 1 \
                        --skip_frames 1 \
                        --save_dir './training_results' \
                        --pretrain_flownet './pretrain_models/flownets_bn_EPE2.459.pth.tar'\
                        --batch_size 1 \
                        --v_f_len 512 \
                        --i_f_len 512 \
                        --rnn_hidden_size 512 \
                        --imu_encoder 'InertialTransformer' \
                        --optimizer 'AdamW' \
                        --epochs_warmup 40 \
                        --epochs_joint 40 \
                        --epochs_fine 21 \
                        --lr_warmup 5e-4 \
                        --lr_joint 5e-5 \
                        --lr_fine 1e-6 \
                        --experiment_name 'ADVIO-test' \
                        --print_frequency 20"
