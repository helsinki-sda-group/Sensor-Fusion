import argparse
import os
import torch
import logging
from path import Path
from utils import custom_transform
from dataset.KITTI_dataset import KITTI
from model import *
from collections import defaultdict
from utils.kitti_eval import Vanilla_KITTI_tester
import numpy as np
import math
from pt_distr_env import DistributedEnviron

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_type', type=str, default='DeepCrossVIO', help='[DeepCrossVIO, CrossVIO, BNF]')
parser.add_argument('--data_dir', type=str, default='/nfs/turbo/coe-hunseok/mingyuy/KITTI_odometry', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')

parser.add_argument('--train_seq', type=list, default=['00', '01', '02', '04', '06', '08', '09'], help='sequences for training')
parser.add_argument('--val_seq', type=list, default=['05', '07', '10'], help='sequences for validation')
parser.add_argument('--seed', type=int, default=0, help='random seed') 

parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--img_h', type=int, default=256, help='image height')
parser.add_argument('--v_f_len', type=int, default=512, help='visual feature length') # Originally 512
parser.add_argument('--i_f_len', type=int, default=128, help='imu feature length') # Originally 128
parser.add_argument('--imu_encoder', type=str, default='CNN', help='[CNN, InertialTransformer]')
parser.add_argument('--fuse_method', type=str, default='cat', help='fusion method [cat, soft, hard]')
parser.add_argument('--cross_fusion', default=False, action='store_true', help='use cross fusion') 
parser.add_argument('--num_crossfusion_layers', type=int, default=1, help='what it says')
parser.add_argument('--imu_dropout', type=float, default=0, help='dropout for the IMU encoder')


parser.add_argument('--rnn_hidden_size', type=int, default=1024, help='size of the LSTM latent')
parser.add_argument('--rnn_dropout_out', type=float, default=0.2, help='dropout for the LSTM output layer')
parser.add_argument('--rnn_dropout_between', type=float, default=0.2, help='dropout within LSTM')

parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for the optimizer')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--seq_len', type=int, default=11, help='sequence length for LSTM')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--epochs_warmup', type=int, default=40, help='number of epochs for warmup')
parser.add_argument('--epochs_joint', type=int, default=40, help='number of epochs for joint training')
parser.add_argument('--epochs_fine', type=int, default=20, help='number of epochs for finetuning')
parser.add_argument('--lr_warmup', type=float, default=5e-4, help='learning rate for warming up stage')
parser.add_argument('--lr_joint', type=float, default=5e-5, help='learning rate for joint training stage')
parser.add_argument('--lr_fine', type=float, default=1e-6, help='learning rate for finetuning stage')
parser.add_argument('--eta', type=float, default=0.05, help='exponential decay factor for temperature')
parser.add_argument('--temp_init', type=float, default=5, help='initial temperature for gumbel-softmax')
parser.add_argument('--Lambda', type=float, default=3e-5, help='penalty factor for the visual encoder usage')

parser.add_argument('--experiment_name', type=str, default='experiment', help='experiment name')
parser.add_argument('--optimizer', type=str, default='AdamW', help='type of optimizer [AdamW, SGD, RAdam]')
parser.add_argument('--scheduler', type=str, default='default', help='type of optimizer [CyclicalLR, LambdaLR, default]')

parser.add_argument('--pretrain_flownet',type=str, default='./pretrain_models/flownets_bn_EPE2.459.pth.tar', help='whether to use the pre-trained flownet')
parser.add_argument('--pretrain', type=str, default=None, help='path to the pretrained model')
parser.add_argument('--hflip', default=False, action='store_true', help='whether to use horizonal flipping as augmentation')
parser.add_argument('--color', default=False, action='store_true', help='whether to use color augmentations')
parser.add_argument('--imu_noise', default=False, action='store_true', help='whether to use add noise to IMU')


parser.add_argument('--print_frequency', type=int, default=10, help='print frequency for loss values')
parser.add_argument('--weighted', default=False, action='store_true', help='whether to use weighted sum')

# GMflow
parser.add_argument('--gmflow_weights', type=str, default='./pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth', help='gmflow weights path.')
parser.add_argument('--imu_channels', type=int, default=6, help='IMU input dim.')
parser.add_argument('--num_scales', default=1, type=int,
                    help='basic gmflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
parser.add_argument('--with_shift', default=False, action='store_true', help='with shift attribute gmflow')
parser.add_argument('--feature_channels', default=128, type=int)
parser.add_argument('--upsample_factor', default=8, type=int)
parser.add_argument('--num_transformer_layers', default=6, type=int)
parser.add_argument('--num_head', default=1, type=int)
parser.add_argument('--attention_type', default='swin', type=str)
parser.add_argument('--ffn_dim_expansion', default=4, type=int)
parser.add_argument('--reg_refine', default=False, action='store_true', help='with regrefine')
parser.add_argument('--num_reg_refine', default=6, type=int)

parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                    help='number of splits in attention')
parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                    help='correlation radius for matching, -1 indicates global matching')
parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                    help='self-attention radius for flow propagation, -1 indicates global attention')

# Bottleneck fusion (only)
parser.add_argument('--num_layers', default=8, type=int)
parser.add_argument('--num_bottleneck_tokens', default=6, type=int)
parser.add_argument('--num_bottleneck_layers', default=8, type=int)
parser.add_argument('--nhead', default=2, type=int)

# Transformer Layer dropout
parser.add_argument('--transformer_dropout', type=float, default=0.15, help='dropout for the Transformer layer')


args = parser.parse_args()

# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def main():

    # Create Dir
    experiment_dir = Path('./evaluation')
    experiment_dir.mkdir_p()
    file_dir = experiment_dir.joinpath('{}/'.format(args.experiment_name))
    file_dir.mkdir_p()
    result_dir = file_dir.joinpath('files/')
    result_dir.mkdir_p()
    
    # GPU selections
    distr_env = DistributedEnviron()
    dist.init_process_group(backend="nccl", init_method='env://')
    torch.manual_seed(args.seed) # Set identical seed for all nodes
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = distr_env.local_rank
    device_count = torch.cuda.device_count()

    gpu_ids = [range(device_count)]

    if len(gpu_ids) > 0:
        torch.cuda.set_device(device)  #gpu_ids[0])
    
    # Initialize the tester
    tester = Vanilla_KITTI_tester(args)

    # Model initialization
    if args.model_type == 'DeepCrossVIO':
        model = DeepCrossVIO(args)
    elif args.model_type == 'CrossAttentionVIO':
        model = CrossAttentionVIO(args)
    elif args.model_type == 'CrossVIO':
        model = CrossVIO(args)
    elif args.model_type == 'BottleneckFusion':
        model = TransformerFusion(args)

    model.load_state_dict(torch.load(args.pretrain))
    print('load model %s'%args.pretrain)

    # Feed model to GPU
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=False) #Trying False from now...
    model.eval()

    errors = tester.eval(model, num_gpu=len(gpu_ids))
    tester.generate_plots(result_dir, 30)
    tester.save_text(result_dir)
    
    for i, seq in enumerate(args.val_seq):
        message = f"Seq: {seq}, t_rel: {tester.errors[i]['t_rel']:.4f}, r_rel: {tester.errors[i]['r_rel']:.4f}, "
        message += f"t_rmse: {tester.errors[i]['t_rmse']:.4f}, r_rmse: {tester.errors[i]['r_rmse']:.4f}, "
        message += f"usage: {tester.errors[i]['usage']:.4f}"
        print(message)
    
    

if __name__ == "__main__":
    main()




