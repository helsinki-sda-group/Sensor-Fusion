import argparse
import os
import torch
import logging
from path import Path
from utils import custom_transform
from dataset.KITTI_dataset import KITTI
from dataset.ADVIO_dataset import ADVIO
from model import *
from collections import defaultdict
from utils.kitti_eval import Vanilla_KITTI_tester
from utils.pytorch_utils import seq_accu
import numpy as np
import math
from pt_distr_env import DistributedEnviron
import cv2

import torch
from torch.optim.lr_scheduler import CyclicLR, LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_type', type=str, default='DeepCrossVIO', help='[DeepCrossVIO, CrossVIO, BNF]')
parser.add_argument('--data_dir', type=str, default='/nfs/turbo/coe-hunseok/mingyuy/KITTI_odometry', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')

parser.add_argument('--train_seq', nargs='+', default=['00', '01', '02', '04', '06', '08', '09'], help='sequences for training')
parser.add_argument('--val_seq', nargs='+', default=['05', '07', '10'], help='sequences for validation')
parser.add_argument('--skip_frames', type=int, default=1, help='frames to skip to reduce fps')
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
parser.add_argument('--workers', type=int, default=8, help='number of workers')
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

def lr_function(epoch):
    # Initial learning rate and final learning rate for the warmup phase
    initial_lr = 1e-8
    warmup_lr = 1e-4
    finetune_lr = 1e-6

    # Number of epochs for the warmup and decay phases
    warmup_epochs = 5
    decay_epochs = 120

    # Linear warmup phase
    if epoch < warmup_epochs:
        lr = initial_lr + (warmup_lr - initial_lr) * (epoch / warmup_epochs)

    # Linear decay phase
    elif epoch >= warmup_epochs and epoch < warmup_epochs + decay_epochs:
        epoch_in_decay_phase = epoch - warmup_epochs
        lr = warmup_lr - (warmup_lr - initial_lr) * (epoch_in_decay_phase / decay_epochs)

    # Maintain the minimum learning rate after the decay phase
    else:
        lr = finetune_lr

    return lr

def update_status(ep, args, model, scheduler=None):
    if scheduler is None:
        #lr = lr_function(ep)
        if ep < args.epochs_warmup:  # Warmup stage
            lr = args.lr_warmup
        elif ep >= args.epochs_warmup and ep < args.epochs_warmup + args.epochs_joint: # Joint training stage
            lr = args.lr_joint
        elif ep >= args.epochs_warmup + args.epochs_joint: # Finetuning stage
            lr = args.lr_fine
    else:
        lr = scheduler.get_last_lr()[0]
    return lr

def compute_loss(poses, gts, weight, weighted):
    # In se3 representation the angle is in the last 3 values
    if not weighted:
        angle_loss = torch.nn.functional.mse_loss(poses[:,:,:3], gts[:, :, :3])
        translation_loss = torch.nn.functional.mse_loss(poses[:,:, 3:], gts[:, :, 3:])
    else:
        weight = weight/weight.sum()
        angle_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,:3] - gts[:, :, :3]) ** 2).mean()
        translation_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:, 3:] - gts[:, :, 3:]) ** 2).mean()
    return 1000 * angle_loss + translation_loss

def compute_seq_loss(poses, gts):
    poses = seq_accu(poses)
    gts = seq_accu(gts)
    cosine_distance = -torch.mean(F.cosine_similarity(poses.flatten(-2,-1), gts.flatten(-2,-1)))
    return cosine_distance

def train(model, optimizer, train_loader, logger, ep, rank, weighted=False):
    
    mse_losses = []
    penalties = []
    data_len = len(train_loader)

    for i, (imgs, imus, gts, rot, weight) in enumerate(train_loader):

        imgs = imgs.cuda().float()
        imus = imus.cuda().float()
        gts = gts.cuda().float()
        weight = weight.cuda().float()

        optimizer.zero_grad(set_to_none=True)
        
        # Save flow on beginning of every epoch
        if i == 0:
            save_flow = True

        out = model(imgs, imus, save_flow=save_flow)
        save_flow = False
        pose_loss = 0
        
        for j in range(len(out)-1): # last output is hidden state
            pose_loss += compute_loss(out[j], gts, weight, weighted)
        #pose_loss += compute_seq_loss(out[0], gts)

        pose_loss.backward()
        optimizer.step()
        
        if rank == 0 and i % args.print_frequency == 0:
            message = f'Epoch: {ep}, iters: {i}/{data_len}, pose loss: {pose_loss.item():.6f}'
            print(message)
            logger.info(message)

        mse_losses.append(pose_loss.item())

    return np.mean(mse_losses)


def main():

    # Create Dir
    experiment_dir = Path('./results')
    experiment_dir.mkdir_p()
    file_dir = experiment_dir.joinpath('{}/'.format(args.experiment_name))
    file_dir.mkdir_p()
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir_p()
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir_p()

    save_plots_path = file_dir.joinpath('plots/')
    save_plots_path.mkdir_p()
    
    # Create logs
    logger = logging.getLogger(args.experiment_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s.txt'%args.experiment_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    # GPU selections
    distr_env = DistributedEnviron()
    dist.init_process_group(backend="nccl", init_method='env://')
    torch.manual_seed(args.seed) # Set identical seed for all nodes
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = distr_env.local_rank
    device_count = torch.cuda.device_count()

    logger.info(f"World Size: {world_size}")
    logger.info(f"Local rank (device): {device}")
    logger.info(f"Rank : {rank}")
    logger.info(f"All available GPUs: {device_count}")
    logger.info(f"Device : {device}")

    gpu_ids = [range(device_count)]

    if len(gpu_ids) > 0:
        torch.cuda.set_device(device)  #gpu_ids[0])
    
    # Load the dataset
    transform_train = [custom_transform.ToTensor(),
                       custom_transform.Resize((args.img_h, args.img_w))]
    if args.hflip:
        transform_train += [custom_transform.RandomHorizontalFlip()]
    if args.color:
        transform_train += [custom_transform.RandomColorAug()]
    if args.imu_noise:
        transform_train += [custom_transform.IMUNoise()]
    transform_train = custom_transform.Compose(transform_train)
    print()
    train_dataset = ADVIO(args.data_dir, #KITTI
                    sequence_length=args.seq_len,
                    train_seqs=args.train_seq,
                    transform=transform_train,
                    skip_frames=args.skip_frames
                    )
    logger.info('train_dataset: ' + str(train_dataset))

    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        num_replicas=device_count,
        rank=rank,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, # per gpu
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        )
    logger.info('train_dataset: ' + str(train_dataset))

    # Initialize the tester
    tester = Vanilla_KITTI_tester(args)
    
    # Model selection
    if args.model_type == 'DeepCrossVIO':
        model = DeepCrossVIO(args)
        #Use the pre-trained flownet or not
        if args.pretrain_flownet and args.pretrain is None:
            pretrained_w = torch.load(args.pretrain_flownet, map_location='cpu')
            model_dict = model.Feature_net.state_dict()
            update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
            model_dict.update(update_dict)
            model.Feature_net.load_state_dict(model_dict)
            print('Using pretrained Flownet')
            logger.info('Using pretrained Flownet')
    elif args.model_type == 'CrossAttentionVIO':
        model = CrossAttentionVIO(args)
        #Use the pre-trained flownet or not
        if args.pretrain_flownet and args.pretrain is None:
            pretrained_w = torch.load(args.pretrain_flownet, map_location='cpu')
            model_dict = model.Feature_net.state_dict()
            update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
            model_dict.update(update_dict)
            model.Feature_net.load_state_dict(model_dict)
            print('Using pretrained Flownet')
            logger.info('Using pretrained Flownet')
    elif args.model_type == 'CrossVIO':
        model = CrossVIO(args)
    elif args.model_type == 'TransformerFusion':
        model = TransformerFusion(args)
    elif args.model_type == 'IterativeAttention':
        model = IterativeAttention(args)
        if args.pretrain_flownet and args.pretrain is None:
            pretrained_w = torch.load(args.pretrain_flownet, map_location='cpu')
            model_dict = model.vis_encoder.state_dict()
            update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
            model_dict.update(update_dict)
            model.vis_encoder.load_state_dict(model_dict)
            print('Using pretrained Flownet')
            logger.info('Using pretrained Flownet')
            # for param in model.vis_encoder.parameters():
            #     param.requires_grad = False
            # logger.info('Flownet parameters frozen')
                

    # Continual training or not
    if args.pretrain is not None:
        # Load the state_dict from the provided path
        state_dict = torch.load(args.pretrain)

        # Load the modified state_dict into the model
        model.load_state_dict(state_dict, strict=False)

        print('load model %s' % args.pretrain)
        logger.info('load model %s' % args.pretrain)

    else:
        print('Training from scratch')
        logger.info('Training from scratch')

    # Feed model to GPU
    model.to(device) #gpu_ids[0])
    find_unused_parameters = True if args.model_type == 'TransformerFusion' else False
    model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=find_unused_parameters)

    # Load pretrained weights for GMFlow (must be done after DistributedDataParallel)
    if args.model_type == 'TransformerFusion':
        map_location = {'cuda:0' : f'cuda:{rank}'}
        checkpoint = torch.load(args.gmflow_weights, map_location=map_location) # map_location=loc)
        model.module.visual_backbone.load_state_dict(checkpoint['model'], strict=False) # strict=True previously
        for param in model.module.visual_backbone.parameters():
            param.requires_grad = False
        model.module.visual_backbone.eval()
        print('Using pretrained GMFlow')
        logger.info('Using pretrained GMFlow')

    pretrain = args.pretrain 
    #init_epoch = int(pretrain[-7:-4])+1 if args.pretrain is not None else 0
    init_epoch = 0
    
    # Initialize the optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
    elif args.optimizer == 'RAdam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)

    # Initialize the scheduler.
    if args.scheduler == 'CyclicalLR':
        scheduler = CyclicLR(optimizer, base_lr=args.lr_fine, max_lr=args.lr_warmup, 
                     step_size_up=5, mode='triangular2', gamma=0.85, cycle_momentum=False)
    elif args.scheduler == 'default':
        scheduler = None
    
    best = 10000

    for ep in range(init_epoch, args.epochs_warmup+args.epochs_joint+args.epochs_fine):
        
        lr = update_status(ep, args, model, scheduler=scheduler)
        optimizer.param_groups[0]['lr'] = lr
        message = f'Epoch: {ep}, lr: {lr}'
        print(message)
        logger.info(message)

        model.train()
        if args.model_type == 'TransformerFusion':
            model.module.visual_backbone.eval()
        avg_pose_loss = train(model, optimizer, train_loader, logger, ep, rank=rank)
        if scheduler is not None:
            scheduler.step()
        
        # Save the model after training
        if rank == 0:
            torch.save(model.module.state_dict(), f'{checkpoints_dir}/{ep:003}.pth')
            message = f'Epoch {ep} training finished, pose loss: {avg_pose_loss:.6f} model saved'
            print(message)
            logger.info(message)
        
        # Evaluate the model every 5 epochs
        if ep != 0 and ep % 5 == 0: #ep > args.epochs_warmup+args.epochs_joint:
            print('Evaluating the model')
            logger.info('Evaluating the model')
            with torch.no_grad(): 
                model.eval()
                errors = tester.eval(model, num_gpu=len(gpu_ids))
            
            tester.generate_plots(save_plots_path, 10)
        
            t_rel = np.mean([errors[i]['t_rel'] for i in range(len(errors))])
            r_rel = np.mean([errors[i]['r_rel'] for i in range(len(errors))])
            t_rmse = np.mean([errors[i]['t_rmse'] for i in range(len(errors))])
            r_rmse = np.mean([errors[i]['r_rmse'] for i in range(len(errors))])
            usage = np.mean([errors[i]['usage'] for i in range(len(errors))])

            if t_rel < best:
                best = t_rel
                if rank == 0:
                    torch.save(model.module.state_dict(), f'{checkpoints_dir}/best_{best:.2f}.pth')
        
            message = f'Epoch {ep} evaluation finished , t_rel: {t_rel:.4f}, r_rel: {r_rel:.4f}, t_rmse: {t_rmse:.4f}, r_rmse: {r_rmse:.4f}, usage: {usage:.4f}, best t_rel: {best:.4f}'
            logger.info(message)
            print(message)
    
    message = f'Training finished, best t_rel: {best:.4f}'
    logger.info(message)
    print(message)

if __name__ == "__main__":
    main()




