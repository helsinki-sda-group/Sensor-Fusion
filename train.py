import argparse
import os
import logging
from path import Path
from utils import custom_transform
from dataset.KITTI_dataset import KITTI
from model import DeepVIO
from collections import defaultdict
from utils.kitti_eval import KITTI_tester
import numpy as np
import math
from pt_distr_env import DistributedEnviron

import torch
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
parser.add_argument('--epochs_fine', type=int, default=21, help='number of epochs for finetuning')
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

def update_status(ep, args, model):
    if ep < args.epochs_warmup:  # Warmup stage
        lr = args.lr_warmup
        selection = 'random'
        temp = args.temp_init
        for param in model.module.Policy_net.parameters(): # Disable the policy network
            param.requires_grad = False
    elif ep >= args.epochs_warmup and ep < args.epochs_warmup + args.epochs_joint: # Joint training stage
        lr = args.lr_joint
        selection = 'gumbel-softmax'
        temp = args.temp_init * math.exp(-args.eta * (ep-args.epochs_warmup))
        for param in model.module.Policy_net.parameters(): # Enable the policy network
            param.requires_grad = True
    elif ep >= args.epochs_warmup + args.epochs_joint: # Finetuning stage
        lr = args.lr_fine
        selection = 'gumbel-softmax'
        temp = args.temp_init * math.exp(-args.eta * (ep-args.epochs_warmup))
    return lr, selection, temp

def train(model, optimizer, train_loader, selection, temp, logger, ep, p=0.5, weighted=False, device='cuda:0'):
    
    mse_losses = []
    penalties = []
    data_len = len(train_loader)

    for i, (imgs, imus, gts, rot, weight) in enumerate(train_loader):

        imgs = imgs.to(device).float()
        imus = imus.to(device).float()
        gts = gts.to(device).float() 
        weight = weight.to(device).float()

        optimizer.zero_grad()
                
        poses, decisions, probs, _ = model(imgs, imus, is_first=True, hc=None, temp=temp, selection=selection, p=p)
        
        if not weighted:
            angle_loss = torch.nn.functional.mse_loss(poses[:,:,:3], gts[:, :, :3])
            translation_loss = torch.nn.functional.mse_loss(poses[:,:,3:], gts[:, :, 3:])
        else:
            weight = weight/weight.sum()
            angle_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,:3] - gts[:, :, :3]) ** 2).mean()
            translation_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,3:] - gts[:, :, 3:]) ** 2).mean()
        
        pose_loss = 100 * angle_loss + translation_loss        
        penalty = (decisions[:,:,0].float()).sum(-1).mean()
        loss = pose_loss + args.Lambda * penalty 
        
        loss.backward()
        optimizer.step()
        
        if i % args.print_frequency == 0: 
            message = f'Epoch: {ep}, iters: {i}/{data_len}, pose loss: {pose_loss.item():.6f}, penalty: {penalty.item():.6f}, loss: {loss.item():.6f}'
            print(message)
            logger.info(message)

        mse_losses.append(pose_loss.item())
        penalties.append(penalty.item())

    return np.mean(mse_losses), np.mean(penalties)


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
    transform_train = custom_transform.Compose(transform_train)

    #batch_size_per_gpu = int(args.batch_size / torch.cuda.device_count())

    train_dataset = KITTI(args.data_dir,
                    sequence_length=args.seq_len,
                    train_seqs=args.train_seq,
                    transform=transform_train
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
    tester = KITTI_tester(args)

    # Model initialization
    model = DeepVIO(args)

    # Continual training or not
    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')
    
    # Use the pre-trained flownet or not
    if args.pretrain_flownet and args.pretrain is None:
        pretrained_w = torch.load(args.pretrain_flownet, map_location='cpu')
        model_dict = model.Feature_net.state_dict()
        update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
        model_dict.update(update_dict)
        model.Feature_net.load_state_dict(model_dict)

    # Feed model to GPU
    model.to(f"cuda:{device}")
    model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    pretrain = args.pretrain 
    init_epoch = int(pretrain[-7:-4])+1 if args.pretrain is not None else 0    
    
    # Initialize the optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
    
    best = 10000

    for ep in range(init_epoch, args.epochs_warmup+args.epochs_joint+args.epochs_fine):
        
        lr, selection, temp = update_status(ep, args, model)
        optimizer.param_groups[0]['lr'] = lr
        message = f'Epoch: {ep}, lr: {lr}, selection: {selection}, temperaure: {temp:.5f}'
        print(message)
        logger.info(message)

        model.train()
        avg_pose_loss, avg_penalty_loss = train(model, optimizer, train_loader, selection, temp, logger, ep, p=0.5, device=device)
        
        # Save the model after training
        torch.save(model.module.state_dict(), f'{checkpoints_dir}/{ep:003}.pth')
        message = f'Epoch {ep} training finished, pose loss: {avg_pose_loss:.6f}, penalty_loss: {avg_penalty_loss:.6f}, model saved'
        print(message)
        logger.info(message)
        
        if ep != 0 and ep % 5 == 0:
            # Evaluate the model
            print('Evaluating the model')
            logger.info('Evaluating the model')
            with torch.no_grad(): 
                model.eval()
                errors = tester.eval(model, selection='gumbel-softmax', num_gpu=world_size)#len(gpu_ids))

            tester.generate_plots(save_plots_path, 10)
        
            t_rel = np.mean([errors[i]['t_rel'] for i in range(len(errors))])
            r_rel = np.mean([errors[i]['r_rel'] for i in range(len(errors))])
            t_rmse = np.mean([errors[i]['t_rmse'] for i in range(len(errors))])
            r_rmse = np.mean([errors[i]['r_rmse'] for i in range(len(errors))])
            usage = np.mean([errors[i]['usage'] for i in range(len(errors))])

            if t_rel < best:
                best = t_rel 
                torch.save(model.module.state_dict(), f'{checkpoints_dir}/best_{best:.2f}.pth')
        
            message = f'Epoch {ep} evaluation finished , t_rel: {t_rel:.4f}, r_rel: {r_rel:.4f}, t_rmse: {t_rmse:.4f}, r_rmse: {r_rmse:.4f}, usage: {usage:.4f}, best t_rel: {best:.4f}'
            logger.info(message)
            print(message)
    
    message = f'Training finished, best t_rel: {best:.4f}'
    logger.info(message)
    print(message)

if __name__ == "__main__":
    main()




