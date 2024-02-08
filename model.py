import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math


from gmflow.gmflow import GMFlow
from gmflow.gmflow_lite import GMFlowLite
from gmflow.backbone import CNNEncoder
from gmflow.transformer import *

from unimatch.unimatch import UniMatch


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )

# The inertial encoder for raw imu data
class Inertial_encoder(nn.Module):
    def __init__(self, opt):
        super(Inertial_encoder, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(15, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout))
        self.proj = nn.Linear(256 * 1 * (opt.skip_frames*10+1), opt.i_f_len)
        self.dim = opt.i_f_len

    def forward(self, x):
        #  x: (N, seq_len, 11, 6)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))    # x: (N x seq_len, 11, 6)
        x = self.encoder_conv(x.permute(0, 2, 1))                 # x: (N x seq_len, 64, 11)
        out = self.proj(x.view(x.shape[0], -1))                   # out: (N x seq_len, 256)
        return out.view(batch_size, seq_len, self.dim)            # return: (N, seq_len, 256)

class VisualEncoderCNN(nn.Module):
    def __init__(self, opt):
        super(VisualEncoderCNN, self).__init__()
        # CNN
        self.opt = opt
        self.conv1 = conv(True, 6, 64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2 = conv(True, 64, 128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3 = conv(True, 128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.conv3_1 = conv(True, 256, 256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4 = conv(True, 256, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv4_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv5 = conv(True, 512, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv5_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv6 = conv(True, 512, 1024, kernel_size=3, stride=2, dropout=0.5)
        # Comput the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, opt.img_w, opt.img_h))
        __tmp = self.encode_image(__tmp)

        self.visual_head = nn.Linear(int(np.prod(__tmp.size())), opt.v_f_len)

    def forward(self, img):
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2)
        batch_size = v.size(0)
        seq_len = v.size(1)

        # image CNN
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        v = self.encode_image(v)
        v = v.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        v = self.visual_head(v)  # (batch, seq_len, 256)
        return v

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        # CNN
        self.opt = opt
        self.conv1 = conv(True, 6, 64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2 = conv(True, 64, 128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3 = conv(True, 128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.conv3_1 = conv(True, 256, 256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4 = conv(True, 256, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv4_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv5 = conv(True, 512, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv5_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv6 = conv(True, 512, 1024, kernel_size=3, stride=2, dropout=0.5)
        # Comput the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, opt.img_w, opt.img_h))
        __tmp = self.encode_image(__tmp)

        self.N = opt.skip_frames
        self.visual_head = nn.Linear(int(np.prod(__tmp.size())), opt.v_f_len)

        if opt.imu_encoder == 'InertialTransformer':
            self.inertial_encoder = InertialTransformer(opt)
        else:
            self.inertial_encoder = Inertial_encoder(opt)


    def forward(self, img, imu):
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2)
        batch_size = v.size(0)
        seq_len = v.size(1)

        # image CNN
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        v = self.encode_image(v)
        #print('after CNN', v.shape)
        v = v.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        #print("before visual_head: ", v.shape)
        v = self.visual_head(v)  # (batch, seq_len, 256)
        #print("after visual_head: ", v.shape)
        
        # IMU CNN
        imu = torch.cat([imu[:, i * 10*self.N:i * 10*self.N + (10*self.N + 1), :].unsqueeze(1) for i in range(seq_len)], dim=1) # (N, seq_len, 11, 6)
        imu = self.inertial_encoder(imu)
        return v, imu

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6


# The fusion module
class Fusion_module(nn.Module):
    def __init__(self, opt):
        super(Fusion_module, self).__init__()
        self.fuse_method = opt.fuse_method
        self.f_len = opt.i_f_len + opt.v_f_len
        if self.fuse_method == 'soft':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, self.f_len))
        elif self.fuse_method == 'hard':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, 2 * self.f_len))
        elif self.fuse_method == 'cross-att':
            self.net = MHABlock(n_features=1, n_heads=1)


    def forward(self, v, i):
        if self.fuse_method == 'cat':
            return torch.cat((v, i), -1)
        elif self.fuse_method == 'soft':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            return feat_cat * weights
        elif self.fuse_method == 'hard':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            weights = weights.view(v.shape[0], v.shape[1], self.f_len, 2)
            mask = F.gumbel_softmax(weights, tau=1, hard=True, dim=-1)
            return feat_cat * mask[:, :, :, 0]

# The policy network module
class PolicyNet(nn.Module):
    def __init__(self, opt):
        super(PolicyNet, self).__init__()
        in_dim = opt.rnn_hidden_size + opt.i_f_len
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2))

    def forward(self, x, temp):
        logits = self.net(x)
        hard_mask = F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
        return logits, hard_mask

# The pose estimation network
class Pose_RNN(nn.Module):
    def __init__(self, opt):
        super(Pose_RNN, self).__init__()

        if opt.cross_fusion:
            f_len = opt.v_f_len
            self.fuse = CrossFusion(opt)
        else:
            f_len = opt.v_f_len + opt.i_f_len
            self.fuse = Fusion_module(opt)

        # The main RNN network
        self.rnn = nn.LSTM(
            input_size=f_len,
            hidden_size=opt.rnn_hidden_size,
            num_layers=2,
            dropout=opt.rnn_dropout_between,
            batch_first=True)

        # The output networks
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.regressor_t = nn.Sequential(
            nn.Linear(opt.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 3))

        self.regressor_q = nn.Sequential(
            nn.Linear(opt.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 3))

    def forward(self, fv, fv_alter, fi, dec, prev=None):
        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())
        
        # Select between fv and fv_alter
        v_in = fv * dec[:, :, :1] + fv_alter * dec[:, :, -1:] if fv_alter is not None else fv
        fused = self.fuse(v_in, fi)
        
        out, hc = self.rnn(fused) if prev is None else self.rnn(fused, prev)
        out = self.rnn_drop_out(out)
        pose = torch.concat([self.regressor_t(out), self.regressor_q(out)], dim=-1)

        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        return pose, hc
    
# The pose estimation network
class RNN(nn.Module):
    def __init__(self, opt, f_len=None):
        super(RNN, self).__init__()

        if f_len is None:
            f_len = opt.v_f_len + opt.i_f_len #opt.feature_channels

        # The main RNN network
        self.rnn = nn.LSTM(
            input_size=f_len,
            hidden_size=opt.rnn_hidden_size,
            num_layers=2,
            dropout=opt.rnn_dropout_between,
            batch_first=True)

        # The output networks
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.regressor = nn.Sequential(
            nn.Linear(opt.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6))

    def forward(self, fused, prev=None):
        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())
        
        out, hc = self.rnn(fused) if prev is None else self.rnn(fused, prev)
        out = self.rnn_drop_out(out)
        pose = self.regressor(out)
        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        return pose, hc


class DeepVIO(nn.Module):
    def __init__(self, opt):
        super(DeepVIO, self).__init__()

        self.Feature_net = Encoder(opt)
        self.Pose_net = Pose_RNN(opt)
        self.Policy_net = PolicyNet(opt)
        self.opt = opt
        
        initialization(self)

    def forward(self, img, imu, is_first=True, hc=None, temp=5, selection='gumbel-softmax', p=0.5):

        fv, fi = self.Feature_net(img, imu)
        batch_size = fv.shape[0]
        seq_len = fv.shape[1]

        poses, decisions, logits= [], [], []
        hidden = torch.zeros(batch_size, self.opt.rnn_hidden_size).to(fv.device) if hc is None else hc[0].contiguous()[:, -1, :]
        fv_alter = torch.zeros_like(fv) # zero padding in the paper, can be replaced by other 
        
        for i in range(seq_len):
            if i == 0 and is_first:
                # The first relative pose is estimated by both images and imu by default
                pose, hc = self.Pose_net(fv[:, i:i+1, :], None, fi[:, i:i+1, :], None, hc)
            else:
                if selection == 'gumbel-softmax':
                    # Otherwise, sample the decision from the policy network
                    p_in = torch.cat((fi[:, i, :], hidden), -1)
                    logit, decision = self.Policy_net(p_in.detach(), temp)
                    decision = decision.unsqueeze(1)
                    logit = logit.unsqueeze(1)
                    pose, hc = self.Pose_net(fv[:, i:i+1, :], fv_alter[:, i:i+1, :], fi[:, i:i+1, :], decision, hc)
                    decisions.append(decision)
                    logits.append(logit)
                elif selection == 'random':
                    decision = (torch.rand(fv.shape[0], 1, 2) < p).float()
                    decision[:,:,1] = 1-decision[:,:,0]
                    decision = decision.to(fv.device)
                    logit = 0.5*torch.ones((fv.shape[0], 1, 2)).to(fv.device)
                    pose, hc = self.Pose_net(fv[:, i:i+1, :], fv_alter[:, i:i+1, :], fi[:, i:i+1, :], decision, hc)
                    decisions.append(decision)
                    logits.append(logit)
            poses.append(pose)
            hidden = hc[0].contiguous()[:, -1, :]

        poses = torch.cat(poses, dim=1)
        decisions = torch.cat(decisions, dim=1)
        logits = torch.cat(logits, dim=1)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        return poses, decisions, probs, hc


class CrossFusion(nn.Module):
    """Applies channel-wise cross attention"""
    def __init__(self, opt):
        super(CrossFusion, self).__init__()
        self.mha = MHABlock(n_features=opt.seq_len-1, n_heads=opt.nhead, n_hidden=opt.i_f_len, dropout=0.1)

    def forward(self, img_emb, imu_emb):
        img_emb = img_emb.permute(0,2,1)
        imu_emb = imu_emb.permute(0,2,1)
        for layer in self.mha:
            img_emb = layer(img_emb, imu_emb)
        img_emb = img_emb.permute(0,2,1)
        return img_emb
    
#===============================================================================================
    


# Baseline-no-IMU
class DeepCrossVIO(nn.Module):
    def __init__(self, opt):
        super(DeepCrossVIO, self).__init__()

        self.Feature_net = VisualEncoderCNN(opt)
        self.Pose_net = RNN(opt, f_len=opt.v_f_len)
        self.opt = opt
        
    def forward(self,img, imu, is_first=True, hc=None, save_flow=False):

        fv = self.Feature_net(img)
        poses, hc = self.Pose_net(fv, hc)

        return poses, hc

# IMU-queries-int-multiloss
class DeepCrossVIO(nn.Module):
    def __init__(self, opt):
        super(DeepCrossVIO, self).__init__()

        self.Feature_net = Encoder(opt)
        self.Pose_net = RNN(opt, f_len=opt.v_f_len)
        self.d_model = opt.v_f_len
        self.fusion = MHABlock(n_features=self.d_model, n_heads=4, n_hidden=self.d_model, dropout=0.2)
        self.opt = opt

    def integrate_acceleration_velocity(self, a, dt, v0=0, s0=0):
        """ 
        Numerically integrate batched linear acceleration and angular velocity data to find linear displacement and angular displacement.
        
        Parameters:
        - a (torch.Tensor): Data with shape [B, T, 6] (first 3 channels for linear acceleration, last 3 for angular velocity)
        - dt (float): Time step between samples
        - v0 (float or torch.Tensor): Initial linear velocity (default is 0)
        - s0 (float or torch.Tensor): Initial linear and angular displacement (default is 0)
        
        Returns:
        - v (torch.Tensor): Linear velocity data with shape [B, T, 3]
        - s (torch.Tensor): Linear and angular displacement data with shape [B, T, 6]
        """
        
        B, T, C = a.shape
        v = torch.zeros_like(a[:, :, :3])  # Only linear velocity needs to be calculated
        s = torch.zeros_like(a)  # Both linear and angular displacement will be calculated
        
        # Compute linear velocity using trapezoidal rule
        v[:, 1:, :] = torch.cumsum(0.5 * (a[:, :-1, :3] + a[:, 1:, :3]) * dt, dim=1)
        
        # Compute linear displacement using trapezoidal rule
        s[:, 1:, :3] = torch.cumsum(0.5 * (v[:, :-1, :] + v[:, 1:, :]) * dt, dim=1)
        
        # Compute angular displacement using trapezoidal rule
        s[:, 1:, 3:] = torch.cumsum(0.5 * (a[:, :-1, 3:] + a[:, 1:, 3:]) * dt, dim=1)
        
        # Add initial conditions
        if torch.is_tensor(v0):
            v += v0[:, None, :]
        else:
            v += v0
        
        if torch.is_tensor(s0):
            s += s0[:, None, :]
        else:
            s += s0
        
        return v, s
        
    def forward(self,img, imu, is_first=True, hc=None, save_flow=False):
        v, s = self.integrate_acceleration_velocity(imu, 1/101)
        fi = torch.cat([imu, v, s], dim=-1) # [B, S, 15]
        fv, fi = self.Feature_net(img, fi)
        fv = F.layer_norm(fv, [self.d_model])
        fi = F.layer_norm(fi, [self.d_model])
        fused = self.fusion(fi, fv)
        if self.training:
            hc_vis, hc_imu = None, None
            vis_poses, hc_vis = self.Pose_net(fv, hc_vis)
            imu_poses, hc_imu = self.Pose_net(fi, hc_imu)
        else:
            vis_poses = None
            imu_poses = None
        poses, hc = self.Pose_net(fused, hc)

        return poses, vis_poses, imu_poses, hc

# # IMU-queries-int-multiloss
# class DeepCrossVIO(nn.Module):
#     def __init__(self, opt):
#         super(DeepCrossVIO, self).__init__()

#         self.Feature_net = Encoder(opt)
#         self.d_model = opt.v_f_len + opt.i_f_len
#         self.fusion = TransformerEncoder(self.d_model)
#         self.regress = nn.Sequential(
#             nn.Linear(self.d_model, 128),
#             nn.GELU(),
#             nn.Linear(128, 6))
#         self.opt = opt

#     def integrate_acceleration_velocity(self, a, dt, v0=0, s0=0):
#         """ 
#         Numerically integrate batched linear acceleration and angular velocity data to find linear displacement and angular displacement.
        
#         Parameters:
#         - a (torch.Tensor): Data with shape [B, T, 6] (first 3 channels for linear acceleration, last 3 for angular velocity)
#         - dt (float): Time step between samples
#         - v0 (float or torch.Tensor): Initial linear velocity (default is 0)
#         - s0 (float or torch.Tensor): Initial linear and angular displacement (default is 0)
        
#         Returns:
#         - v (torch.Tensor): Linear velocity data with shape [B, T, 3]
#         - s (torch.Tensor): Linear and angular displacement data with shape [B, T, 6]
#         """
        
#         B, T, C = a.shape
#         v = torch.zeros_like(a[:, :, :3])  # Only linear velocity needs to be calculated
#         s = torch.zeros_like(a)  # Both linear and angular displacement will be calculated
        
#         # Compute linear velocity using trapezoidal rule
#         v[:, 1:, :] = torch.cumsum(0.5 * (a[:, :-1, :3] + a[:, 1:, :3]) * dt, dim=1)
        
#         # Compute linear displacement using trapezoidal rule
#         s[:, 1:, :3] = torch.cumsum(0.5 * (v[:, :-1, :] + v[:, 1:, :]) * dt, dim=1)
        
#         # Compute angular displacement using trapezoidal rule
#         s[:, 1:, 3:] = torch.cumsum(0.5 * (a[:, :-1, 3:] + a[:, 1:, 3:]) * dt, dim=1)
        
#         # Add initial conditions
#         if torch.is_tensor(v0):
#             v += v0[:, None, :]
#         else:
#             v += v0
        
#         if torch.is_tensor(s0):
#             s += s0[:, None, :]
#         else:
#             s += s0
        
#         return v, s
        
#     def forward(self,img, imu, is_first=True, hc=None, save_flow=False):
#         v, s = self.integrate_acceleration_velocity(imu, 1/101)
#         fi = torch.cat([imu, v, s], dim=-1) # [B, S, 15]
#         fv, fi = self.Feature_net(img, fi)

#         #Concat features channel-wise
#         fused = torch.cat([fv, fi], dim=-1)
#         fused = F.layer_norm(fused, [self.d_model])
#         fused = self.fusion(fused)
#         poses = self.regress(fused)

#         return poses, hc
    
# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim):
        super(TransformerEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, batch_first=True, activation="gelu"),
            num_layers=3)
    
    def forward(self, x):
        return self.transformer(x)
    
# Perceiver block for iterative attention
class PerceiverBlock(nn.Module):
    def __init__(self, opt):
        super(PerceiverBlock, self).__init__()
        self.d_model = opt.v_f_len
        self.cross_attention = MHABlock(n_features=self.d_model, n_heads=1, n_hidden=self.d_model // 2, dropout=0.0)
        self.self_attention = MHABlock(n_features=self.d_model, n_heads=1, n_hidden=self.d_model // 2, dropout=opt.transformer_dropout)
    def forward(self, q, kv, attn_mask=None, causal_mask=None):
        q = self.cross_attention(q, kv, attn_mask=attn_mask)
        q = self.self_attention(q, q, attn_mask=causal_mask)
        return q

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(1, max_len, d_model)
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1),:].to(x.device)
        return self.dropout(x)
    
# # SimpleTransformerFusion: Testing very simple addition sensor fusion.
# class IterativeAttention(nn.Module):
#     def __init__(self, opt):
#         super(IterativeAttention, self).__init__()
#         self.d_model = opt.v_f_len
#         self.vis_encoder = VisualEncoderCNN(opt)
#         self.imu_encoder = nn.Sequential(nn.Linear(6, self.d_model // 2), nn.ReLU(), nn.Linear(self.d_model // 2, self.d_model))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))  # class token (global information)
#         self.positional_encoding = PositionalEncoding(d_model=512, max_len=111, dropout=0.1)
#         self.fusion = MHABlock(n_features=opt.v_f_len, n_heads=opt.nhead, n_hidden=opt.feature_channels, dropout=0.1)
#         self.Pose_net = RNN(opt, f_len=opt.v_f_len)
        
#     def forward(self,img, imu, is_first=True, hc=None, save_flow=False):

#         fv = self.vis_encoder(img)
#         fi = self.positional_encoding(self.imu_encoder(imu))
#         token = self.cls_token.expand(fv.size(0), -1, -1)
#         fused = torch.cat([self.fusion(torch.cat([token, fv[:,i:i+1], fi[:, i * 10:i * 10 + 11]], dim=1))[:,0] for i in range(fv.size(1))], dim=1)
#         poses, hc = self.Pose_net(fused, hc)

#         return poses, hc

# SimpleCrossAttention: Testing cross attention
class IterativeAttention(nn.Module):
    def __init__(self, opt):
        super(IterativeAttention, self).__init__()
        self.d_model = opt.v_f_len
        self.vis_encoder = VisualEncoderCNN(opt)
        self.imu_encoder = nn.Sequential(nn.Linear(6, self.d_model // 2), nn.ReLU(), nn.Linear(self.d_model // 2, self.d_model))
        self.positional_encoding = PositionalEncoding(d_model=512, max_len=111, dropout=0.1)
        self.fusion = MHABlock(n_features=opt.v_f_len, n_heads=opt.nhead, n_hidden=opt.feature_channels, dropout=0.1)
        self.Pose_net = RNN(opt, f_len=opt.v_f_len)
        
    def forward(self,img, imu, is_first=True, hc=None, save_flow=False):

        fv = self.vis_encoder(img)
        fi = self.positional_encoding(self.imu_encoder(imu))
        fused = torch.cat([self.fusion(fv[:,i:i+1], fi[:, i * 10:i * 10 + 11]) for i in range(fv.size(1))], dim=1)
        fused_poses = self.Pose_net.regressor(fused)
        poses, hc = self.Pose_net(fused, hc)

        return poses, fused_poses, hc
    
# IterativeAttention-2nd-ed
# class IterativeAttention(nn.Module):
#     def __init__(self, opt):
#         super(IterativeAttention, self).__init__()
#         self.d_model = opt.v_f_len
#         self.vis_encoder = VisualEncoderCNN(opt)
#         self.imu_encoder = nn.Sequential(nn.Linear(6, self.d_model // 2), nn.ReLU(), nn.Linear(self.d_model // 2, self.d_model))

#         self.first_fusion = PerceiverBlock(opt) # First fusion used once
#         self.iterative_fusion = PerceiverBlock(opt) # Using weight sharing
#         self.Pose_net = RNN(opt, f_len=self.d_model)

#     def forward(self,img, imu, is_first=True, hc=None, save_flow=False):
#         query_len = img.size(1) - 1     # Sequence length
#         key_len = imu.size(1)           # IMU length
#         window = 11                     # IMU points per img

#         # Initial encoding and tokenization
#         vis_f = self.vis_encoder(img)
#         imu_f = self.imu_encoder(imu)

#         # Define the local attention window and set those values to 0
#         mask = torch.full((query_len, key_len), float("-inf"), device=vis_f.device)
#         for i in range(query_len):
#             start = i * (window-1)
#             end = start + window
#             mask[i, start:end] = 0.0

#         #causal_mask = torch.triu(torch.ones((query_len, query_len), device=vis_f.device), diagonal=1).bool()

#         # Fusion
#         fused = self.first_fusion(vis_f, imu_f, attn_mask=mask, causal_mask=None)
#         for _ in range(5):
#             fused = self.iterative_fusion(fused, imu_f, attn_mask=mask, causal_mask=None)

#         # Compute pose at multiple steps
#         poses_fused = self.Pose_net.regressor(fused)

#         # Temporal modelling
#         poses, hc = self.Pose_net(fused, hc)

#         return poses, poses_fused, hc

class CrossAttentionVIO(nn.Module):
    def __init__(self, opt):
        super(CrossAttentionVIO, self).__init__()
        self.Feature_net = Encoder(opt)
        self.visual_process = MHABlock(n_features=opt.v_f_len, n_heads=opt.nhead, n_hidden=opt.feature_channels // 2, dropout=0.1)
        self.inertial_process = MHABlock(n_features=opt.v_f_len, n_heads=opt.nhead, n_hidden=opt.feature_channels // 2, dropout=0.1)

        self.fusion =  MHABlock(n_features=opt.v_f_len, n_heads=opt.nhead, n_hidden=opt.feature_channels // 2, dropout=0.1)
        self.Pose_net = RNN(opt, f_len=opt.v_f_len)
        self.opt = opt
        self.window_len = 1
        
        initialization(self)

    def forward(self,img, imu, is_first=True, hc=None, save_flow=False):

        fv, fi = self.Feature_net(img, imu)
        seq_len, imu_len = fv.shape[1], fi.shape[1]

        # Define the local attention window
        query_len = seq_len
        key_len = imu_len
        window = 4
        mask = torch.full((query_len, key_len), float("-inf"), device=fv.device)
        
        # Define the local attention window and set those values to 0
        for i in range(query_len):
            start = max(0, i - self.window_len)
            end = min(key_len, (i) + self.window_len + 1)
            mask[i, start:end] = 0.0

        # Self-att. on features seperately
        fv = self.visual_process(fv, fv, attn_mask=mask)
        fi = self.inertial_process(fi, fi, attn_mask=mask)

        # Feature fusion with CrossAttention
        fused = self.fusion(fi, fv, attn_mask=mask)

        # Pose regression
        vis_poses = []
        imu_poses = []
        poses = []
        hc_v, hc_i = hc, hc
        for i in range(seq_len):
            if self.training:
                vis_pose, hc_v = self.Pose_net(fv[:, i:i+1, :], hc_v)
                imu_pose, hc_i = self.Pose_net(fi[:, i:i+1, :], hc_i)
                vis_poses.append(vis_pose)
                imu_poses.append(imu_pose)
            else:
                vis_poses = None
                imu_poses = None
            pose, hc = self.Pose_net(fused[:, i:i+1, :], hc)
            poses.append(pose)

        poses = torch.cat(poses, dim=1)

        if self.training:
            vis_poses = torch.cat(vis_poses, dim=1)
            imu_poses = torch.cat(imu_poses, dim=1)
        return poses, vis_poses, imu_poses, hc


## Alternative implementation without RNN
# class CrossAttentionVIO(nn.Module):
#     def __init__(self, opt):
#         super(CrossAttentionVIO, self).__init__()
#         self.Feature_net = Encoder(opt)

#         self.visual_process = MHABlock(n_features=opt.v_f_len, n_heads=opt.nhead, n_hidden=opt.feature_channels // 2, dropout=0.1)
#         self.inertial_process = MHABlock(n_features=opt.v_f_len, n_heads=opt.nhead, n_hidden=opt.feature_channels // 2, dropout=0.1)
#         self.fusion = MHABlock(n_features=opt.v_f_len, n_heads=opt.nhead, n_hidden=opt.feature_channels // 2, dropout=0.1)

#         # Pose regression
#         self.regressor_t = nn.Sequential(
#             nn.Linear(opt.feature_channels, 128),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Linear(128, 3))

#         self.regressor_q = nn.Sequential(
#             nn.Linear(opt.feature_channels, 128),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Linear(128, 3))
        
#         #self.layer_norm = nn.LayerNorm([opt.seq_len-1, opt.feature_channels])

#         self.opt = opt
#         self.window_len = 4
        
#         initialization(self)

#     def forward(self,img, imu, is_first=True, hc=None, save_flow=False):

#         #### FEATURE EXTRACTION ####
#         fv, fi = self.Feature_net(img, imu)
#         query_len, key_len = fv.shape[1], fi.shape[1]

#         if hc is not None:
#             fv = torch.cat([hc[0], fv], dim=1)
#             fi = torch.cat([hc[1], fi], dim=1)
#             query_len += self.window_len
#             key_len += self.window_len

#         hc_t = [fv[:,-self.window_len:].detach(), fi[:,-self.window_len:].detach()]

#         #### FUSION ####
#         # Define the local attention window
#         mask = torch.full((query_len, key_len), float("-inf"), device=fv.device)
        
#         # Define the local attention window and set those values to 0
#         for i in range(query_len):
#             start = max(0, i - self.window_len)
#             end = min(key_len, (i) + self.window_len + 1)
#             mask[i, start:end] = 0.0

#         # Self-att. on features seperately
#         fv = self.visual_process(fv, fv, attn_mask=mask)
#         fi = self.inertial_process(fi, fi, attn_mask=mask)

#         # Feature fusion with CrossAttention
#         fused = self.fusion(fv, fi, attn_mask=mask) # [N, seq, D]
        
#         # Trim preds
#         if hc is not None:
#             fv = fv[:, self.window_len:,]
#             fi = fi[:, self.window_len:,]
#             fused = fused[:, self.window_len:,]

#         # Pose regression (summation)
#         if self.training:
#             vis_poses = torch.cat([self.regressor_t(fv), self.regressor_q(fv)], dim=-1)
#             imu_poses = torch.cat([self.regressor_t(fi), self.regressor_q(fi)], dim=-1)
#         else:
#             vis_poses = None
#             imu_poses = None
#         combined_poses = torch.cat([self.regressor_t(fused), self.regressor_q(fused)], dim=-1)

#         return combined_poses, vis_poses, imu_poses, hc_t


class CrossVIO(nn.Module):
    def __init__(self, opt):
        super(CrossVIO, self).__init__()

        self.image_encoder = GMFlow(opt)
        self.flow_conv = CNNEncoder(input_dim=2, output_dim=128, num_output_scales=1, feature_dims=[64, 96, 128])
        self.visual_head = nn.Linear(262144, opt.feature_channels)
        self.imu_encoder = Inertial_encoder(opt)
        self.cross_fusion = CrossFusion(opt)
        self.pose_rnn = FusedPoseRNN(opt)

        pretrained_weights = torch.load(opt.gmflow_weights)['model'] if opt.gmflow_weights else None

        self.opt = opt
        
        # Initialize weights
        initialization(self)

        # Load state dict for GMFlow
        if pretrained_weights:
            print("Loading pretrained weights")
            with torch.no_grad():
                for name, param in self.image_encoder.named_parameters():
                    if name in pretrained_weights and param.data.shape == pretrained_weights[name].shape:
                        param.copy_(pretrained_weights[name])
                    else:
                        print(
                            f"""When loading weights for GMFlow. Skipped loading for {name} \n
                            {name} shape: {param.data.shape} != pretrained weights shape: {pretrained_weights[name].shape}"""
                        )
        # Freeze params for GMFlow
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, img, imu, is_first=True, hc=None):

        #print('Input data: ', img.shape, imu.shape)

        batch_size, seq_len, C, H, W = img.shape
        seq_len = seq_len - 1

        img1 = img[:, :-1].reshape(batch_size * seq_len, C, H, W)
        img2 = img[:, 1:].reshape(batch_size * seq_len, C, H, W)
        
        img_f = self.image_encoder(
            img1,
            img2,  
            attn_splits_list=[2],
            corr_radius_list=[-1],
            prop_radius_list=[-1],
        )['flow_preds'][-1]

        #print("flow preds :", [f.shape for f in img_f['flow_preds']])
        img_f = self.flow_conv(img_f)[0]  # [B*S, C, H, W]
        img_f = img_f.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        img_f = self.visual_head(img_f)

        imu = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
        imu_f = self.imu_encoder(imu) # [B, S, C]

        #print('Embeddings: ', img_f.shape, imu_f.shape)

        # Reshape + Linear before cross-attention
        # NOTE: Look into how the batch and seq_len dimensions are combined for attention, might be problematic.
        #imu_f = imu_f.flatten(0, 1).unsqueeze(1)  # [B,C]
        #img_f = img_f.flatten(2, 3).permute(0, 2, 1)  # [B, HxW, C]

        #print('Embeddings reshaped: ', img_f.shape, imu_f.shape)

        fused_f = self.cross_fusion(img_f, imu_f)

        #print("After Fusion: ", fused_f.shape)

        poses = []
        for i in range(seq_len):
            pose, hc = self.pose_rnn(fused_f[:, i:i+1, :], hc)
            poses.append(pose)
        #print('Pose: ', pose.shape,)

        poses = torch.cat(poses, dim=1)

        return poses, hc
    
# The pose estimation network
class FusedPoseRNN(nn.Module):
    def __init__(self, opt):
        super(FusedPoseRNN, self).__init__()

        # The main RNN network
        f_len = opt.feature_channels
        self.rnn = nn.LSTM(
            input_size=f_len,
            hidden_size=opt.rnn_hidden_size,
            num_layers=2,
            dropout=opt.rnn_dropout_between,
            batch_first=True)

        # The output networks
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.regressor_t = nn.Sequential(
            nn.Linear(opt.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 3))

        self.regressor_q = nn.Sequential(
            nn.Linear(opt.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 3))

    def forward(self, fused_f, prev=None):
        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())
        
        out, hc = self.rnn(fused_f) if prev is None else self.rnn(fused_f, prev)
        out = self.rnn_drop_out(out)
        pose = pose = torch.concat([self.regressor_t(out), self.regressor_q(out)], dim=2)
        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        return pose, hc
    
class LatentCrossFusion(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # Cross Attention
        self.cross_attn_ffn1 = TransformerLayer(
            d_model=opt.feature_channels,
            nhead=opt.num_head,
            attention_type="full_attention",
            ffn_dim_expansion=opt.ffn_dim_expansion,
            with_shift=opt.with_shift,
        )
        self.cross_attn_ffn2 = TransformerLayer(
            d_model=opt.feature_channels,
            nhead=opt.num_head,
            attention_type="full_attention",
            ffn_dim_expansion=opt.ffn_dim_expansion,
            with_shift=opt.with_shift,
        )
        # Cosine Similarity
        self.latent_loss = nn.CosineSimilarity(dim=2)

        # Self-Attention
        self.self_attn = TransformerLayer(
            d_model=opt.feature_channels,
            nhead=opt.num_head,
            attention_type="full_attention",
            no_ffn=True,
            ffn_dim_expansion=opt.ffn_dim_expansion,
            with_shift=opt.with_shift,
        )
        
        initialization(self)

    def forward(self, img_emb, imu_emb):
        # Cross-attention pass
        imu_emb_cross = self.cross_attn_ffn1(img_emb, imu_emb, attn_num_splits=1)  # TODO: make attn_num_splits a parameter to model
        img_emb_cross = self.cross_attn_ffn2(imu_emb, img_emb, attn_num_splits=1)

        # Latent Fusion Loss
        loss = 1 - self.latent_loss(imu_emb_cross, img_emb_cross)

        # Self-attention
        fused_emb = self.self_attn(imu_emb_cross, img_emb_cross, attn_num_splits=1)

        return fused_emb, loss


def initialization(net):
    #Initilization
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(0)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    start, end = n//4, n//2
                    param.data[start:end].fill_(1.)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

###### BOTTLENECK FUSION #######

class TransformerFusion(nn.Module):
    def __init__(
        self,
        opt,
        num_layers=6,
        num_bottleneck_layers=2,
        num_bottleneck_tokens=4,
        d_model=128,
        nhead=1,
        attention_type="swin",
        ffn_dim_expansion=4,
        **kwargs,
    ):
        super().__init__()
        self.d_model = opt.feature_channels

        self.visual_backbone = UniMatch(
                 num_scales=opt.num_scales,
                 feature_channels=128, #opt.feature_channels,
                 upsample_factor=opt.upsample_factor,
                 num_head=opt.num_head,
                 ffn_dim_expansion=opt.ffn_dim_expansion,
                 num_transformer_layers=opt.num_transformer_layers,
                 reg_refine=opt.reg_refine,  # optional local regression refinement
                 task='flow',
        )
        image_size = [int(np.ceil(opt.img_w / 32)) * 32, int(np.ceil(opt.img_h / 32)) * 32]
        patch_size = 32
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.flow2features = nn.Linear(2*patch_size**2, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model)

        #self.inertial_backbone = nn.Sequential(nn.Linear(10 * opt.skip_frames + 1, 128), nn.GELU(), nn.Linear(128, 128))

        self.output_pose = nn.Parameter(torch.randn(1, self.d_model))

        self.fusion = MHABlock(self.d_model, n_heads=2)

        #self.regressor = nn.Sequential(self.d_model, 32, nn.GELU(), nn.Linear(32, 1))
        self.rnn = RNN(opt, f_len=self.d_model)
        self.opt = opt
        
        self.attn_splits_list = opt.attn_splits_list
        self.corr_radius_list = opt.corr_radius_list
        self.prop_radius_list = opt.prop_radius_list

    def forward(
        self,
        img, # [B, S, C, H, W]
        imu, # [B, 10*S + 1, 6]
        is_first=True,
        hc=None,
        attn_num_splits=None,
        save_flow=False,
        **kwargs,
    ):
        ###### VISUAL ENCODING ######
        batch_size, seq_len, C, H, W = img.shape
        seq_len = seq_len - 1

        img1 = img[:, :-1].reshape(batch_size * seq_len, C, H, W)
        img2 = img[:, 1:].reshape(batch_size * seq_len, C, H, W)

        flow_features = self.visual_backbone.inference(
                                        img1, 
                                        img2, 
                                        attn_type='swin',
                                        padding_factor=32,
                                        attn_splits_list=self.attn_splits_list, 
                                        corr_radius_list=self.corr_radius_list, 
                                        prop_radius_list=self.prop_radius_list,
                                        save_flow=save_flow) # [B * seq, 2 , H/4, W/4]
        flow_features = self.unfold(flow_features).permute(0,2,1) # [B * seq, num_patches, patch_size*patch_size*channels]
        flow_features = self.positional_encoding(self.flow2features(flow_features))  # [B * seq, num_patches, patch_size*patch_size*channels]


        pose_emb = self.fusion(self.output_pose.expand(batch_size*seq_len, -1, -1), flow_features).view(batch_size, seq_len, self.d_model)

        poses, hc = self.rnn(pose_emb)

        #imu = torch.cat([imu[:, i * 10*self.N:i * 10*self.N + (10*self.N + 1), :].unsqueeze(1) for i in range(seq_len)], dim=1) # (N, seq_len, 11, 6)
        #imu_features = self.inertial_backbone(imu.permute(0,1,3,2))

        return poses, hc


    
class MHABlock(nn.Module):
    def __init__(self, n_features=128, n_heads=1, n_hidden=64, dropout=0.1, activation=F.gelu):
        super(MHABlock, self).__init__()
        self.norm1 = nn.LayerNorm(n_features)
        self.norm2 = nn.LayerNorm(n_features)
        self.MHA = nn.MultiheadAttention(n_features, n_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ff1 = nn.Linear(n_features, n_hidden)
        self.ff2 = nn.Linear(n_hidden, n_features)
        self.activation = activation

        if n_features % n_heads != 0:
            raise ValueError("n_features should be divisible by n_heads")

    def forward(self, q, kv, src_mask=None, tgt_mask=None, attn_mask=None, return_attention=False):
        z_att, attn_weights = self.MHA(q, kv, kv, key_padding_mask=src_mask, attn_mask=attn_mask)
        q = self.norm1(self.dropout(z_att) + q)

        q_ff = self.activation(self.dropout(self.ff1(q)))
        q_ff = self.ff2(q_ff)
        y = self.norm2(self.dropout(q_ff) + q)

        if return_attention:
            return y, attn_weights
        else:
            return y

class InertialTransformer(nn.Module):
    def __init__(
        self,
        opt,
        **kwargs,
    ):
        super().__init__()
        self.d_model = opt.i_f_len
        self.seq_len = opt.seq_len - 1
        self.tokenize = nn.Sequential(nn.Linear(15, self.d_model // 2), nn.ReLU(), nn.Linear(self.d_model // 2, self.d_model))
        self.class_token = torch.nn.Parameter(torch.randn(1, 1, self.d_model))  # aggregate IMU sequence
        self.layers = nn.ModuleList([MHABlock(n_features=self.d_model, n_heads=opt.nhead) for _ in range(3)])

    def forward(self, imu):
        batch_size, seq_len  = imu.size(0), self.seq_len
        tokens = self.tokenize(imu) # (N, seq_len, 11, D)
        tokens = tokens.view(batch_size * seq_len, tokens.size(2), tokens.size(3)) # (N x seq_len, 11, D)
        tokens = torch.cat([tokens, self.class_token.expand(tokens.size(0), -1, -1)], dim=1) # (N x seq_len, 12, D)
        mask = torch.triu(torch.ones((tokens.size(1), tokens.size(1)), dtype=torch.bool, device=tokens.device), diagonal=1)
        for layer in self.layers:
            tokens = layer(tokens, tokens, attn_mask=mask)
        return tokens[:,-1].view(batch_size, seq_len, -1)


class Perciever(nn.Module):
    def __init__(
        self,
        opt,
        **kwargs,
    ):
        super().__init__()

        self.opt = opt
        self.d_model = opt.feature_channels
        self.nhead = opt.nhead

        # Visual backbone (optical flow + feature propagration w/ transformer)
        self.visual_backbone = UniMatch(
                 num_scales=opt.num_scales,
                 feature_channels=128, #opt.feature_channels,
                 upsample_factor=opt.upsample_factor,
                 num_head=opt.num_head,
                 ffn_dim_expansion=opt.ffn_dim_expansion,
                 num_transformer_layers=opt.num_transformer_layers,
                 reg_refine=opt.reg_refine,  # optional local regression refinement
                 task='flow',
        )

        patch_size = 16
        self.num_patches = (opt.img_w // patch_size) * (opt.img_h // patch_size)
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size) # Patch size determined by GMFlow
        self.flow2features = nn.Linear(2*patch_size**2, self.d_model) # [BxS, P, D]

        # Inertial backbone
        # self.inertial_encoder = InertialTransformer(opt)
        self.tokenize_imu = nn.Sequential(nn.Linear(11, self.d_model // 2), nn.ReLU(), nn.Linear(self.d_model // 2, self.d_model))
        self.layers_imu = nn.ModuleList([MHABlock(n_features=self.d_model, n_heads=opt.nhead) for i in range(opt.num_bottleneck_layers)])

        # Process (self att.)
        self.class_token = torch.nn.Parameter(torch.randn(1, 1, self.d_model))
        self.process = nn.ModuleList([MHABlock(n_features=self.d_model, n_heads=self.nhead, n_hidden=self.d_model // 2, dropout=0.1) for i in range(opt.num_bottleneck_layers)])

        self.attn_splits_list = opt.attn_splits_list
        self.corr_radius_list = opt.corr_radius_list
        self.prop_radius_list = opt.prop_radius_list

    def forward(
        self,
        img, # [B, S, C, H, W]
        imu, # [B, 10*S + 1, 6]
        is_first=True,
        hc=None,
        attn_num_splits=None,
        save_flow=False,
        **kwargs,
    ):
        ###### VISUAL ENCODING ######
        batch_size, seq_len, C, H, W = img.shape
        seq_len = seq_len - 1

        img1 = img[:, :-1].reshape(batch_size * seq_len, C, H, W)
        img2 = img[:, 1:].reshape(batch_size * seq_len, C, H, W)

        flow_features = self.visual_backbone.inference(
                                        img1, 
                                        img2, 
                                        attn_type='swin',
                                        padding_factor=32,
                                        attn_splits_list=self.attn_splits_list, 
                                        corr_radius_list=self.corr_radius_list, 
                                        prop_radius_list=self.prop_radius_list,
                                        save_flow=save_flow) # [B * seq, 2 , H/4, W/4]
                
        #flow_features = torch.concat([img1, img2], dim=1) # channel concat
        flow_features = self.unfold(flow_features).permute(0,2,1) # [B * seq, num_patches, patch_size*patch_size*channels]
        flow_features = self.flow2features(flow_features)  # [B * seq, num_patches, patch_size*patch_size*channels]

        ###### INERTIAL ENCODING #######
        #imu = self.inertial_encoder(imu)
        imu = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1).view(batch_size*seq_len, 6, 11) # (N x seq_len, 6, 11)
        imu = self.tokenize_imu(imu)
        for layer in self.layers_imu:
            imu = layer(imu, imu)

        ##### FUSION W. SELF-ATT ######
        tokens = torch.cat([self.class_token.expand(imu.size(0), -1, -1), flow_features, imu], dim=1) # (N x seq_len, 1 + P + 6, D)
        for layer in self.process:
            tokens = layer(tokens, tokens)
        tokens = tokens[:,0].view(batch_size, seq_len, -1)

        return tokens



class BNF(nn.Module):
    def __init__(
        self,
        opt,
        **kwargs,
    ):
        super().__init__()

        self.opt = opt

        self.num_bottleneck_tokens = opt.num_bottleneck_tokens
        self.num_bottleneck_layers = opt.num_bottleneck_layers
        self.d_model = opt.feature_channels
        self.nhead = opt.nhead

        # Visual backbone (optical flow + feature propagration w/ transformer)
        self.visual_backbone = UniMatch(
                 num_scales=opt.num_scales,
                 feature_channels=opt.feature_channels,
                 upsample_factor=opt.upsample_factor,
                 num_head=opt.num_head,
                 ffn_dim_expansion=opt.ffn_dim_expansion,
                 num_transformer_layers=opt.num_transformer_layers,
                 reg_refine=opt.reg_refine,  # optional local regression refinement
                 task='flow',
        ) #GMFlow(opt)
        self.flow_conv = CNNEncoder(input_dim=2, output_dim=128, num_output_scales=1, feature_dims=[32, 96, 128])


        self.visual_cls = nn.Parameter(torch.randn(1, 1, self.d_model))  # class token (global information)
        self.visual_feature_propagation = FeatureFlowAttention(in_channels=2)

        patch_size = 16
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size) # Patch size determined by GMFlow
        self.flow2features = nn.Linear(self.d_model*patch_size**2, self.d_model)


        # Inertial backbone (CNN + Transformer)
        self.imu_feature_extract = IMUPatchEmbedding()
        self.imu_layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    attention_type="full_attention",
                    no_ffn=True,
                    ffn_dim_expansion=opt.ffn_dim_expansion,
                )
                for i in range(opt.num_layers)
            ]
        )
        #self.inertial_encoder = Inertial_encoder(opt)

        # Bottleneck fusion
        # self.bottleneck_layers = nn.ModuleList(
        #     [
        #         TransformerLayer(
        #             d_model=self.d_model,
        #             nhead=self.nhead,
        #             attention_type="full_attention",
        #             no_ffn=True,
        #             ffn_dim_expansion=opt.ffn_dim_expansion,
        #         )
        #         for i in range(self.num_bottleneck_layers * 2)
        #     ]
        # )
        self.fusion_layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    attention_type="full_attention",
                    no_ffn=True,
                    ffn_dim_expansion=opt.ffn_dim_expansion,
                )
                for i in range(self.num_bottleneck_layers)
            ]
        )


        self.ff1 = nn.Linear(self.d_model, self.d_model)

        # Initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Load and freeze GMFlow weights
        #self.visual_backbone.load_weights()
        #loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'

        self.attn_splits_list = self.opt.attn_splits_list
        self.corr_radius_list = self.opt.corr_radius_list
        self.prop_radius_list = self.opt.prop_radius_list

    def forward(
        self,
        img,
        imu,
        is_first=True,
        hc=None,
        attn_num_splits=None,
        save_flow=False,
        **kwargs,
    ):
        ###### VISUAL ENCODING ######
        batch_size, seq_len, C, H, W = img.shape
        seq_len = seq_len - 1

        img1 = img[:, :-1].reshape(batch_size * seq_len, C, H, W)
        img2 = img[:, 1:].reshape(batch_size * seq_len, C, H, W)

        flow_raw = self.visual_backbone.inference(
                                        img1, 
                                        img2, 
                                        attn_type='swin',
                                        padding_factor=32,
                                        attn_splits_list=self.attn_splits_list, 
                                        corr_radius_list=self.corr_radius_list, 
                                        prop_radius_list=self.prop_radius_list,
                                        save_flow=save_flow)
        
        #flow_features = self.visual_feature_propagation(flow_raw, features, token=(self.visual_cls, self.flow_cls)) # [B, C, H, W]
        #flow_features = self.flow2features(flow.permute(0,2,3,1).flatten(1,2)) # [B, L ,C]
        #flow_features = self.flow_conv(flow)[-1]
        #flow_features = flow_features.flatten(-2,-1).permute(0,2,1)
        flow_features = self.unfold(flow_raw).permute(0,2,1) # [B, num_patches, patch_size*patch_size*channels]
        flow_features = self.flow2features(flow_features)
        flow_features = torch.cat([self.visual_cls.expand(flow_features.size(0), -1, -1), flow_features], dim=1)


        ###### INERTIAL ENCODING ######
        imu = imu.unfold(1, 11, 10).flatten(0,1) # [B x seq, 6, 11]
        imu = self.imu_feature_extract(imu) # [B x seq, 7, C]
        for layer in self.imu_layers:
            imu = layer(imu, imu)  # [B x seq, 7, C]
        # imu = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
        # imu = self.inertial_encoder(imu) # [B, seq_len, C]
        # imu = imu.unsqueeze(1)

        ##### FUSION #####
        flow_len, imu_len = flow_features.size(1), imu.size(1)
        fusion_features = torch.cat([flow_features, imu], dim=1)
        for layer in self.fusion_layers:
            fusion_features = layer(fusion_features, fusion_features)
            
        ##### BOTTLENECK FUSION #####
        #flow_features = flow_features.permute(0,2,3,1).view(batch_size, -1, self.d_model)
        # b = imu.size(0)
        # z = torch.normal(mean=0, std=0.02, size=(b, self.num_bottleneck_tokens, self.d_model), device=imu.device)

        # for fuse_img, fuse_imu in zip(self.bottleneck_layers[::2], self.bottleneck_layers[1::2]):
        #     f1 = torch.cat([z, flow_features], dim=1)
        #     f2 = torch.cat([z, imu], dim=1)
        #     z1, flow_features = torch.split(fuse_img(f1, f1), [self.num_bottleneck_tokens, flow_features.size(1)], dim=1)
        #     z2, imu = torch.split(fuse_imu(f2, f2), [self.num_bottleneck_tokens, imu.size(1)], dim=1)
        #     z = torch.stack([z1, z2]).mean(dim=0)
            # Alternative method using cross attention only for fusion.
            # flow_features = fuse_img(imu, flow_features)
            # imu = fuse_imu(flow_features, imu)

        ##### AGGREGATING FUSED FEATURES #####
        flow_cls = flow_features[:, 0, :]
        #imu_cls = imu[:, 0, :]
        # flow_features = flow_features.permute(0, 2, 1)
        # imu = imu.permute(0, 2, 1)

        # flow_cls = F.avg_pool1d(flow_features, flow_features.size(2)).squeeze(2)
        # imu_cls = F.avg_pool1d(imu, imu.size(2)).squeeze(2)

        #x1 = self.ff1(flow_cls)
        #x2 = self.ff1(imu_cls)

        #combine = F.gelu(torch.stack([x1, x2]).mean(dim=0))
        combine = F.gelu(self.ff1(flow_cls))

        return combine, imu, flow_features