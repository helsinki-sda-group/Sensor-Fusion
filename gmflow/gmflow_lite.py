import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer, FeatureFlowAttention
from .matching import global_correlation_softmax, local_correlation_softmax
from .geometry import flow_warp
from .utils import normalize_img, feature_add_position


class GMFlowLite(nn.Module):
    def __init__(
        self,
        opt,
        **kwargs,
    ):
        super(GMFlowLite, self).__init__()
        self.opt = opt
        self.num_scales = opt.num_scales
        self.feature_channels = opt.v_f_len
        self.upsample_factor = opt.upsample_factor
        self.attention_type = opt.attention_type
        self.num_transformer_layers = opt.num_transformer_layers

        # CNN backbone
        self.backbone = CNNEncoder(
            output_dim=opt.v_f_len, num_output_scales=opt.num_scales
        )

        # Transformer
        self.transformer = FeatureTransformer(
            num_layers=opt.num_transformer_layers,
            d_model=opt.v_f_len,
            nhead=opt.num_head,
            attention_type=opt.attention_type,
            ffn_dim_expansion=opt.ffn_dim_expansion,
        )

        # flow propagation with self-attn
        #self.feature_flow_attn = FeatureFlowAttention(in_channels=opt.v_f_len)

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(
            concat
        )  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1
    
    def load_weights(self):
        # Load state dict for GMFlow
        print("Loading pretrained weights for GMFLOW:")
        pretrained_weights = torch.load(self.opt.gmflow_weights)['model'] if self.opt.gmflow_weights else None
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in pretrained_weights and param.data.shape == pretrained_weights[name].shape:
                    print(f"Copied {name}")
                    param.copy_(pretrained_weights[name])
                else:
                    print(
                        f"""When loading weights for GMFlow. Skipped loading for {name} \n
                        {name} shape: {param.data.shape} != pretrained weights shape: {pretrained_weights[name].shape}"""
                    )

    def forward(
        self,
        img0,
        img1,
        pred_bidir_flow=False,
        **kwargs,
    ):
        results_dict = {}
        attn_splits_list = self.opt.attn_splits_list
        corr_radius_list = self.opt.corr_radius_list
        prop_radius_list = self.opt.prop_radius_list

        img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]

        # resolution low to high
        feature0_list, feature1_list = self.extract_feature(
            img0, img1
        )  # list of features

        flow = None

        assert (
            len(attn_splits_list)
            == len(corr_radius_list)
            == len(prop_radius_list)
            == self.num_scales
        )

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            if pred_bidir_flow and scale_idx > 0:
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat(
                    (feature1, feature0), dim=0
                )

            upsample_factor = self.upsample_factor * (
                2 ** (self.num_scales - 1 - scale_idx)
            )

            if scale_idx > 0:
                flow = (
                    F.interpolate(
                        flow, scale_factor=2, mode="bilinear", align_corners=True
                    )
                    * 2
                )

            if flow is not None:
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)  # [B, C, H, W]

            attn_splits = attn_splits_list[scale_idx]
            corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(
                feature0, feature1, attn_splits, self.feature_channels
            )

            # Transformer
            feature0, feature1 = self.transformer(
                feature0, feature1, attn_num_splits=attn_splits
            )

            # correlation and softmax
            if corr_radius == -1:  # global matching
                flow_pred = global_correlation_softmax(
                    feature0, feature1, pred_bidir_flow
                )[0]
            else:  # local matching
                flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[
                    0
                ]

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred
            results_dict.update({"correlation_flow": flow})

        #     flow = self.feature_flow_attn(
        #         feature0,
        #         flow.detach(),
        #         local_window_attn=prop_radius > 0,
        #         local_window_radius=prop_radius,
        #     )
        # results_dict.update({"feature_flow": flow})
        results_dict.update({"features": feature0})

        return results_dict
