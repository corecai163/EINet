#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Pingping Cai

import torch
import torch.nn as nn
import math
from extensions.chamfer_dist import ChamferDistanceL1,ChamferDistanceL2

from models.pointnet import PointNet_SA_Module_KNN,Transformer,Interpolation,Extrapolation,fps_subsample


from .build import MODELS

class FeaExtract(nn.Module):
    def __init__(self, dim_feat=384, num_seeds = 512, seed_fea=256):
        '''
        Extract information from partial point cloud
        '''
        super(FeaExtract, self).__init__()
        self.num_seed = num_seeds
        self.sa_module_1 = PointNet_SA_Module_KNN(num_seeds, 16, 3, [32, seed_fea], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(seed_fea, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(num_seeds//4, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = Transformer(256, dim=64)
        #self.sa_module_3 = PointNet_SA_Module_KNN(32, 16, 256, [256, 384], group_all=False, if_bn=False, if_idx=True)
        #self.transformer_3 = Transformer(384, dim=64)
        self.sa_module_4 = PointNet_SA_Module_KNN(None, None, 256, [512, dim_feat], group_all=True, if_bn=False)
        
        #self.expolate1 = Extrapolation(256, dim=64)
  

    def forward(self, point_cloud):
        """
        Args:
             point_cloud: b, 3, n
        """
        
        l0_xyz = point_cloud
        l0_fea = point_cloud

        ## Encoder        
        l1_xyz, l1_fea, idx1 = self.sa_module_1(l0_xyz, l0_fea)  # (B, 3, 512), (B, 128, 512)
        l1_fea = self.transformer_1(l1_fea, l1_xyz)
        l2_xyz, l2_fea, idx2 = self.sa_module_2(l1_xyz, l1_fea)  # (B, 3, 128), (B, 256, 128)
        l2_fea = self.transformer_2(l2_fea, l2_xyz)
        #l3_xyz, l3_points, idx3 = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 32), (B, 384, 32)
        #l3_points = self.transformer_3(l3_points, l3_xyz)
        l4_xyz, l4_fea = self.sa_module_4(l2_xyz, l2_fea)  # (B, 3, 1), (B, out_dim, 1)
        
        
        return l4_fea,l2_xyz,l2_fea,l1_xyz,l1_fea

class UPProcess(nn.Module):
    def __init__(self, embed_dim=256, upscale=[1,2,8], scale=[0.4,0.3,0.2]):
        super(UPProcess, self).__init__()

        up_layers = []
        for i, factor in enumerate(upscale):
            up_layers.append(Interpolation(in_channel=embed_dim, up_factor=factor, scale=scale[i],n_knn=20))

        self.up_layers = nn.ModuleList(up_layers)

    def forward(self, seed, fea, glo_fea):
        """
        Args:
            global_shape_fea: Tensor, (b, dim_feat, 1)
            pcd: Tensor, (b, n,3)
        """
        pred_pcds = [seed]
        constrain = []
        # Upsample layers
        K_prev = fea#self.mlp_1(pcd1)
        pcd = seed.permute(0, 2, 1).contiguous() # (B, 3, 256)

        for layer in self.up_layers:
            #pcd, K_prev = layer(pcd, K_prev, seed, fea)
            pcd, K_prev, cons = layer(pcd, K_prev,seed,fea)
            pred_pcds.append(pcd.permute(0, 2, 1).contiguous())
            
            if cons !=None:
                constrain.append(cons)
        if constrain!=[]:
            const = torch.cat(constrain,dim=-1)
            pred_pcds.append(const)
        else:
            pred_pcds.append(None)
        return pred_pcds
    
# 3D completion
@MODELS.register_module()
class EINet(nn.Module):
    def __init__(self, config):

        super(EINet, self).__init__()
        self.num_pred = config.num_pred // config.num_seeds
        self.dim_seed_fea = config.dim_seed_fea
        
        self.feat_extractor = FeaExtract(dim_feat=config.dim_feat,num_seeds=config.num_seeds,seed_fea=config.dim_seed_fea)
        self.expolate = Extrapolation(dim_feat=config.dim_feat, in_channel=256, out_channel=config.dim_seed_fea,dim=64)
        self.decoder = UPProcess(embed_dim=config.dim_seed_fea,upscale = config.upscales, scale=config.scales)
        
        ## for shapenet dataset
        #self.loss_func = ChamferDistanceL2()

        ## for PCN dataset
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, ret, gt):
        Pc, P1, P2, P3, cons1,cons2= ret

        #gt_2 = fps_subsample(gt, P2.shape[1])
        #gt_1 = fps_subsample(gt_2, P1.shape[1])
        #gt_c = fps_subsample(gt_1, Pc.shape[1])
        cd0 = self.loss_func(Pc, gt)
        cd1 = self.loss_func(P1, gt)
        cd2 = self.loss_func(P2, gt)
        cd3 = self.loss_func(P3, gt)

        const = torch.mean(torch.square(cons1))+ torch.mean(torch.square(cons2))
        #print(Pc.size())

        loss_all = (1*cd0 + 1*cd1 + 1*cd2 + 1*cd3) * 1e3+ 10*const
        return loss_all, cd0, cd3


    def forward(self, partial_cloud):
        """
        Args:
            point_cloud: (B, N, 3)
        """

        #pcd_bnc = point_cloud
        in_pcd = partial_cloud.permute(0, 2, 1).contiguous()     

        l4_fea, l2_xyz,l2_fea,l1_xyz,l1_fea = self.feat_extractor(in_pcd)

        u0_xyz, u0_fea,cons1 = self.expolate(l2_fea,l4_fea)

        u1_fea = torch.cat([l1_fea,u0_fea],dim=2)  # (b, 256, 512)
        u1_xyz = torch.cat([l1_xyz,u0_xyz],dim=2)
        #u1_xyz = self.pos(u1_fea)

        #print(u1_fea.size())
        seed = u1_xyz.permute(0, 2, 1).contiguous() # (B, num_pc, 3)
        #pcd = fps_subsample(torch.cat([seed, partial_cloud], 1), 1024) # (B, num_p0, 3)
        #pcd = pcd.permute(0, 2, 1).contiguous() # (B, 3, num_p0)
        p0,p1,p2,p3,cons2 = self.decoder(seed, u1_fea, l4_fea)
        #new_fea = self.decoder(fea)
        #print(p3.size())

        return p0,p1,p2,p3,cons1,cons2
    

