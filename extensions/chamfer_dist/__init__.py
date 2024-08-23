# -*- coding: utf-8 -*-
# @Author: Thibault GROUEIX
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-18 15:06:25
# @Email:  cshzxie@gmail.com

import torch

import chamfer


class ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


class ChamferDistanceL2(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1) + torch.mean(dist2)

class ChamferDistanceL2_split(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1), torch.mean(dist2)

class ChamferDistanceL1(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()
        dist1 = torch.sqrt(dist1)
        dist2 = torch.sqrt(dist2)
        return (torch.mean(dist1) + torch.mean(dist2))/2

class ChamferDistanceL1_split(torch.nn.Module):
    f''' Chamder Distance L1 Split
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()
        dist1 = torch.sqrt(dist1)
        dist2 = 2*torch.sqrt(dist2)
        return (torch.mean(dist1) + torch.mean(dist2))/2
    
class InfoCDL1(torch.nn.Module):
    '''
    InfoCD L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()
        d1 = torch.sqrt(dist1)
        d2 = torch.sqrt(dist2)
        dd1 = torch.exp(-0.5 * d1)
        dd2 = torch.exp(-0.5 * d2)
        distances1 = - torch.log( dd1/ (torch.sum(dd1 + 1e-7,dim=-1).unsqueeze(-1)**1e-7))
        distances2 = - torch.log( dd2/ (torch.sum(dd2 + 1e-7,dim=-1).unsqueeze(-1)**1e-7))
        return (torch.mean(distances1) + torch.mean(distances2)) / 2


class InfoCDL2(torch.nn.Module):
    '''
    InfoCD L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        d1, d2 = ChamferFunction.apply(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()
        #d1 = torch.sqrt(dist1)
        #d2 = torch.sqrt(dist2)
        distances1 = - torch.log(torch.exp(-0.5 * d1)/(torch.sum(torch.exp(-0.5 * d1) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)
        distances2 = - torch.log(torch.exp(-0.5 * d2)/(torch.sum(torch.exp(-0.5 * d2) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)
        return (torch.mean(distances1) + torch.mean(distances2)) / 2
    
class HyperCDL1(torch.nn.Module):
    '''
    HyperCDL1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    # distances = distances.clamp(-1 + eps, 1 - eps)
    def arcosh(self, x, eps=1e-5):  # pragma: no cover
        # x = x.clamp(-1 + eps, 1 - eps)
        # x = x.clamp(1,)
        x = torch.clamp(x, min=1 + eps)
        return torch.log(x + torch.sqrt(1 + x) * torch.sqrt(x - 1))

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()
        d1 = torch.sqrt(dist1)
        d2 = torch.sqrt(dist2)

        dd1 = self.arcosh(1+ 1 * d1)
        dd2 = self.arcosh(1+ 1 * d2)

        return (torch.mean(dd1) + torch.mean(dd2)) / 2

    
class HyperCDL2(torch.nn.Module):
    '''
    HyperCDL2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    # distances = distances.clamp(-1 + eps, 1 - eps)
    def arcosh(self, x, eps=1e-5):  # pragma: no cover
        # x = x.clamp(-1 + eps, 1 - eps)
        # x = x.clamp(1,)
        x = torch.clamp(x, min=1 + eps)
        return torch.log(x + torch.sqrt(1 + x) * torch.sqrt(x - 1))

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        d1, d2 = ChamferFunction.apply(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()

        dd1 = self.arcosh(1+ 1 * d1)
        dd2 = self.arcosh(1+ 1 * d2)

        return (torch.mean(dd1) + torch.mean(dd2)) 
