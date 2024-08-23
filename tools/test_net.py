import numpy as np
import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    # test_dataloader = torch.utils.data.DataLoader(config.dataset.test, batch_size=1,
    #                                     shuffle = False, 
    #                                     drop_last = False,
    #                                     num_workers = int(args.num_workers),
    #                                     pin_memory=True)
 
    base_model = builder.model_builder(config.model)
    print_log(base_model, logger = logger)
    
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    #state_dict = torch.load(args.ckpts, map_location='cpu')
    #base_model.load_state_dict(state_dict)
    if args.use_gpu:
        base_model.to(args.local_rank)
    if args.sync_bn:
        base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)

    #  DDP    
    base_model = nn.parallel.DistributedDataParallel(base_model, \
                                                        device_ids=[args.local_rank % torch.cuda.device_count()], \
                                                        find_unused_parameters=True)

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
                
                
                ret = base_model(partial)
                coarse_points = ret[0]
                up1=ret[1]
                up2=ret[2]
                dense_points = ret[3]
                                         
                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, \
                                    dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points ,gt)
                test_metrics.update(_metrics)


                if False:
                    path = 'pcn_outputs/'+str(idx)
                    save_pcd(path,dense_points.squeeze().cpu().numpy())
                    path1 = 'pcn_inputs/'+str(idx)
                    save_pcd(path1,partial.squeeze().cpu().numpy())
                    path2 = 'pcn_outputs_0/'+str(idx)
                    save_pcd(path2,coarse_points.squeeze().cpu().numpy())
                    path3 = 'pcn_outputs_1/'+str(idx)
                    save_pcd(path3,up1.squeeze().cpu().numpy())
                    path4 = 'pcn_outputs_2/'+str(idx)
                    save_pcd(path4,up2.squeeze().cpu().numpy())

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                for item in choice:           
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    # NOTE: subsample the input
                    partial = misc.fps(partial, 2048)
                    ret = base_model(partial)
                    coarse_points = ret[0]
                    dense_points = ret[3]

                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                    dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                    # sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                    # sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                    # dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                    # dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, \
                                        dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                    _metrics = Metrics.get(dense_points ,gt)

                    #test_metrics.update(_metrics)

                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)
                    
            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )
                continue
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            
            # Visualize
            if idx % args.test_interval == 0:                
                # input_pc = partial.squeeze().detach().cpu().numpy()
                # input_img = misc.get_ptcloud_img(input_pc)
                # test_writer.add_image('Model%02d-test/Input'% idx , input_img, dataformats='HWC')

                # sparse = coarse_points.squeeze().cpu().numpy()
                # sparse_img = misc.get_ptcloud_img(sparse)
                # test_writer.add_image('Model%02d-test/Sparse' % idx, sparse_img, dataformats='HWC')

                # dense = dense_points.squeeze().cpu().numpy()
                # dense_img = misc.get_ptcloud_img(dense)
                # test_writer.add_image('Model%02d-test/Dense' % idx, dense_img, dataformats='HWC')
                
                # gt_ptcloud = gt.squeeze().cpu().numpy()
                # gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
                # test_writer.add_image('Model%02d-test/DenseGT' % idx, gt_ptcloud_img, dataformats='HWC')
                    
                # Save output results
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
                
        # Compute testing results
        if dataset_name == 'KITTI':
            return
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

     

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)


    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    return 
