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

def run_net(args, config, train_writer=None):
    logger = get_logger(args.log_name)
    # Build Dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = \
        builder.dataset_builder(args, config.dataset.train), builder.dataset_builder(args, config.dataset.val)
    # Build Model
    base_model = builder.model_builder(config.model)

    if args.use_gpu:
        base_model.to(args.local_rank)
        
    # Parameter Setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # Resume Ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)
        if not os.path.exists(args.start_ckpts):
            print_log(f'[RESUME INFO] no checkpoint file from path {args.start_ckpts}...', logger = logger)
            return 0, 0, 0
        print_log(f'[RESUME INFO] Loading optimizer from {args.start_ckpts}...', logger = logger )
        state_dict = torch.load(args.start_ckpts, map_location='cpu')
        start_epoch = state_dict['epoch'] + 1

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, \
                                                         device_ids=[args.local_rank % torch.cuda.device_count()], \
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    
    print("Number of Parameters:")
    print(sum(p.numel() for p in base_model.parameters() if p.requires_grad))

    # Optimizer & Scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)
    elif args.start_ckpts is not None:
        # if not os.path.exists(args.start_ckpts):
        #     print_log(f'[RESUME INFO] no checkpoint file from path {args.start_ckpts}...', logger = logger)
        #     return 0, 0, 0
        # print_log(f'[RESUME INFO] Loading optimizer from {args.start_ckpts}...', logger = logger )
        # state_dict = torch.load(args.start_ckpts, map_location='cpu')
        # optimizer
        optimizer.load_state_dict(state_dict['optimizer'])

    # Training

    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)



        dataset_name = config.dataset.train._base_.NAME
        npoints = config.dataset.train._base_.N_POINTS
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            
            
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI finetune

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = choice)
                partial = partial.cuda()

            elif dataset_name == 'MVP':
                gt = data[1].cuda()
                partial = data[0].cuda()
                
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1
            #print(idx)
           
            ret = base_model(partial)
            #print(ret[3].size())
            loss_pcd, sparse_loss, dense_loss = base_model.module.get_loss(ret, gt)

            _loss = loss_pcd #+ loss_cons
            _loss.backward()

            # Forward
            if num_iter == config.step_per_update:
                num_iter = 0
                nn.utils.clip_grad_norm_(base_model.parameters(), 2e-3)
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)

                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
            else:
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])


            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            # if train_writer is not None:
            #     train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
            #     train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)
            #     train_writer.add_scalar('LR/training', optimizer.param_groups[0]['lr'], n_itr)
            #     train_writer.add_scalar('Penalty/Batch/cons1', orth_cons1.item() * 1000, n_itr)
            #     train_writer.add_scalar('Penalty/Batch/cons2', orth_cons2.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                print_log('[Memory: %f GB Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (mem, epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        # if isinstance(scheduler, list):
        #     for item in scheduler:
        #         item.step(epoch)
        # else:
        #     scheduler.step(epoch)
        epoch_end_time = time.time()
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
        print_log('[Training] Memory: %f GB EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (mem, epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, train_writer, args, config, logger=logger)

            # Save checkpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 10:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger) 
        torch.cuda.empty_cache()
    
    if train_writer is not None:  
        train_writer.close()


    
def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = choice)
                partial = partial.cuda()
            elif dataset_name == 'MVP':
                gt = data[1].cuda()
                partial = data[0].cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            ret = base_model(partial)
            coarse_points = ret[0]
            dense_points = ret[3]

            sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
            dense_loss_l1 =  ChamferDisL1(dense_points, gt)
            dense_loss_l2 =  ChamferDisL2(dense_points, gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, \
                                dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

            _metrics = Metrics.get(dense_points, gt) 

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            if val_writer is not None and idx % args.val_interval == 0:
                #print(partial.size())
                # val_writer.add_mesh('partial', vertices=partial.detach().cpu().numpy())
                # val_writer.add_mesh('pc0', vertices=coarse_points.detach().cpu().numpy())
                # val_writer.add_mesh('pc3', vertices=dense_points.detach().cpu().numpy())
                # input_pc = partial.squeeze().detach().cpu().numpy()
                # input_pc = misc.get_ptcloud_img(input_pc)
                # val_writer.add_image('Model%02d-%d/Input'% (idx, epoch) , input_pc, epoch, dataformats='HWC')

                # sparse = coarse_points.squeeze().cpu().numpy()
                # sparse_img = misc.get_ptcloud_img(sparse)
                # val_writer.add_image('Model%02d-%d/Sparse' % (idx, epoch), sparse_img, epoch, dataformats='HWC')
                # pred_sparse_img = misc.get_ordered_ptcloud_img(sparse[0:224,:])
                # val_writer.add_image('Model%02d-%d/PredSparse' % (idx, epoch), pred_sparse_img, epoch, dataformats='HWC')

                # dense = dense_points.squeeze().cpu().numpy()
                # dense_img = misc.get_ptcloud_img(dense)
                # val_writer.add_image('Model%02d-%d/Dense' % (idx, epoch), dense_img, epoch, dataformats='HWC')
                
                # gt_ptcloud = gt.squeeze().cpu().numpy()
                # gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
                # val_writer.add_image('Model%02d-%d/DenseGT' % (idx, epoch), gt_ptcloud_img, epoch, dataformats='HWC')
        
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        if val_writer is not None:
            #print(partial.size())
            val_writer.add_mesh('partial', vertices=partial.detach().cpu().numpy())
            val_writer.add_mesh('pc0', vertices=coarse_points.detach().cpu().numpy())
            val_writer.add_mesh('pc3', vertices=dense_points.detach().cpu().numpy())
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
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
        msg += (str(taxonomy_id) + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        if taxonomy_id in shapenet_dict:
            msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Test/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Test/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())

