torchrun --standalone --nproc_per_node=4 train.py --sync_bn --config ./cfgs/PCN_models/EINet.yaml --exp_name PCN_EINet --val_freq 4 --val_interval 50
#torchrun --standalone --nproc_per_node=4 train.py --sync_bn --config ./cfgs/ShapeNet55_models/EINet.yaml --exp_name shape55_EINet --num_workers 8 --val_freq 5 --val_interval 100 
