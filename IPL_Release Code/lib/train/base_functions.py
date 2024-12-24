import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import LasHeR_trainingSet,RGBT234,LasHeR_testingSet,VTUAV
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process
import timm.scheduler


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader, attr=None):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        #print('name',name)
        assert name in ['LasHeR_testingSet','RGBT234',"LasHeR_trainingSet",'LasHeR',"LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "COCO17", "VID", "TRACKINGNET","VTUAV"]
        if name == "LasHeR_trainingSet" or name == 'LasHeR':
            datasets.append(LasHeR_trainingSet(settings.env.lasher_trainingset_dir, split='train', image_loader=image_loader, attr=attr))
                
        if name == "VTUAV":
            datasets.append(VTUAV(settings.env.vtuav_dir, split='train', image_loader=image_loader))
                                
        if name == "RGBT234":
            datasets.append(RGBT234(settings.env.rgbt234_dir, image_loader=image_loader))
        if name == "LasHeR_testingSet":
            datasets.append(LasHeR_testingSet(settings.env.lasher_testingset_dir, split='val', image_loader=image_loader, attr=attr))

    return datasets



def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.STARKProcessing_RGBT(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    data_processing_val = processing.STARKProcessing_RGBT(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    print("sampler_mode", sampler_mode)
    attr = getattr(cfg.DATA.TRAIN, 'CHALLENGE', None)
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader, attr=attr),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders
    attr = getattr(cfg.DATA.VAL, 'CHALLENGE', None)
    dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader, attr=attr),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_cls)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    param_key = getattr(cfg.TRAIN, "PARAM_KEY", False)
    if train_cls:
        print("Only training classification head. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "cls" in n and p.requires_grad]}
        ]

        for n, p in net.named_parameters():
            if "cls" not in n:
                p.requires_grad = False
            else:
                print(n)
    elif param_key:
        print("Learnable parameters are shown below.")
        if isinstance(param_key, str):
            fintune_weight = lambda name: 1 if (param_key=='>all' or param_key in name) else 0
            param_dicts = [
                {"params": [p for n, p in net.named_parameters() if fintune_weight(n) and p.requires_grad]}
            ]

            for n, p in net.named_parameters():
                if not fintune_weight(n):
                    p.requires_grad = False
                else:
                    print(n)

        elif isinstance(param_key, list):
            # 是否是参与训练的参数
            fintune_weight = lambda name: sum([1 for k,l in param_key if (k in name or k=='>all')])>0
            # 是否是某个参数集合
            fintune_weight_p = lambda name: sum([1 for k,l in param_key if k in name])>0
            fintune_weight_key = lambda name,key: 1 if (key in name or (key=='>all' and not fintune_weight_p(name))) else 0
            param_dicts = []
            for k,lr in param_key:
                param_dicts.append(
                    {
                        "params": [p for n, p in net.named_parameters() if fintune_weight_key(n,k) and p.requires_grad],
                        "lr": lr,
                    }
                )

            for n, p in net.named_parameters():
                if not fintune_weight(n):
                    p.requires_grad = False
                else:
                    print(n)

    #ostrack_manet_train
    # else:
    #     param_dicts = [
    #         {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
    #         {
    #             "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
    #             "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
    #         },
    #     ]
    #     if is_main_process():
    #         print("Learnable parameters are shown below.")
    #         for n, p in net.named_parameters():
    #             if p.requires_grad:
    #                 print(n)

    #ostrack_manet_prompt_train
    else:
        # stage2 train adapter and patch embeding
        for n, p in net.named_parameters():
            if 'backbone' in n or 'box_head' in n:
                p.requires_grad = False
            if 'patch_embed_' in n or 'align_adapter' in n:
                p.requires_grad = True

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "patch_embed_" in n and p.requires_grad],
             "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
             },
            {"params": [p for n, p in net.named_parameters() if "align_adapter" in n and p.requires_grad],
             "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
             },
        ]
        
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)
        
        ## stage3 train cls head
        # for n, p in net.named_parameters():
        #     if 'backbone' in n or 'box_head' in n:
        #         p.requires_grad = False
        #     if 'box_head' in n:
        #         p.requires_grad = True
        # param_dicts = [
        #     {"params": [p for n, p in net.named_parameters() if "box_head" in n and p.requires_grad]},

        # ]
        # if is_main_process():
        #     print("Learnable parameters are shown below.")
        #     for n, p in net.named_parameters():
        #         if p.requires_grad:
        #             print(n)
        
        


    # 选择优化器
    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    elif cfg.TRAIN.SCHEDULER.TYPE == "CosineLR":        # warm up 加 余弦退火   
        lr_scheduler = timm.scheduler.CosineLRScheduler(optimizer=optimizer,
                                                        t_initial=cfg.TRAIN.EPOCH+1,    # 加1保证最后一个epoch学习率不为0
                                                        lr_min=cfg.TRAIN.SCHEDULER.LR_MIN,
                                                        warmup_t=cfg.TRAIN.SCHEDULER.WARMUP_T,
                                                        warmup_lr_init=cfg.TRAIN.SCHEDULER.WARMUP_LR_INIT)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler


