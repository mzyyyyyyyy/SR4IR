from .multiscale_dataset import SameSizeBatchDataset, LazyMultiScaleDataset
from .data_transform import get_multiscale_transforms

import torch
from torch.utils.data import DataLoader, random_split
import json
import numpy as np

def load_reg_data(config):

    # Convert patch sizes to tuples
    patch_sizes = [(size, size) for size in config['dataset']['patch_sizes']]
    
    # Create datasets
    if config['dataset']['efficient_batching']:
        print('Using efficient batching')
        base_dataset = LazyMultiScaleDataset(
            input_dir=config['dataset']['input_dir'], 
            output_dir=config['dataset']['output_dir'],
            selected_bands=config['dataset']['selected_bands'],
            selected_bands_hr=config['dataset']['selected_bands_hr'],
            file_type_input=config['dataset']['file_type_input'],
            file_type_output= config['dataset']['file_type_output'],
            patch_sizes=patch_sizes,
            patches_per_tile=config['dataset']['num_patches_per_tile'], 
            normalize_target=config['dataset']['normalize_target'],
            year=config['dataset']['year'],
            selected_percentile=config['dataset']['selected_percentile'],
            patch_coord_path = config['dataset']['patch_coord_path'],
            tile_emphasis = config['dataset']['tile_emphasis'],
            nodata_cleaning = config['dataset']['nodata_cleaning'],
            correlation_cleaning = config['dataset']['correlation_cleaning'],
            input_dir_hr = config['dataset']['input_dir_hr'],
            target_range_edges = config['dataset']['target_range_edges'],
            num_patches_per_target_range = config['dataset']['num_patches_per_target_range']
        )

        total_size = len(base_dataset)
        train_size = int(config['dataset']['train_split'] * total_size)
        val_size = int(config['dataset']['val_split'] * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            base_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(config['dataset']['split_seed'])
        )
        
        train_indices = set(train_dataset.indices)
        val_indices = set(val_dataset.indices)
        test_indices = set(test_dataset.indices)

        assert len(train_indices.intersection(val_indices)) == 0
        assert len(train_indices.intersection(test_indices)) == 0
        assert len(val_indices.intersection(test_indices)) == 0

        input_channels = base_dataset.input_channels
        in_out_scale_factor = base_dataset.in_out_scale_factor # width, input/output
        out_in_scale_factor = base_dataset.out_in_scale_factor # width, output/input
        # in_out_scale_factor = round(in_out_scale_factor) if (in_out_scale_factor>1) else in_out_scale_factor # for planet

        input_channels_hr = base_dataset.input_channels_hr

    else:
       raise ValueError('Crop-first batching is not supported yet')

    # collect global norm statistics if using global norm
    if config['dataset']['use_global_norm'] or config['dataset']['use_global_minmax']:
        with open(config['dataset']['globalnorm_stats_file'], "r") as f:
            all_stats = json.load(f)
            try:
                # ipdb.set_trace()
                gb_stats = all_stats[str(config['dataset']['year'])]
                selected_bands = config['dataset']['selected_bands']
                gb_stats = {k: [v[i] for i in selected_bands] for k, v in gb_stats.items()}
            except:
                print(f'Can not load input statistics for year {config["dataset"]["year"]}')
    else:
        gb_stats = None

    # collect global norm statistics for HR image if using global norm
    if config['dataset']['use_global_norm_hr'] or config['dataset']['use_global_minmax_hr']:
        with open(config['dataset']['globalnorm_stats_file_hr'], "r") as f:
            all_stats = json.load(f)
            try:
                # ipdb.set_trace()
                gb_stats_hr = all_stats[str(config['dataset']['year'])]
                selected_bands_hr = config['dataset']['selected_bands_hr']
                gb_stats_hr = {k: [v[i] for i in selected_bands_hr] for k, v in gb_stats_hr.items()}
            except:
                print(f'Can not load input statistics for year {config["dataset"]["year"]}')
    else:
        gb_stats_hr = None   

    # collect target norm statistics if using target normalization
    if config['dataset']['normalize_target'] or config['dataset']['minmax_target']:
        with open(config['dataset']['targetnorm_stats_file'], "r") as f:
            all_tn_stats = json.load(f)
            try:
                tn_stats = all_tn_stats[str(config['dataset']['year'])]
            except:
                print(f'Can not load target statistics for year {config["dataset"]["year"]}')
    else:
        tn_stats = None


    train_input_transform = get_multiscale_transforms( # 这个函数的作用是根据配置参数创建多尺度数据变换操作的组合。即，train_input_transform是包含多个操作的函数
        patch_sizes, num_channels=input_channels, is_training=True, is_output=False,
        use_global_minmax= config['dataset']['use_global_minmax'],
        use_global_norm=config['dataset']['use_global_norm'],
        use_local_norm = config['dataset']['use_local_norm'],
        globalnorm_stats=gb_stats
    )
    train_input_transform_hr = get_multiscale_transforms(
        patch_sizes, num_channels=input_channels_hr, is_training=True, is_output=False,
        use_global_minmax= config['dataset']['use_global_minmax'],
        use_global_norm=config['dataset']['use_global_norm'],
        use_local_norm = config['dataset']['use_local_norm'],
        globalnorm_stats=gb_stats_hr
    )
    train_output_transform = get_multiscale_transforms(
        patch_sizes, num_channels=input_channels, is_training=True, is_output=True, 
        normalize_target=config['dataset']['normalize_target'],
        minmax_target=config['dataset']['minmax_target'],
        unit_scale_ratio=config['dataset']['unit_scale_ratio'],
        target_mean = base_dataset.target_mean if config['dataset']['normalize_target'] else None,
        target_std = base_dataset.target_std if config['dataset']['normalize_target'] else None,
        targetnorm_stats= tn_stats
        # use_shift_augmentation=config['data']['use_shift_augmentation'],
        # shift_limit=config['data']['shift_limit'],
        # shift_augmentation_p=config['data']['shift_augmentation_p']
    )
    val_input_transform = get_multiscale_transforms(
        patch_sizes, num_channels=input_channels, is_training=False, is_output=False,
        use_global_minmax=config['dataset']['use_global_minmax'],
        use_global_norm=config['dataset']['use_global_norm'],
        use_local_norm=config['dataset']['use_local_norm'],
        globalnorm_stats=gb_stats
    )
    val_input_transform_hr = get_multiscale_transforms(
        patch_sizes, num_channels=input_channels_hr, is_training=False, is_output=False,
        use_global_minmax= config['dataset']['use_global_minmax'],
        use_global_norm=config['dataset']['use_global_norm'],
        use_local_norm = config['dataset']['use_local_norm'],
        globalnorm_stats=gb_stats_hr
    )
    val_output_transform = get_multiscale_transforms(
        patch_sizes, num_channels=input_channels, is_training=False, is_output=True,
        normalize_target=config['dataset']['normalize_target'],
        minmax_target=config['dataset']['minmax_target'],
        unit_scale_ratio=config['dataset']['unit_scale_ratio'],
        target_mean = base_dataset.target_mean if config['dataset']['normalize_target'] else None,
        target_std = base_dataset.target_std if config['dataset']['normalize_target'] else None,
        targetnorm_stats= tn_stats
    )
    test_input_transform = get_multiscale_transforms(
        patch_sizes, num_channels=input_channels, is_training=False, is_output=False,
        use_global_minmax=config['dataset']['use_global_minmax'],
        use_global_norm=config['dataset']['use_global_norm'],
        use_local_norm=config['dataset']['use_local_norm'],
        globalnorm_stats=gb_stats
    )
    test_input_transform_hr = get_multiscale_transforms(
        patch_sizes, num_channels=input_channels_hr, is_training=False, is_output=False,
        use_global_minmax=config['dataset']['use_global_minmax'],
        use_global_norm=config['dataset']['use_global_norm'],
        use_local_norm=config['dataset']['use_local_norm'],
        globalnorm_stats=gb_stats_hr
    )
    test_output_transform = get_multiscale_transforms(
        patch_sizes, num_channels=input_channels, is_training=False, is_output=True,
        normalize_target=config['dataset']['normalize_target'],
        minmax_target=config['dataset']['minmax_target'],
        unit_scale_ratio=config['dataset']['unit_scale_ratio'],
        target_mean = base_dataset.target_mean if config['dataset']['normalize_target'] else None,
        target_std = base_dataset.target_std if config['dataset']['normalize_target'] else None,
        targetnorm_stats= tn_stats
    )

    print('Warning: target mean and std are calculated on the whole dataset (after filtering), not excluding test set!')
    

    # Create efficient batching datasets
    train_dataset = SameSizeBatchDataset(
        train_dataset, 
        patch_sizes, 
        config['train']['batch_size'],
        transform_input=train_input_transform,
        transform_input_hr=train_input_transform_hr,
        transform_output=train_output_transform,
        lons=np.array(config['dataset']['tile_lon_range']),
        lats=np.array(config['dataset']['tile_lat_range'])
    )
    
    val_dataset = SameSizeBatchDataset(
        val_dataset, 
        patch_sizes, 
        config['train']['batch_size'],
        transform_input=val_input_transform,
        transform_input_hr=val_input_transform_hr,
        transform_output=val_output_transform,
        lons=np.array(config['dataset']['tile_lon_range']),
        lats=np.array(config['dataset']['tile_lat_range'])
    )
    
    test_dataset = SameSizeBatchDataset(
        test_dataset, 
        patch_sizes, 
        config['train']['batch_size'],
        transform_input=test_input_transform,
        transform_input_hr=test_input_transform_hr,
        transform_output=test_output_transform,
        lons=np.array(config['dataset']['tile_lon_range']),
        lats=np.array(config['dataset']['tile_lat_range'])
    )
    
    train_sampler = None
    val_sampler = None
    test_sampler = None
    
    # Create data loaders
    num_workers = config['dataset']['workers']
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # SameSizeBatchDataset already returns batches, iter(dataloader) will reture (1, bs, dimension)
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # SameSizeBatchDataset already returns batches
        shuffle=(val_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        sampler=val_sampler
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # SameSizeBatchDataset already returns batches
        shuffle=(test_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        sampler=test_sampler
    )

    print(f'Total patches: {len(base_dataset)}')
    print(f'Train patches (len(base_dataset) // batch_size): {len(train_dataset)}')  # len(base_dataset) // batch_size
    print(f'Val patches (len(base_dataset) // batch_size): {len(val_dataset)}')


    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler