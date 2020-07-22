from .kitti import KittiDataset
from .nuscenes import NuScenesDataset

dataset_factory = {
    "KITTI": KittiDataset,
    "NUSC": NuScenesDataset,
}

def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
