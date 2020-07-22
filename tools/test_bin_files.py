import sys

import numpy as np
import open3d as o3d # need to be imported befor torch
import torch
from loguru import logger as logging

from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.models import build_detector
from det3d.torchie import Config
from tools.demo_utils import visual_detection


class CenterPointDector(object):
    def __init__(self, config_file, model_file, calib_data=None):
        self.config_file = config_file
        self.model_file = model_file
        self.calib_data = calib_data
        self.points = None
        self.inputs = None
        self._init_model()

    def _init_model(self):
        cfg = Config.fromfile(self.config_file)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        self.net.load_state_dict(torch.load(self.model_file)["state_dict"])
        self.net = self.net.to(self.device).eval()

        self.voxel_generator = VoxelGenerator(
            voxel_size=cfg.voxel_generator.voxel_size,
            point_cloud_range=cfg.voxel_generator.range,
            max_num_points=cfg.voxel_generator.max_points_in_voxel,
            max_voxels=cfg.voxel_generator.max_voxel_num,
        )

    @staticmethod
    def load_cloud_from_nuscenes_file(pc_f):
        logging.info('loading cloud from: {}'.format(pc_f))
        num_features = 5
        cloud = np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, num_features])
        # last dimension should be the timestamp.
        print(cloud[:,4])

        cloud[:, 4] = 0
        return cloud

    @staticmethod
    def load_cloud_from_deecamp_file(pc_f):
        logging.info('loading cloud from: {}'.format(pc_f))
        num_features = 4
        cloud = np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, num_features])
        # last dimension should be the timestamp.
        cloud = np.hstack((cloud, np.zeros([cloud.shape[0], 1])))
        return cloud

    def predict_on_nuscenes_local_file(self, cloud_file,save_path=None):

        # load sample from file
        # self.points = self.load_cloud_from_nuscenes_file(cloud_file)
        self.points = self.load_cloud_from_deecamp_file(cloud_file)

        # prepare input
        voxels, coords, num_points = self.voxel_generator.generate(self.points)
        # voxels shape: [num_voxel, max_num_points_per_voxel, feature_dim]
        # coords shape: [num_voxel,3]
        # num_points shape [num_voxel]
        # TODO: 为什么voxel的shape第一维不是grid_size的数量？
        # 原因——去除无效的在定义边界范围外的格点，创造稀疏的条件~
        print(voxels.shape) #(5268, 20, 5)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        grid_size = self.voxel_generator.grid_size
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)

        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=self.device)

        self.inputs = dict(
            voxels=voxels,
            num_points=num_points,
            num_voxels=num_voxels,
            coordinates=coords,
            shape=[grid_size]  # ?
        )

        # predict
        with torch.no_grad():
            outputs = self.net(self.inputs, return_loss=False)[0]

        for k, v in outputs.items():
            if k not in [
                "metadata",
            ]:
                outputs[k] = v.to('cpu')

        # visualization
        visual_detection(np.transpose(self.points), outputs, eval_range=35, conf_th=0.4, show_plot=False, show_3D=False, save_path=save_path)

import os
import glob
if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     logging.error('Input a lidar bin file pls.')
    # else:
    #  usage:
    #  编辑项目路径，配置文件路径，模型参数路径，测试数据路径，输出文件路径！
    project_folder = r'D:\program\detection\3d-detection/CenterPoint'
    config_file = 'configs/centerpoint/kitti_centerpoint_pp_02voxel_circle_nms_demo.py'
    config_file = os.path.join(project_folder,config_file)
    # model_file = 'work_dirs/centerpoint_pillar_512_demo/last.pth'
    model_file = 'work_dirs/ori_config_pp/epoch_20.pth'

    model_file=os.path.join(project_folder,model_file)
    detector = CenterPointDector(config_file, model_file)

    # detector.predict_on_nuscenes_local_file(sys.argv[1])
    folder = r'I:\data\3D-detection\test_video_filter_01'
    # folder = r'I:\data\3D-detection\train_val_filter'

    # folder = r'I:\data\3D-detection\nuScenes\v1.0-mini\samples\LIDAR_TOP'
    # folder = r'I:\data\3D-detection\train_val_filter'

    paths = glob.glob(os.path.join(folder,'*.bin'))
    print(paths)
    save_folder = r'demo_test_video/'
    import os
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    for bin_path in paths:
        # bin_path = os.path.join(folder,'011001.bin')
        basename = os.path.basename(bin_path).replace('bin','png')
        save_path = os.path.join(save_folder, basename)
        detector.predict_on_nuscenes_local_file(
            bin_path, save_path
        )
