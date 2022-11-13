import os
import json
import random
from PIL import ImageFile, Image
import torch
import os.path as osp

from ._util import download as download_data, check_exits
from .keypoint_dataset import Hand21KeypointDataset
from .util import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Hand3DStudio(Hand21KeypointDataset):
    
    def __init__(self, root, split='train', task='noobject', download=False, **kwargs):
        assert split in ['train', 'test', 'all']
        self.split = split
        assert task in ['noobject', 'object', 'all']
        self.task = task

        

        root = osp.join(root, "H3D_crop")
        # load labels
        annotation_file = os.path.join(root, 'annotation.json')
        print("loading from {}".format(annotation_file))
        with open(annotation_file) as f:
            samples = list(json.load(f))
        if task == 'noobject':
            samples = [sample for sample in samples if int(sample['without_object']) == 1]
        elif task == 'object':
            samples = [sample for sample in samples if int(sample['without_object']) == 0]

        random.seed(42)
        random.shuffle(samples)
        samples_len = len(samples)
        samples_split = min(int(samples_len * 0.2), 3200)
        if split == 'train':
            samples = samples[samples_split:]
        elif split == 'test':
            samples = samples[:samples_split]

        super(Hand3DStudio, self).__init__(root, samples, **kwargs)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample['name']
        image_path = os.path.join(self.root, image_name)
        image = Image.open(image_path)
        keypoint3d_camera = np.array(sample['keypoint3d'])  # NUM_KEYPOINTS x 3
        keypoint2d = np.array(sample['keypoint2d'])  # NUM_KEYPOINTS x 2
        intrinsic_matrix = np.array(sample['intrinsic_matrix'])
        Zc = keypoint3d_camera[:, 2]

        image, data = self.transforms(image, keypoint2d=keypoint2d, intrinsic_matrix=intrinsic_matrix)
        keypoint2d = data['keypoint2d']
        intrinsic_matrix = data['intrinsic_matrix']
        keypoint3d_camera = keypoint2d_to_3d(keypoint2d, intrinsic_matrix, Zc)

        # noramlize 2D pose:
        visible = np.ones((self.num_keypoints, ), dtype=np.float32)
        visible = visible[:, np.newaxis]
        # 2D heatmap
        target, target_weight = generate_target(keypoint2d, visible, self.heatmap_size, self.sigma, self.image_size)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        # normalize 3D pose:
        # put middle finger metacarpophalangeal (MCP) joint in the center of the coordinate system
        # and make distance between wrist and middle finger MCP joint to be of length 1
        keypoint3d_n = keypoint3d_camera - keypoint3d_camera[9:10, :]
        keypoint3d_n = keypoint3d_n / np.sqrt(np.sum(keypoint3d_n[0, :] ** 2))

        meta = {
            'image': image_name,
            'keypoint2d': keypoint2d,  # （NUM_KEYPOINTS x 2）
            'keypoint3d': keypoint3d_n,  # （NUM_KEYPOINTS x 3）
        }
        return image, target, target_weight, meta


class Hand3DStudioAll(Hand3DStudio):
   
    def __init__(self,  root, task='all', **kwargs):
        super(Hand3DStudioAll, self).__init__(root, task=task, **kwargs)
