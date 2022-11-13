import os
import json
from PIL import ImageFile
import torch
from .transforms import *
from .util import *
from ._util import download as download_data, check_exits
from .keypoint_dataset import Body16KeypointDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SURREAL(Body16KeypointDataset):
    
    def __init__(self, root, split='train', task='all', download=False, **kwargs):
        assert split in ['train', 'test', 'val']
        self.split = split

        
        all_samples = []
        for part in [0, 1, 2]:
            annotation_file = os.path.join(root, split, 'run{}.json'.format(part))
            print("loading", annotation_file)
            with open(annotation_file) as f:
                samples = json.load(f)
                for sample in samples:
                    sample["image_path"] = os.path.join(root, self.split, 'run{}'.format(part), sample['name'])
                all_samples.extend(samples)

        random.seed(42)
        random.shuffle(all_samples)
        samples_len = len(all_samples)
        samples_split = min(int(samples_len * 0.2), 3200)
        if self.split == 'train':
            all_samples = all_samples[samples_split:]
        elif self.split == 'test':
            all_samples = all_samples[:samples_split]
        self.joints_index = (7, 4, 1, 2, 5, 8, 0, 9, 12, 15, 20, 18, 13, 14, 19, 21)

        super(SURREAL, self).__init__(root, all_samples, **kwargs)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample['name']

        image_path = sample['image_path']
        image = Image.open(image_path)
        keypoint3d_camera = np.array(sample['keypoint3d'])[self.joints_index, :]  # NUM_KEYPOINTS x 3
        keypoint2d = np.array(sample['keypoint2d'])[self.joints_index, :]  # NUM_KEYPOINTS x 2
        intrinsic_matrix = np.array(sample['intrinsic_matrix'])
        Zc = keypoint3d_camera[:, 2]

        image, data = self.transforms(image, keypoint2d=keypoint2d, intrinsic_matrix=intrinsic_matrix)
        keypoint2d = data['keypoint2d']
        intrinsic_matrix = data['intrinsic_matrix']
        keypoint3d_camera = keypoint2d_to_3d(keypoint2d, intrinsic_matrix, Zc)

        # noramlize 2D pose:
        visible = np.array([1.] * 16, dtype=np.float32)
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

    def __len__(self):
        return len(self.samples)
