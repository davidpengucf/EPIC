import scipy.io as scio
import os

from PIL import ImageFile
import torch
from .keypoint_dataset import Body16KeypointDataset
from .transforms import *
from .util import *
from ._util import download as download_data, check_exits


ImageFile.LOAD_TRUNCATED_IMAGES = True


class LSP(Body16KeypointDataset):
   
    def __init__(self, root, split='train', task='all', download=False, image_size=(256, 256), transforms=None, **kwargs):
        

        assert split in ['train', 'test', 'all']
        self.split = split

        samples = []
        annotations = scio.loadmat(os.path.join(root, "joints.mat"))['joints'].transpose((2, 1, 0))
        for i in range(0, 2000):
            image = "im{0:04d}.jpg".format(i+1)
            annotation = annotations[i]
            samples.append((image, annotation))

        self.joints_index = (0, 1, 2, 3, 4, 5, 13, 13, 12, 13, 6, 7, 8, 9, 10, 11)
        self.visible = np.array([1.] * 6 + [0, 0] + [1.] * 8, dtype=np.float32)
        normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms = Compose([
            ResizePad(image_size[0]),
            ToTensor(),
            normalize
        ])
        super(LSP, self).__init__(root, samples, transforms=transforms, image_size=image_size, **kwargs)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample[0]
        image = Image.open(os.path.join(self.root, "images", image_name))
        keypoint2d = sample[1][self.joints_index, :2]
        image, data = self.transforms(image, keypoint2d=keypoint2d)
        keypoint2d = data['keypoint2d']
        visible = self.visible * (1-sample[1][self.joints_index, 2])
        visible = visible[:, np.newaxis]

        # 2D heatmap
        target, target_weight = generate_target(keypoint2d, visible, self.heatmap_size, self.sigma, self.image_size)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_name,
            'keypoint2d': keypoint2d,  # （NUM_KEYPOINTS x 2）
            'keypoint3d': np.zeros((self.num_keypoints, 3)).astype(keypoint2d.dtype),  # （NUM_KEYPOINTS x 3）
        }
        return image, target, target_weight, meta
