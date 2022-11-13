
import torch
import os
import pickle

from ._util import download as download_data, check_exits
from .transforms import *
from .keypoint_dataset import Hand21KeypointDataset
from .util import *


class RenderedHandPose(Hand21KeypointDataset):
   
    def __init__(self, root, split='train', task='all', download=False, **kwargs):

        root = os.path.join(root, "RHD_published_v2")

        assert split in ['train', 'test', 'all']
        self.split = split
        if split == 'all':
            samples = self.get_samples(root, 'train') + self.get_samples(root, 'test')
        else:
            samples = self.get_samples(root, split)

        super(RenderedHandPose, self).__init__(
            root, samples, **kwargs)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample['name']
        image_path = os.path.join(self.root, image_name)
        image = Image.open(image_path)

        #if index == 1: print("rendered_hand_pose: image ", type(image))
        #if index == 1: print("rendered_hand_pose: image ", image)
        #if index == 1: print("rendered_hand_pose: image ",image.size) (320, 320)

        keypoint3d_camera = np.array(sample['keypoint3d'])  # NUM_KEYPOINTS x 3
        keypoint2d = np.array(sample['keypoint2d'])  # NUM_KEYPOINTS x 2
        intrinsic_matrix = np.array(sample['intrinsic_matrix'])
        Zc = keypoint3d_camera[:, 2]

        #if index == 1: print("rendered_hand_pose: keypoint3d_camera ", keypoint3d_camera)
        #if index == 1: print("rendered_hand_pose: keypoint3d_camera ", keypoint3d_camera.shape) (21, 3)
        #if index == 1: print("rendered_hand_pose: keypoint2d ", keypoint2d)
        #if index == 1: print("rendered_hand_pose: keypoint2d ", keypoint2d.shape) (21, 2)
       # if index == 1: print("rendered_hand_pose: intrinsic_matrix ", intrinsic_matrix)
        #if index == 1: print("rendered_hand_pose: intrinsic_matrix ", intrinsic_matrix.shape) (3,3)
        #if index == 1: print("rendered_hand_pose: Zc ", Zc)
        #if index == 1: print("rendered_hand_pose: Zc ", Zc.shape) (21,)

        # Crop the images such that the hand is at the center of the image
        # The images will be 1.5 times larger than the hand
        # The crop process will change Xc and Yc, leaving Zc with no changes
        bounding_box = get_bounding_box(keypoint2d)
        #if index == 1: print("rendered_hand_pose: bounding_box ", bounding_box) #(136.6, 62.39, 193.0, 169.3)
        w, h = image.size
        left, upper, right, lower = scale_box(bounding_box, w, h, 1.5, False)
        #if index == 1: left, upper, right, lower = scale_box(bounding_box, w, h, 1.5, True)(85, 36, 244, 195)
        #if index == 1: print("rendered_hand_pose: scaled_bounding_box ", (left, upper, right, lower))
        # 36, 85, 159, 159

        image, keypoint2d = crop(image, upper, left, lower - upper, right - left, keypoint2d, False)
        #if index == 1: image, keypoint2d = crop(image, upper, left, lower - upper, right - left, keypoint2d, True)
        #if index == 1: print("rendered_hand_pose: cropped keypoint2d ", keypoint2d)

        # Change all hands to right hands
        if sample['left'] is False:
            image, keypoint2d = hflip(image, keypoint2d)
            #if index == 1: print("rendered_hand_pose: cropped keypoint2d ", keypoint2d)

        #if index == 1: print("rendered_hand_pose: transforms ", self.transforms)
        #if index == 1: print("rendered_hand_pose: original intrinsic_matrix ", intrinsic_matrix)
        image, data = self.transforms(image, keypoint2d=keypoint2d, intrinsic_matrix=intrinsic_matrix)
        keypoint2d = data['keypoint2d']
        intrinsic_matrix = data['intrinsic_matrix']
        #if index == 1: print("rendered_hand_pose: transformed img ", image.size)
        #if index == 1: print("rendered_hand_pose: transformed keypoint2d ", keypoint2d)
        #if index == 1: print("rendered_hand_pose: transformed intrinsic_matrix ", intrinsic_matrix)
        keypoint3d_camera = keypoint2d_to_3d(keypoint2d, intrinsic_matrix, Zc)
        #if index == 1: print("rendered_hand_pose: keypoint3d_camera ", keypoint3d_camera) # keep Zc

        # noramlize 2D pose:
        visible = np.array(sample['visible'], dtype=np.float32)
        #if index == 1: print("rendered_hand_pose: original visible ", visible) ones(21)
        visible = visible[:, np.newaxis]
        #if index == 1: print("rendered_hand_pose: extended visible ", visible) ones(1,21)
        # 2D heatmap # preprocess image and keypoints first then produce heatmaps
        #if index == 1: print("index: ", index)
        #if index == 1: target, target_weight = generate_target(keypoint2d, visible, self.heatmap_size, self.sigma, self.image_size, True)
        target, target_weight = generate_target(keypoint2d, visible, self.heatmap_size, self.sigma, self.image_size, False)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        # normalize 3D pose:
        # put middle finger metacarpophalangeal (MCP) joint in the center of the coordinate system
        # and make distance between wrist and middle finger MCP joint to be of length 1
        keypoint3d_n = keypoint3d_camera - keypoint3d_camera[9:10, :]
        keypoint3d_n = keypoint3d_n / np.sqrt(np.sum(keypoint3d_n[0, :] ** 2))
        z = keypoint3d_n[:, 2]

        meta = {
            'image': image_name,
            'keypoint2d': keypoint2d,  # （NUM_KEYPOINTS x 2）
            'keypoint3d': keypoint3d_n,  # （NUM_KEYPOINTS x 3）
            'z': z,
        }

        return image, target, target_weight, meta

    def get_samples(self, root, task, min_size=64):
        if task == 'train':
            set = 'training'
        else:
            set = 'evaluation'
        # load annotations of this set
        with open(os.path.join(root, set, 'anno_%s.pickle' % set), 'rb') as fi:
            anno_all = pickle.load(fi)

        samples = []
        left_hand_index = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17]
        right_hand_index = [i+21 for i in left_hand_index]
        for sample_id, anno in anno_all.items():
            image_name = os.path.join(set, 'color', '%.5d.png' % sample_id)
            mask_name = os.path.join(set, 'mask', '%.5d.png' % sample_id)
            keypoint2d = anno['uv_vis'][:, :2]
            keypoint3d = anno['xyz']
            intrinsic_matrix = anno['K']
            visible = anno['uv_vis'][:, 2]

            left_hand_keypoint2d = keypoint2d[left_hand_index] # NUM_KEYPOINTS x 2
            left_box = get_bounding_box(left_hand_keypoint2d)
            right_hand_keypoint2d = keypoint2d[right_hand_index]  # NUM_KEYPOINTS x 2
            right_box = get_bounding_box(right_hand_keypoint2d)

            w, h = 320, 320
            scaled_left_box = scale_box(left_box, w, h, 1.5)
            left, upper, right, lower = scaled_left_box
            size = max(right - left, lower - upper)
            if size > min_size and np.sum(visible[left_hand_index]) > 16 and area(*intersection(scaled_left_box, right_box)) / area(*scaled_left_box) < 0.3:
                sample = {
                    'name': image_name,
                    'mask_name': mask_name,
                    'keypoint2d': left_hand_keypoint2d,
                    'visible': visible[left_hand_index],
                    'keypoint3d': keypoint3d[left_hand_index],
                    'intrinsic_matrix': intrinsic_matrix,
                    'left': True
                }
                samples.append(sample)

            scaled_right_box = scale_box(right_box, w, h, 1.5)
            left, upper, right, lower = scaled_right_box
            size = max(right - left, lower - upper)
            if size > min_size and np.sum(visible[right_hand_index]) > 16 and area(*intersection(scaled_right_box, left_box)) / area(*scaled_right_box) < 0.3:
                sample = {
                    'name': image_name,
                    'mask_name': mask_name,
                    'keypoint2d': right_hand_keypoint2d,
                    'visible': visible[right_hand_index],
                    'keypoint3d': keypoint3d[right_hand_index],
                    'intrinsic_matrix': intrinsic_matrix,
                    'left': False
                }
                samples.append(sample)

        return samples
