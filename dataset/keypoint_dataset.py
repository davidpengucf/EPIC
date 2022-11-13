
from abc import ABC
import numpy as np
from torch.utils.data.dataset import Dataset
from webcolors import name_to_rgb
import cv2


class KeypointDataset(Dataset, ABC):
   
    def __init__(self, root, num_keypoints, samples, transforms=None, image_size=(256, 256), heatmap_size=(64, 64),
                 sigma=2, keypoints_group=None, colored_skeleton=None):
        self.root = root
        self.num_keypoints = num_keypoints
        self.samples = samples
        self.transforms = transforms
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.keypoints_group = keypoints_group
        self.colored_skeleton = colored_skeleton

    def __len__(self):
        return len(self.samples)

    def visualize(self, image, keypoints, filename):
        
        assert self.colored_skeleton is not None

        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).copy()
        for (_, (line, color)) in self.colored_skeleton.items():
            for i in range(len(line) - 1):
                start, end = keypoints[line[i]], keypoints[line[i + 1]]
                cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color=name_to_rgb(color),
                         thickness=3)
        for keypoint in keypoints:
            cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 3, name_to_rgb('black'), 1)
        cv2.imwrite(filename, image)

    def group_accuracy(self, accuracies):
        
        grouped_accuracies = dict()
        for name, keypoints in self.keypoints_group.items():
            grouped_accuracies[name] = sum([accuracies[idx] for idx in keypoints]) / len(keypoints)
        return grouped_accuracies


class AnimalKeypointDataset(Dataset, ABC):
   
    def __init__(self, root, num_keypoints, samples, transforms=None, image_size=(256, 256), heatmap_size=(64, 64),
                 sigma=2, keypoints_group=None, colored_skeleton=None):
        self.root = root
        self.num_keypoints = num_keypoints
        self.samples = samples
        self.transforms = transforms
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.keypoints_group = keypoints_group
        self.colored_skeleton = colored_skeleton

    def __len__(self):
        return len(self.samples)

    def visualize(self, image, keypoints, filename):
        
        assert self.colored_skeleton is not None

        #image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).copy()
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB).copy()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for (_, (line, color)) in self.colored_skeleton.items():
            for i in range(len(line) - 1):
                start, end = keypoints[line[i]], keypoints[line[i + 1]]
                cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color=name_to_rgb(color),
                         thickness=3)
        for keypoint in keypoints:
            cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 3, name_to_rgb('black'), 1)
        cv2.imwrite(filename, image)

    def group_accuracy(self, accuracies):
        
        grouped_accuracies = dict()
        for name, keypoints in self.keypoints_group.items():
            grouped_accuracies[name] = sum([accuracies[idx] for idx in keypoints]) / len(keypoints)
        return grouped_accuracies


class Body16KeypointDataset(KeypointDataset, ABC):
    """
    Dataset with 16 body keypoints.
    """
    # TODO: add image
    head = (9,)
    shoulder = (12, 13)
    elbow = (11, 14)
    wrist = (10, 15)
    hip = (2, 3)
    knee = (1, 4)
    ankle = (0, 5)
    all = (12, 13, 11, 14, 10, 15, 2, 3, 1, 4, 0, 5)
    right_leg = (0, 1, 2, 8)
    left_leg = (5, 4, 3, 8)
    backbone = (8, 9)
    right_arm = (10, 11, 12, 8)
    left_arm = (15, 14, 13, 8)

    def __init__(self, root, samples, **kwargs):
        colored_skeleton = {
            "right_leg": (self.right_leg, 'yellow'),
            "left_leg": (self.left_leg, 'green'),
            "backbone": (self.backbone, 'blue'),
            "right_arm": (self.right_arm, 'purple'),
            "left_arm": (self.left_arm, 'red'),
        }
        keypoints_group = {
            "head": self.head,
            "shoulder": self.shoulder,
            "elbow": self.elbow,
            "wrist": self.wrist,
            "hip": self.hip,
            "knee": self.knee,
            "ankle": self.ankle,
            "all": self.all
        }
        super(Body16KeypointDataset, self).__init__(root, 16, samples, keypoints_group=keypoints_group,
                                                    colored_skeleton=colored_skeleton, **kwargs)


class Animal18KeypointDataset(AnimalKeypointDataset, ABC):
    """
    Dataset with 18 animal keypoints.
    """
    # parts = {'eye': [0, 1], 'chin': [2], 'hoof': [3, 4, 5, 6], 'hip': [7], 'knee': [8, 9, 10, 11], 'shoulder': [12, 13], 'elbow': [14, 15, 16, 17]}
    # TODO: add image
    
    eye = (0, 1)
    chin = (2,)
    shoulder = (12, 13)
    hip = (7,)
    elbow = (14, 15, 16, 17)
    knee = (8, 9, 10, 11)
    hoof = (3, 4, 5, 6)
    all = tuple(range(18))
    
    front1 = (0, 2)
    front2 = (1, 2)
    mid1 = (12, 7)
    mid2 = (13, 7)
    up1 = (14, 8)
    up2 = (15, 9)
    up3 = (16, 10)
    up4 = (17, 11)
    down1 = (8, 3)
    down2 = (9, 4)
    down3 = (10, 5)
    down4 = (11, 6)
    

    def __init__(self, root, samples, **kwargs):
        colored_skeleton = {
            "front1": (self.front1, 'red'),
            "front2": (self.front2, 'blue'),
            "mid1": (self.mid1, 'red'),
            "mid2": (self.mid2, 'blue'),
            "up1": (self.up1, 'purple'),
            "down1": (self.down1, 'purple'),
            "up2": (self.up2, 'orange'),
            "down2": (self.down2, 'orange'),
            "up3": (self.up3, 'yellow'),
            "down3": (self.down3, 'yellow'),
            "up4": (self.up4, 'green'),
            "down4": (self.down4, 'green'),
            
        }
        keypoints_group = {
            "Eye": self.eye,
            "Chin": self.chin,
            "Sld": self.shoulder,
            "Hip": self.hip,
            "Elb": self.elbow,
            "Knee": self.knee,
            "Hoof": self.hoof,
            "all": self.all
        }
        super(Animal18KeypointDataset, self).__init__(root, 18, samples, keypoints_group=keypoints_group,
                                                    colored_skeleton=colored_skeleton, **kwargs)
                                                    
class Hand21KeypointDataset(KeypointDataset, ABC):
    """
    Dataset with 21 hand keypoints.
    """
    # TODO: add image
    MCP = (1, 5, 9, 13, 17)
    PIP = (2, 6, 10, 14, 18)
    DIP = (3, 7, 11, 15, 19)
    fingertip = (4, 8, 12, 16, 20)
    all = tuple(range(21))
    thumb = (0, 1, 2, 3, 4)
    index_finger = (0, 5, 6, 7, 8)
    middle_finger = (0, 9, 10, 11, 12)
    ring_finger = (0, 13, 14, 15, 16)
    little_finger = (0, 17, 18, 19, 20)

    def __init__(self, root, samples, **kwargs):
        colored_skeleton = {
            "thumb": (self.thumb, 'yellow'),
            "index_finger": (self.index_finger, 'green'),
            "middle_finger": (self.middle_finger, 'blue'),
            "ring_finger": (self.ring_finger, 'purple'),
            "little_finger": (self.little_finger, 'red'),
        }
        keypoints_group = {
            "MCP": self.MCP,
            "PIP": self.PIP,
            "DIP": self.DIP,
            "fingertip": self.fingertip,
            "all": self.all
        }
        super(Hand21KeypointDataset, self).__init__(root, 21, samples, keypoints_group=keypoints_group,
                                                    colored_skeleton=colored_skeleton, **kwargs)
