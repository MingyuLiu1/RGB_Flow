import numpy as np
import open3d as o3d
import argparse
import fps
import torch
import glob
import torch
from torch_geometric.nn import fps, knn_graph
import os.path

nodes = 42
FPS_N = 2048


class KITTIPoleSample:
    """
    Add this class into our KITTI dataloader.

    Note:
        We need to update the data_path function to make sure that it can be loaded samples automatically.

    Args:
        data_path: The path to KITTI dataset
        pcd_file: The point cloud sample
        lables_file: The labels of each sample
        calib_file: calib labels

    Return:
        labels of each points: Car[1, 0, 0, 0], Pedestrian[0, 1, 0, 0], Cyclist[0, 0, 1, 0], Background(DontCare):[0, 0, 0, 1]

    """

    def __init__(self, data_path, nodes, pcd_file, lables_file, calib_file, idx_file, device='cuda'):
        super(KITTIPoleSample, self).__init__()
        self.data_path = data_path
        self.num_nodes = nodes
        self.device = device
        self.idx_file = os.path.join('/mmdetection3d/data/kitti/ImageSets', idx_file)

        # self.pcd_files = self._get_file_list(pcd_file)
        # self.lables_files = self._get_file_list(lables_file)
        # self.calib_files = self._get_file_list(calib_file)

    def _get_file_list(self, path):
        """Prepare file list for the dataset"""
        lines = sorted(glob.glob(path))
        # print(lines)
        # print(len(lines))
        return lines

    def get_points_labels(self, idx): 
        with open(self.idx_file, 'r') as f:
            line = f.readlines()[idx]
        file_idx = int(line)
        label_filename = os.path.join('/mmdetection3d/data/kitti/training/label_2', '%06d.txt'%(file_idx))
        pc_filename = os.path.join('/mmdetection3d/data/kitti/training/velodyne', '%06d.bin'%(file_idx))
        calib_filename = os.path.join('/mmdetection3d/data/kitti/training/calib', '%06d.txt'%(file_idx))

        # get points
        pcd = o3d.geometry.PointCloud()
        points = np.fromfile(pc_filename, dtype=np.float32).reshape(-1, 4)[:, :3]
        # points_front = points[np.where(points[:, 0]>0)]
        points_x = torch.from_numpy(points).cuda(0)#.to(self.device)
        # downsample the input point cloud
        indices_fps = fps(points_x, ratio=float(2048 / points.shape[0])).cuda(0)#.to(self.device)
        points_x_sampled = points_x[indices_fps]

        pcd.points = o3d.utility.Vector3dVector(points)
    
        # points_front = torch.from_numpy(points_front).cuda(0)
        # get R0_rect, Tr_vel_cam
        R0_rect, Tr_vel_cam = self.get_R0_rect(calib_filename)

        # get labels
        labels, classes = self.get_labels(label_filename)

        # get bboxes
        bboxes = []
        for idx, label in enumerate(labels):
            h, w, l = label[0], label[1], label[2]
            cx, cy, cz = label[3], label[4], label[5]
            rotation = label[6]
            if classes[idx] != 'DontCare':
                bbox = self.get_bbox(cx, cy, cz, h, w, l, rotation, R0_rect, Tr_vel_cam)
                bboxes.append(bbox)
    
    # get front points and center points
        np_bb_centers = np.zeros([self.num_nodes, 3])
        p_in_bb_idx_list = []
        for idx in range(len(bboxes)):
            np_bb_centers[idx] = np.array([bboxes[idx].center])
            indices_p_in_bb = bboxes[idx].get_point_indices_within_bounding_box(pcd.points)
            p_in_bb_idx_list.append(indices_p_in_bb)

        p_in_bb_idx_list = np.concatenate(p_in_bb_idx_list).astype(int)
    
        angle  = points[:, 1] / points[:, 0] # calculate the pole angle
        points_front = points[np.where((points[:, 0] > 0) & (angle < 1.3) & (angle > -1.3) & (points[:, 1] > -20) & (points[:, 1] < 20) & (points[:, 2] > -3) & (points[:, 2] < 1))]
        pcd.points = o3d.utility.Vector3dVector(points_front)
        points_labels = np.full((len(points_front), 2), [0, 1])

        if len(p_in_bb_idx_list) == 0:
            labels_sample = points_labels[0: 10000]
            labels_sample = np.reshape(labels_sample, (1, 1, len(labels_sample), 2))
            points_front_sample = torch.from_numpy(points_front[0: 10000]).cuda(0).unsqueeze(0)


            return points_front_sample.unsqueeze(0), labels_sample
        
        for i in range(len(bboxes)):
            front_p_in_bb = bboxes[i].get_point_indices_within_bounding_box(pcd.points)
            
            if classes[i] == 'Car':
                points_labels[front_p_in_bb] = [1, 0]

        x = points_front[:, 0]
        y = points_front[:, 1]
        r = np.sqrt(x**2 + y**2)
        # r <= 5
        points_nn = points_front[np.where((r <= 5))]
        labels_nn = points_labels[np.where((r <= 5))]
        nn_i = np.random.permutation(len(points_nn))[0: int(len(points_nn) * 0.01)]
        points_nn_sample = points_nn[nn_i]
        labels_nn_sample = labels_nn[nn_i]
        # print("nn_i {}".format(nn_i.shape))
        # 5 < r <= 10
        points_n = points_front[np.where((5 < r) & (r <= 10))]
        labels_n = points_labels[np.where((5 < r) & (r <= 10))]
        n_i = np.random.permutation(len(points_n))[0: int(len(points_n) * 0.005)]
        points_n_sample = points_n[n_i]
        labels_n_sample = labels_n[n_i]
        # print("n_i {}".format(n_i.shape))
        # 10 < r <= 20
        points_m = points_front[np.where((10 < r) & (r <= 20))]
        labels_m = points_labels[np.where((10 < r) & (r <= 20))]
        m_i = np.random.permutation(len(points_m))[0: int(len(points_m) * 0.2)]
        points_m_sample = points_m[m_i]
        labels_m_sample = labels_m[m_i]
        # print("m_i {}".format(m_i.shape))
        # 20 < r <= 40
        points_f = points_front[np.where((20 < r) & (r <= 40))]
        labels_f = points_labels[np.where((20 < r) & (r <= 40))]
        f_i = np.random.permutation(len(points_f))[0: int(len(points_f) * 1)]
        points_f_sample = points_f[f_i]
        labels_f_sample = labels_f[f_i]
        # 40 < r
        points_ff = points_front[np.where((40 < r))]
        labels_ff = points_labels[np.where((40 < r))]
        ff_i = np.random.permutation(len(points_ff))[0: int(len(points_ff) * 1)]
        points_ff_sample = points_ff[ff_i]
        labels_ff_sample = labels_ff[ff_i]
        # print("f_i {}".format(f_i.shape))
        # points_front = torch.from_numpy(points_front).cuda(0)
        points_front_sample = np.concatenate((points_nn_sample, points_n_sample, points_m_sample, points_f_sample, points_ff_sample))
        points_front_sample = torch.from_numpy(points_front_sample).cuda(0).unsqueeze(0)
        labels_sample = np.concatenate((labels_nn_sample, labels_n_sample, labels_m_sample, labels_f_sample, labels_ff_sample))
        labels_sample = np.reshape(labels_sample, [1, 1, len(labels_sample), 2])
        #print(points_front_sample.shape) 
        # print(labels_sample.shape)
        center_points = torch.from_numpy(np_bb_centers)


        return points_front_sample.unsqueeze(0), labels_sample


    # auxilary function
    # get bounding boxes
    def get_bbox(self, cx, cy, cz, h, w, l, rotation, R0_vect, Tr_vel_cam):
        rotation_matrix = np.array([
            [np.cos(rotation), 0, np.sin(rotation)],
            [0, 1, 0.0],
            [-np.sin(rotation), 0.0, np.cos(rotation)]])

        bb_box = o3d.geometry.OrientedBoundingBox(np.array([cx, cy - h / 2, cz]), rotation_matrix, np.array([l, w, h]))
        # bb_box.color = np.array([0, 0, 1])
        bb_box.rotate(np.linalg.inv(R0_vect), center=(0, 0, 0))
        Tf_inv = np.linalg.inv(Tr_vel_cam)
        bb_box.rotate(Tf_inv[:3, :3], center=(0, 0, 0))
        bb_box.translate(Tf_inv[:3, 3], relative=True)
        return bb_box

    # get R0_rect, Tr_vel_cam
    def get_R0_rect(self, calibPath):
        with open(calibPath, 'r') as f:
            tmp = f.read().split('\n')

        for idx in range(len(tmp)):
            if tmp[idx].split(' ')[0] == "R0_rect:":
                R0 = tmp[idx]

        for idx in range(len(tmp)):
            if tmp[idx].split(' ')[0] == "Tr_velo_to_cam:":
                Tr = tmp[idx]

        R0 = np.loadtxt(R0.split(' ')[1:]).reshape((3, 3))
        Tr = np.loadtxt(Tr.split(' ')[1:]).reshape((3, 4))
        return R0, np.append(Tr, np.array([[0, 0, 0, 1]]), axis=0)

    # get labels
    def get_labels(self, labels_path):
        labels = []
        classes = []
        with open(labels_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line != '\n':
                    line = line.split()
                    classes.append(line[0])  # get class of each labels
                    line = line[8: 15]
                    line = [float(x) for x in line]
                    labels.append(line)
        labels = np.array(labels)
        return labels, classes

    def __len__(self):
        return len(open(self.idx_file, 'r').readlines())

    def __getitem__(self, idx):
        data = {}
        points, points_labels = self.get_points_labels(idx)

        #data['pcd_downsampled'] = pcd_sampled
        #data['centroids'] = centroids
        
        return points, points_labels
