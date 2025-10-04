import os
import torch
import random
import numpy as np
import cv2
import json
from torch.utils.data import Dataset
import PIL
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms


from utils import *
from shapely.geometry import Polygon
from shapely import affinity
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


import pickle
import pdb
import scipy.io as io
from tqdm import tqdm
from tqdm import trange




class DrawEllipse(object):
    def __init__(self, cuts=120):
        self.cuts = cuts

    def make_ellipse(self, w, h, cuts):
        points = []
        
        for i in range(cuts):
            deg = i*(360/cuts)*3.14/180
            points.append((w*np.sin(deg), h*np.cos(deg)))
        return Polygon(points)

    def draw_ellipse(self, points, size, fill=1):
        axis = Image.new('L', size)
        # w, h = img.size
        draw = ImageDraw.Draw(axis)
            
        draw.polygon(points, fill=fill, outline=None)
        axis = np.asarray(axis).astype(np.float32)
        return axis#, theta_degree

    def dist(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        dist = (x1 - x2) ** 2 + (y1 - y2) **2
        return dist ** 0.5

    def get_theta(self, point1, point2):
        b, a = (point2[1] - point1[1]), (point2[0] - point1[0])
        if a == 0:
            # print('line |')
            return 90
        else:
            tan = - b / a # i changed this too
        return np.arctan(tan) * 180 / np.pi
        
    def __call__(self, points, size, cuts=120):
        if len(points) == 4:
            ctr_x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
            ctr_y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
            _points = [(ctr_x, ctr_y)]
            _points += points
            points = _points
        # elif len(points) != 5:
        #     print('sth wrong ,', len(points))
            
        up, down, left, right = 1, 2, 3, 4
        left, right, up, down = points[left], points[right], points[up], points[down]
        w = (self.dist(points[0], left) + self.dist(points[0], right)) / 2
        h = (self.dist(points[0], up) + self.dist(points[0], down)) / 2
        theta = (self.get_theta(up, down) + 180) % 180 - 90
        
        ellipse = self.make_ellipse(w, h, cuts)
        ellipse = affinity.translate(ellipse, xoff=points[0][0], yoff=points[0][1])
        ellipse = affinity.rotate(ellipse, -theta, origin='centroid')
        
        axis = self.draw_ellipse(ellipse.exterior.coords, size)
        return axis, (points[0][0], points[0][1])

class NewSymmetryDatasetsBase(Dataset):
    def __init__(self, sym_type, get_polygon=2, split='train', root='./sym_datasets/DENDI', with_ref_circle=1, n_theta=8):
        super(NewSymmetryDatasetsBase, self).__init__()
        self.root = root
        self.split = split
        self.sym_type = sym_type
        self.get_polygon = get_polygon
        self.with_ref_circle = with_ref_circle
        self.order_list = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 20, 22, 26, 28, 29] # 20+1
        self.img_list, self.gt_list = self.get_data_list()
        self.ellipse = DrawEllipse()
        self.n_theta = n_theta
        self.ellipse_theta_filter = self.construct_theta_filter()

    def construct_theta_filter(self):
        # angle_interval, n_theta = 45, 8
        angle_interval = int(360 / self.n_theta)
        n_theta = self.n_theta        
        if self.split == 'test':
            c = int(417*5)
            # c = int(417 * 1.6)
        else:
            c = int(417 * 1.6)
        self.filter_ks = c
        base = torch.ones((c*2+1, c*2+1))
        indices_all = torch.nonzero(base)
        center = torch.tensor([c, c])
        dh_dw = indices_all - center
        tangents = - (dh_dw[:, 0]) / (dh_dw[:, 1] + 1e-2)
        theta = np.arctan(tangents)
        theta = (theta * 180 / np.pi) % 360
        t_lbl = torch.zeros(c*2+1, c*2+1, n_theta)

        d = angle_interval / 2
        k = theta // d
        a = k + 1 - theta / d
        indices1, indices2 = (k + 3) % n_theta, (k + 4) % n_theta

        t_lbl[indices_all[:, 0], indices_all[:, 1], indices1.long()] = a
        t_lbl[indices_all[:, 0], indices_all[:, 1], indices2.long()] = 1 - a

        return t_lbl

    def get_data_list(self):
        split_dict = torch.load(os.path.join(self.root, '%s_split.pt') % self.sym_type)
        if self.split == 'all':
            split_dict = split_dict['all']
            img_key, ann_key = 'img', 'ann'
        else:
            img_key, ann_key  = '%s_img' % self.split, '%s_ann' % self.split
        img_list = [os.path.join(self.root, name) for name in split_dict[img_key]]
        gt_list = []

        for path in split_dict[ann_key]:
            json_path = os.path.join(self.root, path)
            with open(json_path) as json_file:
                json_data = json.load(json_file)
                gt = json_data['figures']
                gt_list.append(gt)        

        return img_list, gt_list
    
    def process_data_ref(self, gt, size):
        gt_dict = {}
        lines = []
        ellipse_pts = []
        
        for f in gt:
            if f['label'] in ['reflection']: # polyline, non circle
                for i in range(len(f['shape']['coordinates']) - 1):
                    x1, y1 = f['shape']['coordinates'][i]
                    x2, y2 = f['shape']['coordinates'][i+1]
                    lines.append([x1, y1, x2, y2])
            elif f['label'] in ['reflection-circle']:
                if len(f['shape']['coordinates']) == 2:
                    x1, y1 = f['shape']['coordinates'][0]
                    x2, y2 = f['shape']['coordinates'][1]
                    lines.append([x1, y1, x2, y2])
                else:
                    ellipse_pts.append(f['shape']['coordinates'])

        ellipse_axis_lbl = []
        ellipse_coords = []

        if self.with_ref_circle in [1]:
            for i, pts in enumerate(ellipse_pts):
                ellipse_axis, center_coords = self.ellipse(pts, size)
                _ellipse_axis_lbl = ellipse_axis * (i + 1000 + 1)
                # _ellipse_axis_lbl = - ellipse_axis * (i+1+axis_lbl.max())
                ellipse_axis_lbl.append(_ellipse_axis_lbl)
                # (cx, cy, cs, cy)
                _coords = [center_coords[0] / size[0], center_coords[1] / size[1], \
                    center_coords[0] / size[0], center_coords[1] / size[1] ]
                
                ellipse_coords.append(_coords)
        elif self.with_ref_circle in [2]:
            for i, pts in enumerate(ellipse_pts):
                if len(pts) == 4:
                    lines.append([pts[0][0], pts[0][1], pts[1][0], pts[1][1]])
                    lines.append([pts[2][0], pts[2][1], pts[3][0], pts[3][1]])
                else:
                    lines.append([pts[1][0], pts[1][1], pts[2][0], pts[2][1]])
                    lines.append([pts[3][0], pts[3][1], pts[4][0], pts[4][1]])

        axis_lbl, line_coords = draw_axis(lines, size)

        if len(ellipse_axis_lbl):
            gt_dict['ellipse_axis_lbl'] = ellipse_axis_lbl
            ellipse_axis_lbl.append(axis_lbl)
            maps = np.stack(ellipse_axis_lbl, axis=0)
            axis_lbl = np.max(maps, axis=0) # ellipse -1

        if len(line_coords) and len(ellipse_coords):
            line_coords = np.concatenate((np.array(line_coords), np.array(ellipse_coords)), axis=0)
        elif len(ellipse_coords):
            line_coords = np.array(ellipse_coords)

        gt_dict['axis_lbl'] = axis_lbl
        gt_dict['axis'] = (axis_lbl > 0).astype(axis_lbl.dtype)
        gt_dict['line_coords'] = line_coords
        
        return gt_dict

    def process_order(self, N):
        if N in self.order_list:
            return self.order_list.index(N)
        else:
            return 1

    def process_data_rot(self, gt, size):
        w, h = size
        gt_dict = {}
        centers = []
        orders = []

        for f in gt:
            if f['label'] in ['rotation-polygon']:
                x1, y1 = f['shape']['coordinates'][0]
                centers.append((int(x1), int(y1)))
                N = abs(int(f['attributes'][0]['value']))
                N = self.process_order(N)
                orders.append(N)
            elif f['label'] in ['rotation-circle']:
                x1, y1 = f['shape']['coordinates'][0]
                centers.append((int(x1), int(y1)))
                N = abs(int(f['attributes'][0]['value']))
                N = self.process_order(N) 
                orders.append(N)

        if len(centers):
            maps = draw_points(centers, orders, size)
            maps = np.stack(maps, axis=0)
            center_map = (np.sum(maps, axis=0) > 0).astype(np.float32)
            order_map = np.max(maps, axis=0)
        else:
            center_map = np.zeros((h, w)).astype(np.float32)
            order_map = np.zeros((h, w)).astype(np.float32)
        
        gt_dict['order_map'] = order_map
        gt_dict['axis_map'] = center_map
        return gt_dict

    def process_theta_ref(self, axis_lbl, axis_coords):
        im_h, im_w = axis_lbl.shape[-2], axis_lbl.shape[-1]
        a_lbl = torch.zeros_like(axis_lbl).unsqueeze(0).unsqueeze(1).expand(-1, self.n_theta, -1, -1)
        ellipse_mask = (axis_lbl > 1000).float()
        ellipse_a_lbl = torch.zeros_like(axis_lbl).unsqueeze(-1).expand(-1, -1, self.n_theta)

        if axis_lbl.max() > 1000:
            # ellipse_lbl = axis_lbl * ellipse_mask
            num_ellipse = axis_lbl.max() - 1000
            ellipse_masks = []
            ellipse_b_lbl = []
            ellipse_n_pix = []
            
            n = num_ellipse.long()
            for i in range(n):
                cx, cy = axis_coords[-n+i][0], axis_coords[-n+i][1]
                b_lbl = (axis_lbl == (i + 1000 + 1)).float()
                n_pix = b_lbl.sum()
                ellipse_masks.append(b_lbl)
                ellipse_n_pix.append(n_pix)
                cx, cy, ks = int(cx * im_w), int(cy * im_h), int(self.filter_ks)
                _filter = self.ellipse_theta_filter[ks - cy:ks - cy + im_h, \
                                                    ks - cx:ks - cx + im_w, :]
                b_lbl = _filter * b_lbl.unsqueeze(-1)
                ellipse_b_lbl.append(b_lbl.float())
            
            stack_order = torch.argsort(torch.tensor(ellipse_n_pix), descending=True)
            for idx in stack_order:
                ellipse_a_lbl = ellipse_a_lbl * (1 - ellipse_masks[idx].unsqueeze(-1)) \
                    + ellipse_b_lbl[idx] * ellipse_masks[idx].unsqueeze(-1)
                
            axis_lbl = axis_lbl * (1 - ellipse_mask)
        num_lines = axis_lbl.max().long()

        if num_lines > 0:
            ### input axis_lbl (0 ~ N), # axis_coords (x1, y1, x2, y2) normalized
            ### output axis (0 or 1) # angle label (H, W, nangle) sum 1
            axis_coords = np.array(axis_coords[:num_lines])
            ### y in image coordinate is different from the world coord
            theta = np.where(axis_coords[:, 2] == axis_coords[:, 0], 90, \
                            np.arctan(-(axis_coords[:, 3] - axis_coords[:, 1]) / (axis_coords[:, 2] - axis_coords[:, 0])) \
                                * 180 / np.pi )
            # kernel theta interval
            d = self.angle_interval / 2
            k = theta // d
            a = k + 1 - theta / d
            indices1, indices2 = (k + 3) % self.n_theta, (k + 4) % self.n_theta
            # indices1, indices2 = (k) % self.n_theta, (k + 1) % self.n_theta
            # a_lbl [dummy 0, kernel1, kernel2, ...]
            a_lbl = np.zeros((theta.shape[0] + 1, self.n_theta), dtype=np.float32)
            a_lbl[np.arange(theta.shape[0]) + 1, indices1.astype(np.uint8)] = a
            a_lbl[np.arange(theta.shape[0]) + 1, indices2.astype(np.uint8)] = 1 - a
            # a_lbl (H, W, nangle)

            a_lbl = a_lbl[axis_lbl.int(), :]
            a_lbl = torch.from_numpy(a_lbl).permute(2, 0, 1).unsqueeze(0)
        
        ellipse_a_lbl = ellipse_a_lbl.permute(2, 0, 1).unsqueeze(0)
        a_lbl = torch.where(axis_lbl > 0, \
                            a_lbl, ellipse_a_lbl)
        ### move max_pool to model (GPU, ver.) for training
        # a_lbl = F.max_pool2d(a_lbl, kernel_size=5, stride=1, padding=2)
        a_lbl = F.interpolate(a_lbl, (im_h, im_w), mode='nearest')
        return a_lbl

    def process_theta_rot(self, a_lbl):
        # a_lbl (order list index 0~N), out of bound -> 1 
        # draw_points conver 0 -> 255
        a_lbl = a_lbl.unsqueeze(0)
        ### move max_pool to model (GPU, ver.) for training
        # a_lbl = F.max_pool2d(a_lbl, kernel_size=5, stride=1, padding=2).squeeze(1).squeeze(0)
        fg_mask = (a_lbl > 0).float()
        # a_lbl (255->0, 1, 2, ..., N-1)
        a_lbl = (a_lbl != 255).float() * a_lbl
        # a_lbl ((0, 255)->1, 1->2, 2, ..., N-1) * fg_mask (discard 0)
        # print('s, ', a_lbl.shape, self.n_classes, fg_mask.shape, F.one_hot(a_lbl.long()+1, num_classes=self.n_classes).shape)
        a_lbl = F.one_hot(a_lbl.long()+1, num_classes=self.n_classes).squeeze(0).permute(2, 0, 1) * fg_mask
        # a_lbl = F.one_hot(a_lbl.long()+1, num_classes=self.n_classes).permute(2, 0, 1) * fg_mask
        # initial a_lbl (BG, 1, 2, ..., N-1, 255) (255 for order 0)
        # a_lbl (255->0, 1, 2, ..., N-1) * BG_mask
        # angle (0, 1, 2, ...., N-1) one_hot, zero at BG pixels
        # become (BG, 0, 1, 2, ..., N) in model.py (1-> OOB pixels, ignore at training maybe)
        return a_lbl

    def __getitem__ (self, index):
        img_path = self.img_list[index]
        gt = self.gt_list[index]
        img = Image.open(img_path).convert('RGB')
        return img, gt, img_path
    
    def __len__(self):
        return len(self.img_list)
        
def draw_points(points, orders, size):
    maps = []
    for p, o in zip(points, orders):
        cntr = Image.new('L', size)
        draw = ImageDraw.Draw(cntr)
        if o == 0:
            o = 255
        draw.point(p, fill=o)
        cntr = np.asarray(cntr).astype(np.float32)
        maps.append(cntr)
    return maps

class NewSymmetryDatasets(NewSymmetryDatasetsBase):
    def __init__(self, sym_type='rotation', input_size=(417, 417), get_polygon=2, split='train', root='./sym_datasets/DENDI', \
                get_theta=False, n_classes=21, with_ref_circle=1, t_resize=False, n_theta=8):
        super(NewSymmetryDatasets, self).__init__(sym_type, get_polygon, split, root, with_ref_circle, n_theta)
        self.label = [sym_type]
        self.sym_type = sym_type
        self.split = split
        self.input_size = input_size
        self.get_theta = get_theta
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        if self.split == 'all':
            self.mean = [0, 0, 0]
            self.std = [1, 1, 1]
        self.n_classes = n_classes
        self.angle_interval = (360 // n_theta)
        self.n_theta = n_theta
        self.t_resize = t_resize 

    def process_data(self, gt, size):
        if self.sym_type in ['rotation']:
            return None, self.process_data_rot(gt, size)
        elif self.sym_type in ['reflection']:
            return self.process_data_ref(gt, size), None
        elif self.sym_type in ['joint']:
            return self.process_data_ref(gt, size), self.process_data_rot(gt, size)
        return gt

    def transform_data(self, img, gt, transform, reflection=True, t_resize=None):
        if gt is None:
            return None

        if reflection:
            axis, axis_lbl, axis_coords = gt['axis'], gt['axis_lbl'], gt['line_coords']
            
            axis_coords1, axis_coords2 = [], []
            for c in axis_coords:
                axis_coords1.append([c[0], c[1], c[0], c[1]])
                axis_coords2.append([c[2], c[3], c[2], c[3]])
            axis_gs = cv2.GaussianBlur(axis, (5,5), cv2.BORDER_DEFAULT)
            axis_gs = np.clip(axis_gs, 0, 0.21260943) # in case of the intersections
        else:
            axis, a_lbl = gt['axis_map'], gt['order_map']
            axis_gs = cv2.GaussianBlur(axis, (11, 11), cv2.BORDER_DEFAULT)
            axis_gs = np.clip(axis_gs, 0, 0.01) # in case of the intersections

        if self.split in ['test', 'val', 'all'] and t_resize is not None:
            t_resize = t_resize(image=img, axis_gs=axis_gs)
            img, axis_gs = t_resize['image'], t_resize['axis_gs']

        if reflection:
            t = transform(image = img, axis = axis, axis_gs = axis_gs, axis_lbl=axis_lbl, 
                          axis_coords1=axis_coords1, axis_coords2=axis_coords2,
                          rotate = [[100,100,150,150]])
            img, axis, axis_gs, axis_lbl, axis_coords1, axis_coords2, rotate = \
                t["image"], t["axis"], t["axis_gs"], t["axis_lbl"], t["axis_coords1"], t["axis_coords2"], t['rotate']
            axis_coords = []
            for a, b in zip(axis_coords1, axis_coords2):
                axis_coords.append([a[0], a[1], b[0], b[1]])
        else:
            t = transform(image = img, axis = axis, axis_gs = axis_gs, a_lbl=a_lbl)
            img, axis, axis_gs, a_lbl = t["image"], t["axis"], t["axis_gs"], t["a_lbl"]

        mask = (axis_gs != 255).unsqueeze(0)
        axis = axis.unsqueeze(0)
        axis_gs = axis_gs.unsqueeze(0)
        axis_gs = axis_gs / (axis_gs.max() + 1e-5)
        r_dict = {'img': img, 'mask': mask, 'axis': axis, 'axis_gs': axis_gs, 'rotate': rotate}

        if reflection:
            r_dict['axis_lbl'], r_dict['axis_coords'] = axis_lbl, axis_coords
        else:
            r_dict['a_lbl'] = a_lbl
        return r_dict
                
    def __getitem__ (self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        ref_gt, rot_gt = self.process_data(self.gt_list[index], img.size)
        img = match_input_type(img)

        t_resize, rot_a_lbl, ref_a_lbl = None, 0, 0
        additional_targets={'axis': 'mask', 'axis_gs': 'mask', 'a_lbl': 'mask'}
        if self.split in ['test', 'val', 'all']:
            additional_targets['axis_lbl'] = 'mask'
            transform = A.Compose(
                        [ A.Normalize(self.mean, self.std),
                          ToTensorV2(),
                        ], additional_targets=additional_targets)
            t_resize = A.Compose([
                A.LongestMaxSize(max_size=self.input_size[0]),
                A.PadIfNeeded(min_height=self.input_size[0], min_width=self.input_size[1], \
                              border_mode=cv2.BORDER_CONSTANT, mask_value=255),
            ], additional_targets={'axis_gs': 'mask'})
            r_angle = 0
            angle_deg = 0
        else:
            additional_targets['axis_lbl'] = 'mask'
            additional_targets['axis_coords1'] = 'bboxes'
            additional_targets['axis_coords2'] = 'bboxes'
            additional_targets['rotate'] = 'bboxes'
            transform = A.Compose(
                    [ 
                        A.LongestMaxSize(max_size=self.input_size[0]),
                        A.PadIfNeeded(min_height=self.input_size[0], min_width=self.input_size[1], \
                                      border_mode=cv2.BORDER_CONSTANT,),
                    A.RandomRotate90(),
                    #A.Rotate(limit = 15, border_mode = cv2.BORDER_CONSTANT),
                    A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                    A.Normalize(self.mean, self.std),
                    ToTensorV2(),
                    ], additional_targets=additional_targets)
                    
            r_angle = random.uniform(0, 45) # random rotate 0-45 degree
            
            transform_2 = A.Compose(
                    [ 
                    A.LongestMaxSize(max_size=self.input_size[0]),
                        A.PadIfNeeded(min_height=self.input_size[0], min_width=self.input_size[1], \
                                      border_mode=cv2.BORDER_CONSTANT,),
                    A.RandomRotate90(),
                    #A.Rotate(limit = 15, border_mode = cv2.BORDER_CONSTANT),
                    A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                    A.Rotate(limit = (r_angle, r_angle) , border_mode = cv2.BORDER_CONSTANT, p=1), # dual input: rotate r_angle degree
                    A.Normalize(self.mean, self.std),
                    ToTensorV2(),
                    ], additional_targets=additional_targets)
                    
                    
        
        if not self.t_resize:
            t_resize = None

        ref_return = self.transform_data(img, ref_gt, transform, True, t_resize)
        if self.split not in ['test', 'val', 'all']:
            ref_return_2 = self.transform_data(img, ref_gt, transform_2, True, t_resize)

            x1_rotated, y1_rotated, x2_rotated, y2_rotated = ref_return_2['rotate'][0]
            x1,y1,x2,y2 = 100,100,150,150
            vec_before = np.array([x2 - x1, y2 - y1])
            vec_after = np.array([x2_rotated - x1_rotated, y2_rotated - y1_rotated])
            angle_rad = np.arccos(np.dot(vec_before, vec_after) / (np.linalg.norm(vec_before) * np.linalg.norm(vec_after)))
            # ������ת��Ϊ�Ƕ�
            angle_deg = np.degrees(angle_rad)
        

        rot_return = self.transform_data(img, rot_gt, transform, False, t_resize)

        if self.get_theta:
            if rot_gt is not None:
                rot_a_lbl = self.process_theta_rot(rot_return['a_lbl'])
            if ref_gt is not None:
                axis, axis_lbl, axis_coords = ref_return['axis'], ref_return['axis_lbl'], ref_return['axis_coords']
                if len(axis_coords) == 0:
                    ref_a_lbl = torch.zeros(self.n_theta, axis.shape[-2], axis.shape[-1])
                else:
                    ref_a_lbl = self.process_theta_ref(axis_lbl, axis_coords).squeeze(0)

        if self.sym_type == 'reflection':
            if self.split in ['test', 'val', 'all']:
                # 测试集返回这个
                return ref_return['img'], ref_return['mask'], ref_return['axis'], ref_return['axis_gs'], False, ref_a_lbl, 0
            else : 
                return [
                [ref_return['img'], ref_return['mask'], ref_return['axis'], ref_return['axis_gs'], False, ref_a_lbl, r_angle+angle_deg],
                [ref_return_2['img'], ref_return_2['mask'], ref_return_2['axis'], ref_return_2['axis_gs'], False, ref_a_lbl, r_angle+angle_deg]
                ]
        elif self.sym_type == 'rotation':
            return rot_return['img'], rot_return['mask'], rot_return['axis'], rot_return['axis_gs'], False, rot_a_lbl
        elif self.sym_type == 'joint':
            ref_return = ref_return['img'], ref_return['mask'], ref_return['axis'], ref_return['axis_gs'], False, ref_a_lbl
            rot_return = rot_return['img'], rot_return['mask'], rot_return['axis'], rot_return['axis_gs'], False, rot_a_lbl
            return ref_return, rot_return


class CustomSymmetryDatasets(Dataset):
    def __init__(self, input_size=(417, 417), root='./demo/img'):
        super(CustomSymmetryDatasets, self).__init__()
        self.input_size = input_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.angle_interval = 45
        self.n_theta = 8
        self.img_list = self.get_img_list(root)

    def get_img_list(self, root_dir):
        img_names = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg"):
                    img_names.append(os.path.join(root, file))
        return img_names
            
    def __getitem__ (self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path).convert('RGB')
        img = match_input_type(img)

        transform = A.Compose([
                        A.Normalize(self.mean, self.std),
                        ToTensorV2(),
                        ])
        
        t_img = transform(image = img)["image"]
        return t_img, img_path

    def __len__(self):
        return len(self.img_list)


###############################################
# Ohter datasets
###############################################


class SymmetryDatasets(Dataset):
    def __init__(self, root='./sym_datasets', dataset=['SYM_LDRS'], split='train', resize=False, input_size=[513, 513], angle_interval=45, **kwargs):
        super(SymmetryDatasets, self).__init__()
        self.split = split
        self.resize = resize
        self.input_size = input_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.angle_interval = angle_interval
        self.n_angle = (int)(360 / self.angle_interval)
        
        self.length = 0 
        self.dataset_names = ['SYM_NYU', 'SYM_LDRS', 'SYNTHETIC_COCO', 'SYM_SDRW']
        self.datasets = []
        for name in dataset:
            if name not in self.dataset_names: continue

            if 'SYN' in name:
                data_set_cls = SYNTHETIC_COCO(split)
            else:
                data_set_cls = RealSymmetryDatasets(split, name)

            print(name, data_set_cls.__len__())
            self.length += data_set_cls.__len__()
            self.datasets.append((name, data_set_cls))
            
    def __getitem__(self, index):
        for name, dataset in self.datasets:
            if index >= dataset.__len__():
                index -= dataset.__len__()
                continue
            if 'SYN' in name:
                img, axis, mask = dataset.__getitem__(index)
                is_syn = True
            else:
                img, axis, axis_lbl, axis_coords = dataset.__getitem__(index)
                is_syn = False
            break

        axis_gs = cv2.GaussianBlur(axis, (5,5), cv2.BORDER_DEFAULT)

        if self.split == 'test' or self.split =='val':
            transform = A.Compose(
                        [ A.Normalize(self.mean, self.std),
                          ToTensorV2(),
                        ], additional_targets={'axis': 'mask', 'axis_gs': 'mask'})
        else:
            transform = A.Compose(
                    [ A.Resize(height=self.input_size[0], width=self.input_size[1]),
                    A.RandomRotate90(),
                    A.Rotate(limit = 15, border_mode = cv2.BORDER_CONSTANT),
                    #A.RandomCrop(height=self.input_size[0], width=self.input_size[1]),
                    A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                    A.Normalize(self.mean, self.std),
                    ToTensorV2(),
                    ], additional_targets={'axis': 'mask', 'axis_gs': 'mask', 'axis_lbl': 'mask', 'axis_coords': 'bboxes'})
        
        
        if is_syn:
            t = transform(image = img, mask = mask, axis = axis, axis_gs = axis_gs)
            img, mask, axis, axis_gs = t["image"], t["mask"], t["axis"], t["axis_gs"]
        else:
            if self.split == 'test' or self.split =='val':
                t_resize = A.Compose([A.Resize(height=self.input_size[0], width=self.input_size[1])])
                img = t_resize(image = img)["image"]
            t = transform(image = img, axis = axis, axis_gs = axis_gs, axis_lbl=axis_lbl, axis_coords=axis_coords)
            img, axis, axis_gs, axis_lbl, axis_coords = t["image"], t["axis"], t["axis_gs"], t["axis_lbl"], t["axis_coords"]
            mask = torch.zeros_like(axis)
        
        mask = mask.unsqueeze(0)
        axis = axis.unsqueeze(0)
        axis_gs = axis_gs.unsqueeze(0)
        axis_gs = axis_gs / (axis_gs.max() + 1e-5)

        ref_a_lbl = torch.zeros(8, axis.shape[-2], axis.shape[-1])
        return img, mask, axis, axis_gs, is_syn, ref_a_lbl, 0
        
    def __len__(self):
        return self.length
    
class RealSymmetryDatasets(Dataset):
    def __init__(self, split, dataset, root='./sym_datasets'):
        super(RealSymmetryDatasets, self).__init__()
        self.root = root
        self.get_data_list(dataset, split)

    def get_data_list(self, dataset, split):
        if dataset in ['SYM_NYU', 'NYU']:
            self.img_list, self.gt_list = self.nyu_get_data_list()
        elif dataset in ['SYM_LDRS', 'LDRS']:
            self.img_list, self.gt_list = self.ldrs_get_data_list(split)
        elif dataset in ['SYM_SDRW', 'sdrw']:
            self.img_list, self.gt_list = self.sdrw_get_data_list(split)
        
    def nyu_load_gt(self, path):
        coords = io.loadmat(path)['segments'][0]
        coords = [coord.ravel() for coord in coords]
        return coords
    
    def nyu_get_data_list(self):
        single_path = os.path.join(self.root, 'NYU', 'S')
        multi_path = os.path.join(self.root, 'NYU', 'M')
        single_names = ['I%.3d' % i for i in range(1, 176 + 1)]
        multi_names = ['I0%.2d' % i for i in range(1, 63 + 1)]
        single_img_list = [os.path.join(single_path, single + '.png') for single in single_names]
        multi_img_list = [os.path.join(multi_path, multi + '.png') for multi in multi_names]
        single_gt_list = [self.nyu_load_gt(os.path.join(single_path, single + '.mat')) for single in single_names]
        multi_gt_list = [self.nyu_load_gt(os.path.join(multi_path, multi + '.mat')) for multi in multi_names]

        dataset = {}
        for i, (img, gt) in enumerate(zip(single_img_list + multi_img_list, single_gt_list + multi_gt_list)):
            dataset[img] = gt
                
        img_list = list(dataset.keys())
        gt_list = list(dataset.values())
        
        return img_list, gt_list

    def ldrs_get_data_list(self, split):
        LDRS_path = os.path.join(self.root, 'LDRS', split)
        files = os.listdir(LDRS_path)
        json_files = list(filter(lambda x: x.find('.json') != -1, files))
        
        img_list = []
        gt_list = []
        
        for file in json_files:
            json_path = os.path.join(LDRS_path, file)
            img_path = json_path.rstrip('.json') + '.jpg'
            with open(json_path) as json_file:
                json_data = json.load(json_file)
                gt_list.append([(axis['points'][0][0], axis['points'][0][1], axis['points'][1][0], axis['points'][1][1]) for axis in json_data['shapes']])
            img_list.append(img_path)
        return img_list, gt_list

    def sdrw_get_data_list(self, split):
        if split in ['test', 'val']:
            cvpr_path = os.path.join(self.root, 'SDRW')
            single_path = os.path.join(cvpr_path, 'reflection_testing', 'single_testing')
            multi_path = os.path.join(cvpr_path, 'reflection_testing', 'multiple_testing')
            cvpr_gt = io.loadmat(os.path.join(cvpr_path, 'reflectionTestGt2013.mat'))
            single_gt = cvpr_gt['gtS']
            multi_gt = cvpr_gt['gtM']

            single_img_list = [os.path.join(single_path, single[0][0].strip()) for single in single_gt[0][0][0]]
            single_img_list[5] = single_img_list[5][:-4] + '.png'
            single_gt_list = [single for single in single_gt[0][0][1]]
            multi_img_list = [os.path.join(multi_path, multi[0][0]) for multi in multi_gt[0][0][0]]
            multi_gt_list = [multi for multi in multi_gt[0][0][1]]
            multi_img_list[0] = multi_img_list[0][:-4] + '.png'
            multi_img_list[1] = multi_img_list[1][:-4] + '.png'
            
            dataset = {}

            for i, (img, gt) in enumerate(zip(single_img_list + multi_img_list, single_gt_list + multi_gt_list)):
                if img in dataset: 
                    dataset[img].append(gt)
                else:
                    dataset[img] = [gt]
            
        else:
            cvpr_path = os.path.join(self.root, 'SDRW', 'reflection_training')
            single_path = os.path.join(cvpr_path, 'single_training')
            multi_path = os.path.join(cvpr_path, 'multiple_training')
            single_gt = io.loadmat(os.path.join(cvpr_path, 'singleGT_training', 'singleGT_training.mat'))['gt']
            multi_gt = io.loadmat(os.path.join(cvpr_path, 'multipleGT_training', 'multipleGT_training.mat'))['gt']

            single_img_list = [os.path.join(single_path, single[0][0].strip()) for single in single_gt['name']]
            multi_img_list = [os.path.join(multi_path, multi[0][0].strip()) for multi in multi_gt['name']]
            single_gt_list = [single[0][0] for single in single_gt['ax']]
            multi_gt_list = [multi[0] for multi in multi_gt['ax']]

            dataset = {}

            for img, gt in zip(single_img_list, single_gt_list):
                if img in dataset: 
                    dataset[img].append(gt)
                else:
                    dataset[img] = [gt]

            for img, gt in zip(multi_img_list, multi_gt_list):
                for _gt in gt:
                    if img in dataset:
                        dataset[img].append(_gt)
                    else:
                        dataset[img] = [_gt]
                
        img_list = list(dataset.keys())
        gt_list = list(dataset.values())
        
        return img_list, gt_list
    
    def __getitem__ (self, index):
        img = self.img_list[index]
        img = Image.open(img).convert('RGB')
        axis = self.gt_list[index]
        axis, line_coords = draw_axis(axis, img.size)
        img = match_input_type(img)
        _axis = (axis > 0).astype(axis.dtype)
        return img, _axis, axis, line_coords # 返回图像、轴线标签、轴线坐标、轴线坐标
        # return img, _axis, axis, _axis, False, None, 0
    
    def __len__(self):
        return len(self.img_list)

class SYNTHETIC_COCO(Dataset):
    def __init__(self, split, root='./coco_data', year='2014'):
        super(SYNTHETIC_COCO, self).__init__()
        
        self.root = root
        self.nangle = 2
        self.ninst = 10
        self.data_list = self.get_data_list(split, year)
        
    def get_data_list(self, split, year):
        ann_file = os.path.join(self.root, 'annotations', f"instances_{split}{year}.json")
        ids_file = os.path.join(self.root, 'annotations', f"{split}_ids.mx")
        self.root = os.path.join(self.root, f"{split}{year}")
        self.coco = COCO(ann_file)
        self.ids_all = list(self.coco.imgs.keys())
        if os.path.exists(ids_file):
            with open(ids_file, 'rb') as f:
                ids = pickle.load(f)
        else:
            ids = self._preprocess(self.ids_all, ids_file)
        
        return ids
    
    def get_rot_matrices (self, h, w):
        center = (w/2, h/2)
        rand_ang = np.random.randint(180 / self.nangle, size = self.nangle)
        angles = np.linspace(start = -90, stop = 90, num = self.nangle, endpoint= False) + rand_ang
        rot_m = []
        unrot_m = []
        for angle in angles:
            m = cv2.getRotationMatrix2D(center, angle, 1.0)
            un_m = cv2.getRotationMatrix2D(center, -angle, 1.0)
            rot_m.append(m)
            unrot_m.append(un_m)
        rot_m = np.stack(rot_m, axis=0) # ang x 2 x 3
        unrot_m = np.stack(unrot_m, axis=0) # ang x 2 x 3
        return rot_m, unrot_m, angles
        
    def get_rot_anns(self, ann, h, w):
        rot_ann = []
        
        ann = sorted(ann, key=lambda x: -x['area'])
        for idx, instance in enumerate(ann):
            rot_m, unrot_m, angles = self.get_rot_matrices(h, w)
            rot_instances = []
            xmin, ymin = 1e3 * np.ones(self.nangle), 1e3 * np.ones(self.nangle)
            xmax, ymax = np.zeros(self.nangle), np.zeros(self.nangle)
            
            ## 'iscrowd' option change 'segmentation' in other formats, thus bypass the instance having this option
            if instance['iscrowd']: 
                continue
            for seg in instance['segmentation']:
                ## rotate contour points for all angles.
                points = np.asarray(seg).reshape(-1, 2)
                ones = np.ones(shape=(len(points), 1))
                points_ones = np.hstack([points, ones])
                rot_points = np.matmul(rot_m, points_ones.T).transpose(1, 2, 0) # (x,y) x points x ang
                
                ## limit boundaries
                rot_points = np.where(rot_points < 0, 0, rot_points)
                rot_points[0] = np.where(rot_points[0] >= w - 1, w - 1, rot_points[0])
                rot_points[1] = np.where(rot_points[1] >= h - 1, h - 1, rot_points[1])
            
                ## integrate boundaries into one segment
                xmin_inst, xmax_inst, ymin_inst, ymax_inst = np.min(rot_points[0], 0), np.max(rot_points[0], 0), np.min(rot_points[1], 0), np.max(rot_points[1], 0)
                xmin, ymin = np.minimum(xmin, xmin_inst), np.minimum(ymin, ymin_inst)
                xmax, ymax = np.maximum(xmax, xmax_inst), np.maximum(ymax, ymax_inst)
                rot_instances.append(rot_points.transpose(2, 1, 0))
            
            ## filtering & set cut axis by random
            avail_ang = (xmax - xmin + 1) * (ymax - ymin)> h*w/16
            cut_axis = (xmin + np.random.uniform(1/3, 2/3) * (xmax - xmin)).astype(int)
            for i in np.argwhere(avail_ang):
                i = i[0]
                ## flip points that are left from cut axis, but right from the limitx.
                flip_instances = []
                limitx = cut_axis[i] - (w - cut_axis[i])
                for rot_points in rot_instances:
                    npoints = len(rot_points[i,:,0])
                    is_left = rot_points[i,:,0] < cut_axis[i]
                    ind_left = np.where(np.diff(np.where(is_left)[0]) > 1)[0] + 1
                    left_points = np.split(rot_points[i,is_left,:], ind_left, axis=0)
                    
                    if len(left_points[0]) == 0 or len(left_points[0]) == npoints: continue
                    if (is_left[0] == True) and (is_left[-1] == True):
                        left_points[0] = np.concatenate((left_points[-1], left_points[0]), axis=0)
                        del left_points[-1]
                    for lpoints in left_points:
                        lpoints[:,0] = np.where(lpoints[:,0]<limitx, limitx, lpoints[:,0])
                        right_points = np.stack((lpoints[:,0] + 2 * (cut_axis[i] - lpoints[:,0]), lpoints[:,1]), axis=1)
                        flip_points = np.concatenate((lpoints, np.flip(right_points, 0)), axis=0)
                        
                        if len(flip_points) <= 2:
                            continue
                        flip_instances.append(flip_points)
                if len(flip_instances) == 0:
                    continue
                
                ## make coco-style instance after un-rotating flipped instance.
                instance_copy = instance.copy()
                instance_copy['segmentation'] = []
                instance_copy['angle'] = angles[i]
                instance_copy['cutaxis'] = cut_axis[i]
                
                only_flip_pos = unrot_pos = distort_pos = np.ones([4]) * -1e3
                for flip_points in flip_instances:
                    ones = np.ones(shape=(len(flip_points), 1))
                    flip_points_ones = np.hstack([flip_points, ones])
                    unrot_points = np.matmul(unrot_m[i], flip_points_ones.T).T # points x (x,y)
                                        
                    unrot_points = np.where(unrot_points < 0, 0, unrot_points)
                    unrot_points[:, 0] = np.where(unrot_points[:, 0] >= w - 1, w - 1, unrot_points[:, 0])
                    unrot_points[:, 1] = np.where(unrot_points[:, 1] >= h - 1, h - 1, unrot_points[:, 1])
                    
                    flip_points[:, 0] = np.where(flip_points[:, 0] >= w - 1, w - 1, flip_points[:, 0])
                
                    instance_copy['segmentation'].append(unrot_points.flatten())
                    
                    bound_unrot_inst = -np.min(unrot_points[:, 0], 0), np.max(unrot_points[:, 0], 0), -np.min(unrot_points[:, 1], 0), np.max(unrot_points[:, 1], 0) 
                    bound_flip_inst = -np.min(flip_points[:, 0], 0), np.max(flip_points[:, 0], 0), -np.min(flip_points[:, 1], 0), np.max(flip_points[:, 1], 0) 
                    unrot_pos = np.maximum(unrot_pos, bound_unrot_inst)
                    only_flip_pos = np.maximum(only_flip_pos, bound_flip_inst)
                
                unrot_pos = np.ceil(unrot_pos).astype(int)
                only_flip_pos = np.ceil(only_flip_pos).astype(int)
                unrot_bbox = np.stack([-unrot_pos[0], -unrot_pos[2], (unrot_pos[0] + unrot_pos[1] + 1), (unrot_pos[2] + unrot_pos[3] + 1)], 0)
                only_flip_bbox = np.stack([-only_flip_pos[0], -only_flip_pos[2], (only_flip_pos[0] + only_flip_pos[1] + 1), (only_flip_pos[2] + only_flip_pos[3] + 1)], 0)
                instance_copy['bbox'] = unrot_bbox
                instance_copy['only_flip_bbox'] = only_flip_bbox
                rot_ann.append(instance_copy)
            
            if (len(rot_ann) >= self.ninst): break
        
        return rot_ann
    
    def rot_flip_unrot_merge (self, img, sym_ann):
        merged_img = np.asarray(img).copy()
        for seg in sym_ann:
            sym_img = np.asarray(img.rotate(seg['angle'], resample=PIL.Image.BILINEAR)).copy()
            bbox = seg['only_flip_bbox']
            sym_img[bbox[1]:bbox[1]+bbox[3], bbox[0]+bbox[2]-1:seg['cutaxis']:-1] = sym_img[bbox[1]:bbox[1]+bbox[3], seg['cutaxis'] - (bbox[0] + bbox[2] - 1 - seg['cutaxis']):seg['cutaxis']]
            
            sym_img = Image.fromarray(sym_img).rotate(-seg['angle'], resample=PIL.Image.BILINEAR)
            sym_img = np.asarray(sym_img)
            
            bbox = seg['bbox']
            merged_img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = sym_img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        return merged_img
    
    def construct_graph (self, rot_ann):
        ## Get maximally occupied region by DFS.
        E = np.zeros((len(rot_ann), len(rot_ann))).astype(bool)
        for i in range(1, len(rot_ann)):
            for j in range(i):
                flag = 0
                if (rot_ann[i]['bbox'][0] + rot_ann[i]['bbox'][2] < rot_ann[j]['bbox'][0]): flag = 1 
                if (rot_ann[i]['bbox'][0] > rot_ann[j]['bbox'][0] + rot_ann[j]['bbox'][2]): flag = 1 
                if (rot_ann[i]['bbox'][1] + rot_ann[i]['bbox'][3] < rot_ann[j]['bbox'][1]): flag = 1 
                if (rot_ann[i]['bbox'][1] > rot_ann[j]['bbox'][1] + rot_ann[j]['bbox'][3]): flag = 1 
                if flag :
                    E[i][j]=True
                    E[j][i]=True
        return E
    
    def get_sym_info (self, rot_ann, h, w, preprocess = False):
        sym_axis = np.zeros((h, w))
        sym_axis_gs = np.zeros((h, w))
        sym_mask =  np.zeros((h, w), dtype=np.float32)
        inst_axis = []
        inst_mask = []
        E = self.construct_graph(rot_ann)
        S = np.zeros((len(rot_ann)))
        
        for i, seg in enumerate(rot_ann):
            rle = coco_mask.frPyObjects(seg['segmentation'], h, w)
            mask = coco_mask.decode(rle)
            
            if len(mask.shape) >= 3:
                mask = ((np.sum(mask, axis=2)) > 0).astype(np.uint8)
                
            axis = np.zeros((h, w))
            bbox = seg['only_flip_bbox']
            axis[bbox[1]:bbox[1]+bbox[3], seg['cutaxis'] : seg['cutaxis'] + 1] = 1.0
            
            axis = np.asarray(Image.fromarray(axis).rotate(-seg['angle'], resample=PIL.Image.NEAREST))
            axis = np.logical_and(axis > 0, mask > 0).astype(np.float)
            axis_length = axis.sum()
            
            if axis_length < 0.8 * bbox[3]: axis_length = -1
            S[i] = axis_length
            
            inst_axis.append(axis)
            inst_mask.append(mask)
                
        dfs = DFS(E, S)
        max_state = dfs.forward()
        
        if preprocess:
            if len(S) == 0:
                return 0, 0
            return dfs.max_score, dfs.max_visited
        else:
            sym_ann = [rot_ann[i] for i in max_state]
            inst_axis = [inst_axis[i] for i in max_state]
            inst_mask = [inst_mask[i] for i in max_state]
            
            for idx, (mask, axis) in enumerate(zip(inst_mask, inst_axis)):
                sym_axis += axis
                sym_mask += mask * (1 + idx)
            return sym_ann, sym_axis, sym_mask
    
    def blend_img(self, fg, bg, mask):
        fg = (fg / 255.0).astype(np.float32)
        bg = (bg / 255.0).astype(np.float32)
        
        mean_fg, std_fg = fg.mean(axis=(0, 1), keepdims=True), fg.std(axis=(0, 1), keepdims=True)
        mean_bg, std_bg = bg.mean(axis=(0, 1), keepdims=True), bg.std(axis=(0, 1), keepdims=True)

        norm_fg = (fg - mean_fg) / std_fg
        norm_fg = (norm_fg * std_bg) + mean_bg

        mask = np.expand_dims((mask > 0), -1).astype(np.float)
        alpha = np.expand_dims(cv2.GaussianBlur(mask, (5, 5), cv2.BORDER_DEFAULT), -1)
        alpha_blend = alpha * (norm_fg) + (1 - alpha) * bg
        alpha_blend = np.clip((alpha_blend * 255.0).astype(np.int32), 0, 255) 
        return alpha_blend.astype(np.uint8)
    
    def resize_bg_as_fg (self, bg, fg_shape):
        self.transform = A.Compose(
                        [ A.SmallestMaxSize(max_size=max(fg_shape)+100),
                          A.RandomCrop(height=fg_shape[0], width=fg_shape[1])])
        t = self.transform(image = bg)
        return t["image"]
    
    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while." + 
              "But don't worry, it only run once for each split.")
        coco = self.coco
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            ann = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
            img_metadata = coco.loadImgs(img_id)[0]
            h, w = img_metadata['height'], img_metadata['width']
            
            rot_ann = self.get_rot_anns(ann, h, w)
            sym_length_all, nvisit = self.get_sym_info(rot_ann, h, w, True)
            if sym_length_all >= (h + w) / 2 and nvisit > 2:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        with open(ids_file, 'wb') as f:
            pickle.dump(new_ids, f)
        return new_ids
    
    
    def __getitem__(self, index):
        coco = self.coco
        n = len(self.ids_all)
        bg_id = self.ids_all[random.randint(0, n - 1)]
        bg_metadata = coco.loadImgs(bg_id)[0]
        bg_path = bg_metadata['file_name']
        bg = Image.open(os.path.join(self.root, bg_path)).convert('RGB')
        bg = np.asarray(bg)
        
        img_id = self.data_list[index]
        img_metadata = coco.loadImgs(img_id)[0]
        img_path = img_metadata['file_name']
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        h, w = img_metadata['height'], img_metadata['width']
        
        ann = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        rot_ann = self.get_rot_anns(ann, h, w)
        sym_ann, sym_axis, sym_mask = self.get_sym_info(rot_ann, h, w)
        fg = self.rot_flip_unrot_merge(img, sym_ann)
        bg = self.resize_bg_as_fg(bg, fg.shape)
        sym_img = self.blend_img(fg, bg, sym_mask)
        
        return sym_img, sym_axis, sym_mask
    
    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':
    import torch.utils.data as data
    
    trainset = COCOSegmentation()
    #trainset.__getitem__(17)
    train_data = data.DataLoader(
        trainset, batch_size=16, shuffle=False,
        num_workers=4)
    
    for i, data in enumerate(tqdm(train_data)):
        img, mask, axis = preprocess_batch(data)
        for j in range(len(img)):
             plt.imshow(transforms.ToPILImage()(img[j]))
             plt.show()
             plt.imshow(transforms.ToPILImage()(mask[j]))
             plt.show()
