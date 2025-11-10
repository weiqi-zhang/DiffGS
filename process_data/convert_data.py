import argparse

import numpy as np
from plyfile import PlyData, PlyElement
import torch
import os
import torch
import random
import numpy as np
import traceback
from multiprocessing import Pool
from fnmatch import fnmatch
from scipy.spatial import cKDTree
import multiprocessing as mp
import json

def process(paths):
    path, iiidx, sou, save_path = paths
    
    print(iiidx, ' ', path)

    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))

    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    _xyz = torch.tensor(xyz, dtype=torch.float, device="cpu").requires_grad_(True)
    _features_dc = torch.tensor(features_dc, dtype=torch.float, device="cpu").transpose(1, 2).contiguous().requires_grad_(True)
    _features_rest = torch.tensor(features_extra, dtype=torch.float, device="cpu").transpose(1, 2).contiguous().requires_grad_(True)
    _opacity = torch.tensor(opacities, dtype=torch.float, device="cpu").requires_grad_(True)
    _scaling = torch.tensor(scales, dtype=torch.float, device="cpu").requires_grad_(True)
    _rotation = torch.tensor(rots, dtype=torch.float, device="cpu").requires_grad_(True)

    # 写入
    xyz = _xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = _features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = _features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = _opacity.detach().cpu().numpy()
    scale = _scaling.detach().cpu().numpy()
    rotation = _rotation.detach().cpu().numpy()

    data = np.concatenate([xyz, f_dc, f_rest, opacities, scale, rotation], axis=-1).reshape(-1, 59)


    occ = np.ones((1_048_576, 4))

    
    zero_num = 100_000 * 2
    occ[:100_000,:3] = xyz
    occ[100_000:zero_num,:3] = xyz

    sample_near_num = int((1_048_576 - zero_num) * 0.8)
    sample_far_num = 1_048_576 - sample_near_num - zero_num

    sample = []
    sample_far = []
    ptree = cKDTree(xyz)
    sigmas = []
    for p in np.array_split(xyz, 100, axis=0):
        d = ptree.query(p, 151)
        sigmas.append(d[0][:,-1])
    sigmas = np.concatenate(sigmas)
    
    
    POINT_NUM = xyz.shape[0] // 60
    POINT_NUM_GT = xyz.shape[0] // 60 * 60
    QUERY_EACH = sample_near_num // xyz.shape[0] + 5
    for i in range(QUERY_EACH):
        # scale = 0.25 if 0.25 * np.sqrt(POINT_NUM_GT / 20000) < 0.25 else 0.25 * np.sqrt(POINT_NUM_GT / 20000)
        tt = xyz + np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=xyz.shape)
        # tt = xyz + np.random.normal(0.0, 0.1, size = xyz.shape)
        for j in range(100_000):
            dis, _ = ptree.query(tt[j], k = 1)
            if dis < 0.2 :
                tmp = np.zeros(4)
                tmp[:3] = tt[j]
                tmp[3] = (0.2 - dis) / 0.2 
                sample.append(tmp)
                if len(sample) > sample_near_num:
                    break
        if len(sample) > sample_near_num:
            break
    sample = np.asarray(sample[:sample_near_num])
    occ[zero_num:zero_num + sample_near_num] = sample
    

    bbox_min = -0.7  # 在批次和点的维度上找全局最小
    bbox_max = 0.7  # 在批次和点的维度上找全局最大
    space_samples = np.random.uniform(bbox_min, bbox_max, size=(sample_far_num * 3, 3))

    for j in range(space_samples.shape[0]):
        dis, _ = ptree.query(space_samples[j], k = 1)
        if dis > 0.2:
            tmp = np.zeros(4)
            tmp[:3] = space_samples[j] 
            sample_far.append(tmp)
            if len(sample_far) > sample_far_num:
                break
        if len(sample_far) > sample_far_num:
            break
    sample_far = np.asarray(sample_far[:sample_far_num])
    occ[zero_num + sample_near_num:] = sample_far
    occ = occ.reshape(1_048_576, 4)

    data_path = os.path.join(save_path, str(iiidx))
    os.makedirs(data_path, exist_ok=True)

    np.save(os.path.join(data_path, 'gaussian.npy'), data)
    np.save(os.path.join(data_path, 'occ.npy'), occ)

    return 



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="批量处理高斯溅射（Gaussian Splatting）训练结果，生成用于后续任务的 .npy 文件。"
    )
    parser.add_argument(
        "-s", "--source",
        type=str,
        required=True,
        help="包含训练结果的根目录 (例如: ./output)"
    )
    parser.add_argument(
        "-t", "--target",
        type=str,
        default="./trainset",
        help="用于保存处理后 .npy 文件的根目录。如果未指定，则默认将 .npy 文件保存在其原始物体目录中。"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=20,
        help=f"用于并行处理的进程数。默认为20。"
    )
    args = parser.parse_args()

    save_gaussian_folder = os.path.abspath(args.source)
    save_path = os.path.abspath(args.target)

    pattern = "*.ply"
    paths = []
    i = 0
    for path, subdirs, files in os.walk(save_gaussian_folder):
        for name in files:
            t = path.split('/')[-1]
            if fnmatch(name, pattern) and t == 'iteration_30000':
                with open(os.path.join(path[:-28], 'source.txt')) as f:
                    sou = f.readline().split('/')[-1]
                    paths.append((os.path.join(path, name), i, sou, save_path))
                    i = i + 1
    print(f"{len(paths)} left to be processed!")


    pool = mp.Pool(args.workers)
    pool.map(process, paths)
