import argparse
import trimesh
import numpy as np
import os
import traceback
from multiprocessing import Pool
from fnmatch import fnmatch
import multiprocessing as mp
from plyfile import PlyData, PlyElement

# def sample(arg):
#     path, name = arg
#     mesh = trimesh.load_mesh(os.path.join(path, name))
#
#     num_points = 100000
#     points = mesh.sample(num_points)
#
#     point_cloud = trimesh.points.PointCloud(points)
#
#     save_path = os.path.join(path, 'points3d.ply')
#     point_cloud.export(save_path)

def sample(arg):
    path, name = arg
    full_path = os.path.join(path, name)

    # path 当前指向的是 models/ 目录
    # 例如: /root/autodl-tmp/ShapeNetCorePart/02773838/10a885f5971d9d4ce858db1dc3499392/models
    # target_dir 示例: /root/autodl-tmp/ShapeNetCorePart/02773838/10a885f5971d9d4ce858db1dc3499392
    target_dir = os.path.dirname(path)

    print(f"Processing: {full_path}")
    print(f"Saving to: {target_dir}")  # 打印目标目录，方便确认

    try:
        loaded_data = trimesh.load(full_path)
        mesh = None

        if isinstance(loaded_data, trimesh.Trimesh):
            mesh = loaded_data

        elif isinstance(loaded_data, trimesh.Scene):
            merged_mesh = loaded_data.dump(concatenate=True)
            if merged_mesh is not None:
                mesh = merged_mesh
                print(f"[{name}] 场景已合并为单一网格。")
            else:
                print(f"[{name}] 场景中无有效网格可合并，跳过。")
                return
        else:
            print(f"[{name}] 加载结果类型为 {type(loaded_data)}，跳过。")
            return

        if mesh is None:
            print(f"[{name}] 未能获取有效网格对象，跳过。")
            return

        num_points = 100000

        # 确保网格有面
        if mesh.faces.size == 0:
            print(f"[{name}] 网格无面，无法采样，跳过。")
            return

        # 使用 Trimesh.sample() 获取位置和面索引
        # faces_idx 是每个点采样自的面的索引
        points, faces_idx = mesh.sample(num_points, return_index=True)

        # 根据面索引获取每个采样点的法向量
        # mesh.face_normals[faces_idx] 提供了每个采样点所在面的法向量
        normals = mesh.face_normals[faces_idx]

        # 检查 normals 数组的状态
        print(f"[{name}] 准备导出... 检查法向量数组:")
        print(f"   - Normals 数组的形状: {normals.shape}")
        print(f"   - Normals 数组的数据类型: {normals.dtype}")
        if np.isnan(normals).any():
            print(f"   - 警告: Normals 数组中包含 NaN 值！")
        if normals.shape[0] != num_points:
            print(f"   - 错误: 法向量数量 ({normals.shape[0]}) 与点数 ({num_points}) 不匹配！")

        num_points = points.shape[0]
        vertex_data = np.empty(num_points, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')  # 颜色通常是 8位无符号整数 (0-255)
        ])

        # 填充数据
        vertex_data['x'] = points[:, 0].astype('f4')
        vertex_data['y'] = points[:, 1].astype('f4')
        vertex_data['z'] = points[:, 2].astype('f4')

        vertex_data['nx'] = normals[:, 0].astype('f4')
        vertex_data['ny'] = normals[:, 1].astype('f4')
        vertex_data['nz'] = normals[:, 2].astype('f4')

        # 检查 mesh 是否有可用的面部颜色
        if hasattr(mesh.visual, 'face_colors') and len(mesh.visual.face_colors) > 0:
            # faces_idx 是我们从 mesh.sample 得到的每个点所在面的索引
            # 我们可以直接用它来索引面部颜色
            print("- 网格具有面颜色，正在提取对应点的颜色。")
            point_colors = mesh.visual.face_colors[faces_idx]
        else:
            # 如果没有颜色，则使用默认的灰色
            print("- 警告: 网格无面颜色，使用默认。")
            point_colors = np.full((num_points, 4), [128, 128, 128,255], dtype=np.uint8)

        # 填充颜色数据 (注意 trimesh 颜色可能是 RGBA，我们只需要 RGB)
        vertex_data['red'] = point_colors[:, 0]
        vertex_data['green'] = point_colors[:, 1]
        vertex_data['blue'] = point_colors[:, 2]

        # 创建 PlyElement
        vertex_element = PlyElement.describe(vertex_data, 'vertex')

        # 构造 PlyData 并写入文件
        save_path = os.path.join(target_dir, 'points3d.ply')
        PlyData([vertex_element], text=True).write(save_path)

        print(f"[{name}] ✅ 成功采样 {num_points} 点 (含法向量)，并使用 plyfile 手动保存到 {save_path}。")
        print(f"   -> 文件应包含字段: {vertex_data.dtype.names}")

    except Exception as e:
        print(f"\n❌ ERROR processing {full_path}: {e}")
        # traceback.print_exc()
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample points with normals and colors from 3D models in a directory.")
    parser.add_argument("-s", "--source", type=str, required=True,
                        help="Path to the root directory of the ShapeNet dataset (e.g., ShapeNetCorePart).")
    parser.add_argument("-w", "--workers", type=int, default=20,
                        help=f"Number of worker processes for parallel processing. Defaults to the number of CPU cores ({mp.cpu_count()}).")
    args = parser.parse_args()
    shapenet_folder = args.source
    num_workers = args.workers

    pattern = "*.obj"
    tasks = []
    for path, subdirs, files in os.walk(shapenet_folder):
        for name in files:
            if fnmatch(name, pattern):
                # path: .../models
                # name: model_normalized.obj
                tasks.append((path, name))

    print(f"{len(tasks)} objects left to be processed!")

    with Pool(num_workers) as pool:
        pool.map(sample, tasks)
    print("\n🎉 全部处理完成！")