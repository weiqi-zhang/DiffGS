import argparse
import os
import subprocess
import multiprocessing
from multiprocessing import Pool
from fnmatch import fnmatch

def render_obj(arg):
    # 接收参数
    obj_path, relative_path, output_base = arg

    # 1. 获取 OBJ 文件所在目录的相对路径 (例如: 02773838/10a.../models)
    relative_models_dir = os.path.dirname(relative_path)

    # 2. 获取 OBJ 文件 ID 目录的相对路径 (向上移动一层)
    # 目标路径: 02773838/10a885f5971d9d4ce858db1dc3499392/
    relative_id_dir = os.path.dirname(relative_models_dir)

    # 3. 拼接得到最终的输出根目录 (Blender 脚本将把 images/ 和 transforms 放入此目录)
    # 期望结果: /root/autodl-tmp/ShapeNetCoreImages/02773838/10a.../
    output_folder = os.path.join(output_base, relative_id_dir)

    # 确保目标输出目录存在 (必须在运行 Blender 前创建)
    # 注意：如果目录在 worker 进程中创建，可能会有轻微的竞争条件，
    # 但由于我们在主进程中已经跳过了已存在的目录，这里只是双重保险。
    os.makedirs(output_folder, exist_ok=True)

    print(f"Rendering {obj_path} -> Outputting to {output_folder}")

    cmd = [
        'xvfb-run', '-a', 'blender', '--background', '--python', 'render_blender.py', '--',
        '--output_folder', output_folder,  # 传入最终的输出根路径
        obj_path
    ]

    # 注意：在实际运行中，您可能需要将 blender 命令替换为它的完整路径 (如 /usr/local/bin/blender29)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ 渲染完成: {obj_path}")
    except subprocess.CalledProcessError as e:
        # 如果子进程（Blender/xvfb）报错，记录错误信息
        print(f"❌ 渲染失败: {obj_path}")
        print(f"  错误信息: {e.stderr}")
    except FileNotFoundError:
        print("❌ 错误: 找不到 'blender' 或 'xvfb-run' 命令。请检查您的系统PATH或使用绝对路径。")
    # 也可以添加更多的错误处理...


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="render images from 3D models in a directory.")
    parser.add_argument("-s", "--source", type=str, required=True,
                        help="Path to the root directory of the ShapeNet dataset (e.g., ShapeNetCore).")
    parser.add_argument("-t", "--target", type=str, default=None,
                        help="Path to the target directory of the rendered images.")
    parser.add_argument("-w", "--workers", type=int, default=20,
                        help=f"Number of worker processes for parallel processing. Defaults to the number of CPU cores ({multiprocessing.cpu_count()}).")
    args = parser.parse_args()
    shapenet_folder = args.source
    if args.target is not None:
        output_base = args.source
    else:
        output_base = args.target
    num_workers = args.workers

    pattern = "*.obj"
    tasks = []
    skipped_count = 0

    # 确保路径以斜杠结尾，方便后续处理
    if not shapenet_folder.endswith(os.sep):
        shapenet_folder += os.sep

    for path, subdirs, files in os.walk(shapenet_folder):
        for name in files:
            if fnmatch(name, pattern):
                obj_path = os.path.join(path, name)
                # 计算相对路径
                relative_path = obj_path.replace(shapenet_folder, '')

                # 确定目标输出目录路径
                relative_models_dir = os.path.dirname(relative_path)
                relative_id_dir = os.path.dirname(relative_models_dir)
                output_folder = os.path.join(output_base, relative_id_dir)

                # 检查目标目录是否已存在
                if os.path.exists(output_folder) and os.path.exists(os.path.join(output_folder, 'transforms_test.json')):
                    # 简化处理：根据用户要求，只要目标目录存在就跳过
                        skipped_count += 1
                        continue  # 跳过这个OBJ文件

                # 如果未跳过，则添加到任务列表
                tasks.append((obj_path, relative_path, output_base))

    print(f"--- 任务统计 ---")
    print(f"已发现 {len(tasks) + skipped_count} 个 OBJ 文件。")
    print(f"跳过已渲染的 {skipped_count} 个文件。")
    print(f"将要渲染 {len(tasks)} 个文件。")
    print(f"------------------")

    pool = Pool(num_workers)
    pool.map(render_obj, tasks)