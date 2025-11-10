import argparse
import os
import subprocess
import sys

def find_trainable_scenes(root_path):
    """
    遍历根目录，寻找所有包含 `transforms_train.json` 的子目录。
    这些子目录被视为可训练的场景。
    """
    trainable_dirs = []
    if not os.path.isdir(root_path):
        print(f"❌ 错误: 提供的源路径 '{root_path}' 不是一个有效的目录。")
        return trainable_dirs

    print(f"正在扫描 '{root_path}' 中的可训练场景...")
    for dirpath, _, filenames in os.walk(root_path):
        if 'transforms_train.json' in filenames:
            trainable_dirs.append(dirpath)

    print(f"扫描完成，发现 {len(trainable_dirs)} 个可训练场景。")
    return trainable_dirs

def main():
    parser = argparse.ArgumentParser(
        description="批量训练 ShapeNetCore 数据集中的所有物体。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-s", "--source",
        type=str,
        required=True,
        help="ShapeNetCore 数据集的根目录路径。\n"
             "例如: /path/to/ShapeNetCoreTest (将训练所有类别)\n"
             "或: /path/to/ShapeNetCoreTest/02773838 (将训练该类别下的所有物体)"
    )
    parser.add_argument(
        "-t", "--target",
        type=str,
        default="./output",
        help="存放所有训练结果的根目录。默认为 './output'。"
    )
    args = parser.parse_args()

    source_root = os.path.abspath(args.source)
    output_root = os.path.abspath(args.target)

    # 判断输入的源路径是数据集根目录还是类别子目录，以确定计算相对路径的基准
    source_basename = os.path.basename(source_root)
    if source_basename.isdigit():
        # 如果输入的是一个类别目录 (如 '02773838'),
        # 则使用其父目录作为计算相对路径的基准
        path_base_for_relpath = os.path.dirname(source_root)
        print(f"检测到输入为类别目录，将使用 '{path_base_for_relpath}' 作为路径基准。")
    else:
        # 否则，输入路径本身就是基准
        path_base_for_relpath = source_root
        print(f"检测到输入为数据集根目录，将使用 '{path_base_for_relpath}' 作为路径基准。")


    # 1. 查找所有需要训练的场景
    scenes_to_train = find_trainable_scenes(source_root)

    if not scenes_to_train:
        print("未找到任何可训练的场景，程序退出。")
        sys.exit(0)

    # 2. 准备任务列表并过滤已完成的任务
    tasks = []
    skipped_count = 0

    for scene_path in scenes_to_train:
        # 使用修正后的基准路径来计算相对路径
        relative_path = os.path.relpath(scene_path, path_base_for_relpath)

        # 拼接得到最终的输出路径
        output_path = os.path.join(output_root, relative_path)

        # 检查是否已经训练过 (通过检查是否存在 point_cloud 子目录来判断)
        if os.path.exists(os.path.join(output_path, "point_cloud")):
            skipped_count += 1
            continue

        tasks.append({
            "source": scene_path,
            "target": output_path
        })

    print(f"\n--- 任务统计 ---")
    print(f"总共发现 {len(scenes_to_train)} 个场景。")
    print(f"已跳过 {skipped_count} 个已训练的场景。")
    print(f"即将开始训练 {len(tasks)} 个新场景。")
    print(f"------------------\n")

    # 3. 顺序执行训练任务
    for i, task in enumerate(tasks):
        source_path = task["source"]
        target_path = task["target"]

        print(f"--- [{i+1}/{len(tasks)}] 开始训练 ---")
        print(f"源路径: {source_path}")
        print(f"目标路径: {target_path}")

        os.makedirs(target_path, exist_ok=True)

        command = [
            'python',
            'train_gaussian.py',
            '-s', source_path,
            '--model_path', target_path
        ]

        print(f"执行命令: {' '.join(command)}")

        try:
            process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
            process.wait()

            if process.returncode == 0:
                print(f"✅ 训练成功: {source_path}\n")
            else:
                print(f"❌ 训练失败，返回码: {process.returncode}。源路径: {source_path}\n")

        except FileNotFoundError:
            print("❌ 致命错误: 找不到 'python' 命令。请确保 Python 环境已正确配置。")
            sys.exit(1)
        except Exception as e:
            print(f"❌ 训练过程中发生未知错误: {e}")
            print(f"   源路径: {source_path}\n")

    print("所有训练任务已完成！")

if __name__ == '__main__':
    main()