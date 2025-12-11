import os
import shutil
import sys
import argparse
import zipfile
from pathlib import Path

def get_user_confirmation(prompt_text):
    """
    获取用户确认 (y/n)
    """
    while True:
        choice = input(f"{prompt_text} (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            return True
        if choice in ['n', 'no']:
            return False
        print("输入无效，请输入 y 或 n。")

def format_size(size_bytes):
    """将字节转换为 MB"""
    return f"{size_bytes / (1024 * 1024):.2f} MB"

def process_archives(source_dir, target_dir, threshold_mb):
    src_path = Path(source_dir).resolve()
    dst_path = Path(target_dir).resolve()

    # 1. 检查源目录
    if not src_path.exists():
        print(f"[错误] 源目录不存在: {src_path}")
        return

    print("-" * 50)
    print(f"📂 源目录: {src_path}")
    print(f"📂 目标目录: {dst_path}")
    print(f"⚖️  筛选阈值: < {threshold_mb} MB")
    print("-" * 50)

    # ================== 阶段一：扫描与筛选 ==================
    print("正在扫描并筛选文件...")
    
    files_to_process = []
    
    # 遍历源目录查找 .zip
    try:
        for item in src_path.iterdir():
            if item.is_file() and item.suffix.lower() == '.zip':
                size_bytes = item.stat().st_size
                size_mb = size_bytes / (1024 * 1024)
                
                if size_mb < threshold_mb:
                    files_to_process.append({
                        'path': item,
                        'name': item.name,
                        'stem': item.stem, #文件名不含后缀
                        'size_mb': size_mb
                    })
    except Exception as e:
        print(f"[扫描错误] {e}")
        return

    # 按文件名排序
    files_to_process.sort(key=lambda x: x['name'])

    # ================== 阶段二：确认列表 ==================
    if not files_to_process:
        print("⚠️  没有发现符合条件（小于阈值）的 zip 文件。")
        return

    print(f"\n📦 发现 {len(files_to_process)} 个待处理文件：")
    for f in files_to_process:
        print(f"  - {f['name']} ({f['size_mb']:.2f} MB)")

    print(f"\n共计: {len(files_to_process)} 个文件。")
    
    if not get_user_confirmation(">>> 是否开始解压并整理这些文件？"):
        print(">>> 操作已取消。")
        return

    # ================== 阶段三：执行解压 ==================
    
    # 如果目标根目录不存在，询问是否创建
    if not dst_path.exists():
        print(f"\n目标目录不存在: {dst_path}")
        if get_user_confirmation(">>> 是否创建目标目录？"):
            dst_path.mkdir(parents=True, exist_ok=True)
        else:
            print("无法继续，程序退出。")
            return

    print("\n🚀 开始执行任务...")
    
    # 临时解压目录 (放在目标目录下，保证 move 操作在同一分区，速度快)
    temp_extract_root = dst_path / "temp_unzip_buffer"

    for i, file_info in enumerate(files_to_process, 1):
        zip_file_path = file_info['path']
        category_name = file_info['stem'] # 对应 Shell 中的 CATEGORY_NAME
        final_target_path = dst_path / category_name
        
        print(f"[{i}/{len(files_to_process)}] 处理: {file_info['name']} ... ", end="", flush=True)

        # 清理并重建临时目录
        if temp_extract_root.exists():
            shutil.rmtree(temp_extract_root)
        temp_extract_root.mkdir()

        try:
            # 1. 解压到临时目录
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_root)

            # 2. 处理目录结构 (复刻原 Shell 逻辑)
            # 逻辑：检查解压后是否多套了一层与 zip 同名的文件夹
            inner_dir = temp_extract_root / category_name
            
            # 准备最终目录
            final_target_path.mkdir(parents=True, exist_ok=True)

            source_content_path = None
            
            if inner_dir.exists() and inner_dir.is_dir():
                # 情况 A: 解压后是 temp/123/content... -> 移动 123 里面的内容
                source_content_path = inner_dir
            else:
                # 情况 B: 解压后是 temp/content... -> 移动 temp 里面的内容
                source_content_path = temp_extract_root

            # 3. 移动内容
            # shutil.move 不能直接合并目录，我们需要遍历移动子项
            for content in source_content_path.iterdir():
                # 移动 content 到 final_target_path
                # 注意：如果目标已存在同名文件，shutil.move 可能会报错或覆盖，取决于系统
                # 这里使用 shutil.move，它会尝试重命名
                shutil.move(str(content), str(final_target_path))

            print("✅ 完成")

        except zipfile.BadZipFile:
            print("❌ 错误: 文件损坏")
        except Exception as e:
            print(f"\n   ❌ 处理失败: {e}")
        finally:
            # 清理临时目录
            if temp_extract_root.exists():
                shutil.rmtree(temp_extract_root)

    # 最终清理（如果循环结束临时目录还在）
    if temp_extract_root.exists():
        shutil.rmtree(temp_extract_root)

    print("-" * 50)
    print("✅ 所有操作完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量筛选并解压 Zip 文件 (Python版)")
    
    # 1. 阈值参数 (-t / --threshold)
    parser.add_argument('-t', '--threshold', 
                        type=float, 
                        default=50.0, 
                        help='文件大小筛选阈值，单位 MB (默认: 50)')

    # 2. 源路径参数 (-s / --source)
    # 对应 Shell 中的 SOURCE_ROOT_DIR
    parser.add_argument('-s', '--source', 
                        type=str, 
                        required=True, 
                        help='[必须] zip 文件所在的源目录路径')

    # 3. 输出路径参数 (-p / --path)
    # 对应 Shell 中的 TARGET_ROOT_DIR，默认为当前路径 "."
    parser.add_argument('-p', '--path', 
                        type=str, 
                        default=".", 
                        help='解压后的目标存放路径 (默认: 当前目录)')

    args = parser.parse_args()

    process_archives(args.source, args.path, args.threshold)