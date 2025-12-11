import os
import shutil
import argparse
import sys

def get_user_confirmation(prompt_text):
    """
    获取用户确认，返回 True/False
    """
    while True:
        choice = input(f"{prompt_text} (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            return True
        if choice in ['n', 'no']:
            return False
        print("输入无效，请输入 y 或 n。")

def scan_and_clean(root_dir):
    root_dir = os.path.abspath(root_dir)
    print(f"🚀 开始在 {root_dir} 目录下扫描...")
    
    if not os.path.exists(root_dir):
        print(f"[错误] 路径不存在: {root_dir}")
        return

    # 用于存储待删除的路径列表
    pending_incomplete = []   # 渲染未完成的目录
    pending_checkpoints = []  # .ipynb_checkpoints 目录
    pending_empty = []        # 空目录

    # 使用 os.walk 遍历
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        
        # -------------------------------------------------
        # 1. 检查 .ipynb_checkpoints
        # -------------------------------------------------
        if '.ipynb_checkpoints' in dirnames:
            cp_path = os.path.join(dirpath, '.ipynb_checkpoints')
            pending_checkpoints.append(cp_path)
            # 从遍历列表中移除，防止进入已标记删除的目录
            dirnames.remove('.ipynb_checkpoints') 

        # -------------------------------------------------
        # 2. 检查渲染完整性 (核心逻辑)
        # -------------------------------------------------
        # 仅检查包含 'models' 子目录的文件夹 (符合 target_dir 特征)
        if 'models' in dirnames:
            target_dir = dirpath
            
            # 构建关键文件路径
            images_dir = os.path.join(target_dir, 'images')
            models_dir = os.path.join(target_dir, 'models')
            transforms_file = os.path.join(target_dir, 'transforms_test.json')
            models_file = os.path.join(models_dir, 'model_normalized.obj')
            
            # 逻辑判断
            if os.path.exists(models_file):
                if os.path.exists(transforms_file) and os.path.exists(images_dir):
                    # print(f"✅ 完整: {os.path.basename(target_dir)}") # 可选：太刷屏可以注释掉
                    pass
                else:
                    # 发现未处理目录
                    pending_incomplete.append(target_dir)
                    # 【重要】既然决定删除这个父目录，就清空 dirnames，
                    # 阻止 os.walk 继续深入扫描这个即将被删除的目录的子文件
                    dirnames[:] = [] 
                    continue 

        # -------------------------------------------------
        # 3. 检查空目录
        # -------------------------------------------------
        # 注意：如果刚才清空了 dirnames，这里自然就是空的，但 filenames 还在
        # 只有当确实没有子文件夹且没有文件时，才算空目录
        if not dirnames and not filenames:
            # 只有当它不在 pending_incomplete 列表里时才添加
            # (避免同一个目录既被标记为"未完成"又被标记为"空")
            if dirpath not in pending_incomplete:
                pending_empty.append(dirpath)

    print("--- 扫描完成 ---\n")

    # ================== 确认阶段 1: 渲染未完成目录 ==================
    if pending_incomplete:
        print(f"❌ 发现 {len(pending_incomplete)} 个【未处理/不完整】的目录 (缺少 transforms_test.json)：")
        for p in sorted(pending_incomplete):
            print(f"  -> {p}")
        
        print(f"\n共 {len(pending_incomplete)} 个项目。")
        if get_user_confirmation(">>> 警告：是否立即彻底删除上述目录？"):
            print("正在删除...")
            for p in pending_incomplete:
                try:
                    shutil.rmtree(p)
                    print(f"  [已删除] {os.path.basename(p)}")
                except Exception as e:
                    print(f"  [失败] {p}: {e}")
        else:
            print(">>> 已跳过删除操作。")
    else:
        print("✅ 未发现渲染不完整的目录。")

    print("-" * 30)

    # ================== 确认阶段 2: 垃圾清理 (.ipynb_checkpoints) ==================
    if pending_checkpoints:
        print(f"\n🧹 发现 {len(pending_checkpoints)} 个 .ipynb_checkpoints 缓存目录。")
        # 数量可能很多，只打印前5个示例
        for i, p in enumerate(pending_checkpoints[:5]):
            print(f"  -> {p}")
        if len(pending_checkpoints) > 5:
            print(f"  ... (以及其他 {len(pending_checkpoints)-5} 个)")

        if get_user_confirmation(">>> 是否清理这些缓存目录？"):
            for p in pending_checkpoints:
                try:
                    shutil.rmtree(p)
                except Exception:
                    pass # 忽略小错误
            print("  [完成] 缓存已清理。")
        else:
            print(">>> 已跳过。")

    # ================== 确认阶段 3: 空目录 ==================
    if pending_empty:
        print(f"\n🗑️ 发现 {len(pending_empty)} 个空目录。")
        if get_user_confirmation(">>> 是否删除所有空目录？"):
            count = 0
            for p in pending_empty:
                try:
                    # 再次检查是否存在（可能因为父目录在阶段1被删除了）
                    if os.path.exists(p):
                        os.rmdir(p) # 空目录用 rmdir 即可，比 rmtree 安全
                        count += 1
                except Exception as e:
                    print(f"  [删除空目录失败] {p}: {e}")
            print(f"  [完成] 删除了 {count} 个空目录。")
        else:
            print(">>> 已跳过。")
    else:
        print("\n没有发现空目录。")

    print("\n脚本执行完毕。")

if __name__ == "__main__":
    # 设置参数解析
    parser = argparse.ArgumentParser(description="扫描并清理渲染不完整的目录及垃圾文件。")
    
    # 添加 -p / --path 参数，默认值为当前目录 "."
    parser.add_argument('-p', '--path', 
                        type=str, 
                        default=".", 
                        help='指定搜索路径 (默认为当前目录)')
    
    args = parser.parse_args()
    
    # 执行主函数
    scan_and_clean(args.path)