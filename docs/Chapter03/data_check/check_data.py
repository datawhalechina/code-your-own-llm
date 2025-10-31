"""
检查 Parquet 数据文件的完整性

用法:
    python -m nanochat.check_data
"""

import pyarrow.parquet as pq
import glob
import os
import sys


def check_data_integrity(data_dir=None):
    """检查数据目录中所有 Parquet 文件的完整性"""
    if data_dir is None:
        data_dir = os.path.expanduser("~/.cache/nanochat/base_data")
    
    pattern = os.path.join(data_dir, "*.parquet")
    
    # 查找所有文件
    files = sorted(glob.glob(pattern))
    
    print("=" * 60)
    print("数据完整性检查工具")
    print("=" * 60)
    print()
    print(f"📂 找到 {len(files)} 个 Parquet 文件")
    print(f"📁 位置: {data_dir}")
    print()
    
    if not files:
        print("❌ 没有找到任何文件!")
        print()
        print("请确认:")
        print(f"  1. 数据目录是否正确: {data_dir}")
        print("  2. 是否已下载数据: python -m nanochat.dataset -n 8")
        return False
    
    # 检查每个文件
    broken = []
    total_rows = 0
    
    print("=" * 60)
    print("检查进度:")
    print("=" * 60)
    
    for i, filepath in enumerate(files, 1):
        filename = os.path.basename(filepath)
        try:
            table = pq.read_table(filepath)
            rows = len(table)
            total_rows += rows
            
            if rows == 0:
                print(f"⚠️  [{i:3d}/{len(files)}] {filename}: 空文件")
                broken.append((filepath, "空文件"))
            else:
                print(f"✅ [{i:3d}/{len(files)}] {filename}: {rows:>6,} 条数据")
        except Exception as e:
            print(f"❌ [{i:3d}/{len(files)}] {filename}: 损坏 ({str(e)[:30]}...)")
            broken.append((filepath, str(e)))
    
    # 总结
    print()
    print("=" * 60)
    print("检查结果:")
    print("=" * 60)
    print()
    
    if broken:
        print(f"❌ 发现 {len(broken)} 个问题文件:")
        print()
        for filepath, error in broken:
            filename = os.path.basename(filepath)
            print(f"  - {filename}")
            print(f"    错误: {error}")
        print()
        print("💡 建议:")
        print("  1. 删除损坏的文件")
        print("  2. 重新运行下载命令: python -m nanochat.dataset -n N")
        return False
    else:
        print("✅ 所有文件完整!")
        print()
        print(f"📊 统计信息:")
        print(f"  文件数量: {len(files):,}")
        print(f"  总数据量: {total_rows:,} 条")
        print(f"  平均每文件: {total_rows // len(files):,} 条")
        return True


def main():
    # 允许指定自定义目录
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = None
    
    success = check_data_integrity(data_dir)
    
    print()
    print("=" * 60)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
