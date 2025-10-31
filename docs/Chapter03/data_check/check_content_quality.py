"""
随机抽样检查数据内容质量

用法:
    python -m nanochat.check_content_quality
"""

import pyarrow.parquet as pq
import random
import os
import sys


def check_content_quality(data_path, num_samples=10):
    """随机抽样检查内容质量"""
    if not os.path.exists(data_path):
        print(f"❌ 错误: 文件不存在 {data_path}")
        return
    
    print(f"📝 正在从 {data_path} 抽样...")
    print()
    
    # 读取数据
    table = pq.read_table(data_path)
    texts = table['text'].to_pylist()
    
    # 随机抽样
    if len(texts) < num_samples:
        samples = texts
        print(f"⚠️  数据量较少,只有 {len(texts)} 条,将全部展示")
    else:
        samples = random.sample(texts, num_samples)
    
    print("=" * 60)
    print(f"随机抽样检查 (共 {len(samples)} 条)")
    print("=" * 60)
    
    for i, text in enumerate(samples, 1):
        print(f"\n{'─' * 60}")
        print(f"📄 样本 {i}/{len(samples)}")
        print(f"{'─' * 60}")
        print(f"长度: {len(text)} 字符")
        print()
        
        # 显示前300字符
        if len(text) > 300:
            print(text[:300] + "...")
        else:
            print(text)
    
    print()
    print("=" * 60)
    print("📋 内容质量检查清单:")
    print("=" * 60)
    print()
    print("请手动检查以上样本,确认:")
    print("  ✅ 句子完整,语法正确")
    print("  ✅ 没有大量乱码")
    print("  ✅ 没有大量重复内容")
    print("  ✅ 内容有意义,信息量足够")
    print()
    
    # 自动检测一些明显的问题
    print("🔍 自动检测结果:")
    print()
    
    # 检测重复
    unique_texts = set(samples)
    if len(unique_texts) < len(samples):
        duplicate_count = len(samples) - len(unique_texts)
        print(f"⚠️  发现 {duplicate_count} 条重复文本")
    else:
        print("✅ 无完全重复的文本")
    
    # 检测极短文本
    very_short = [t for t in samples if len(t) < 50]
    if very_short:
        print(f"⚠️  发现 {len(very_short)} 条极短文本 (< 50 字符)")
    else:
        print("✅ 无极短文本")
    
    # 检测可能的乱码 (连续的非ASCII字符)
    def has_many_special_chars(text):
        special_count = sum(1 for c in text if ord(c) > 127 and not ('\u4e00' <= c <= '\u9fff'))
        return special_count > len(text) * 0.3
    
    garbled = [t for t in samples if has_many_special_chars(t)]
    if garbled:
        print(f"⚠️  发现 {len(garbled)} 条可能含乱码的文本")
    else:
        print("✅ 无明显乱码")


def main():
    print("=" * 60)
    print("内容质量抽样检查工具")
    print("=" * 60)
    print()
    
    # 默认检查路径
    default_path = os.path.expanduser("~/.cache/nanochat/base_data/shard_00000.parquet")
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = default_path
    
    # 抽样数量
    num_samples = 10
    if len(sys.argv) > 2:
        num_samples = int(sys.argv[2])
    
    check_content_quality(data_path, num_samples)


if __name__ == "__main__":
    main()

