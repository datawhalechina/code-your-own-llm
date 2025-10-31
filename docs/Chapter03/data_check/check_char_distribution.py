"""
检查数据文件中的字符分布

用法:
    python -m nanochat.check_char_distribution
"""

import pyarrow.parquet as pq
import string
import os
import sys
from collections import Counter


def check_char_distribution(data_path):
    """检查字符分布"""
    if not os.path.exists(data_path):
        print(f"❌ 错误: 文件不存在 {data_path}")
        return
    
    print(f"🔤 正在分析: {data_path}")
    print()
    
    # 读取数据
    table = pq.read_table(data_path)
    texts = table['text'].to_pylist()
    
    # 合并所有文本
    all_text = "".join(texts)
    
    # 统计字符类型
    total_chars = len(all_text)
    chinese_chars = sum(1 for c in all_text if '\u4e00' <= c <= '\u9fff')
    english_chars = sum(1 for c in all_text if c in string.ascii_letters)
    digit_chars = sum(1 for c in all_text if c in string.digits)
    space_chars = sum(1 for c in all_text if c in string.whitespace)
    punctuation_chars = sum(1 for c in all_text if c in string.punctuation)
    other_chars = total_chars - chinese_chars - english_chars - digit_chars - space_chars - punctuation_chars
    
    print("=" * 60)
    print("📊 字符分布统计")
    print("=" * 60)
    print()
    print(f"总字符数: {total_chars:,}")
    print()
    
    # 打印各类字符占比
    categories = [
        ("中文字符", chinese_chars),
        ("英文字符", english_chars),
        ("数字", digit_chars),
        ("空白字符", space_chars),
        ("标点符号", punctuation_chars),
        ("其他字符", other_chars),
    ]
    
    for name, count in categories:
        percentage = 100 * count / total_chars if total_chars > 0 else 0
        bar = "█" * int(percentage / 2)
        print(f"{name:>8}: {bar} {percentage:5.1f}% ({count:,})")
    
    print()
    print("=" * 60)
    print("📋 数据质量评估 (针对中文数据)")
    print("=" * 60)
    print()
    
    # 中文数据质量评估
    chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
    english_ratio = english_chars / total_chars if total_chars > 0 else 0
    digit_ratio = digit_chars / total_chars if total_chars > 0 else 0
    
    if 0.7 <= chinese_ratio <= 0.9:
        print("✅ 中文字符比例正常 (70-90%)")
    elif chinese_ratio < 0.7:
        print(f"⚠️  中文字符比例偏低 ({chinese_ratio*100:.1f}%)")
        print("   建议: 确认是否为中文数据集")
    else:
        print(f"✅ 中文字符比例: {chinese_ratio*100:.1f}%")
    
    if 0.05 <= english_ratio <= 0.2:
        print("✅ 英文字符比例正常 (5-20%)")
    elif english_ratio > 0.2:
        print(f"⚠️  英文字符比例偏高 ({english_ratio*100:.1f}%)")
    else:
        print(f"✅ 英文字符比例: {english_ratio*100:.1f}%")
    
    if 0.01 <= digit_ratio <= 0.05:
        print("✅ 数字比例正常 (1-5%)")
    elif digit_ratio > 0.05:
        print(f"⚠️  数字比例偏高 ({digit_ratio*100:.1f}%)")
    else:
        print(f"✅ 数字比例: {digit_ratio*100:.1f}%")
    
    # 统计最常见的字符
    print()
    print("=" * 60)
    print("🔝 最常见的20个字符")
    print("=" * 60)
    print()
    
    char_counter = Counter(all_text)
    most_common = char_counter.most_common(20)
    
    for i, (char, count) in enumerate(most_common, 1):
        percentage = 100 * count / total_chars
        # 显示可见字符
        if char in string.whitespace:
            char_display = repr(char)
        else:
            char_display = f"'{char}'"
        print(f"{i:2d}. {char_display:>6} : {count:>8,} ({percentage:4.2f}%)")


def main():
    print("=" * 60)
    print("字符分布检查工具")
    print("=" * 60)
    print()
    
    # 默认检查路径
    default_path = os.path.expanduser("~/.cache/nanochat/base_data/shard_00000.parquet")
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = default_path
    
    check_char_distribution(data_path)


if __name__ == "__main__":
    main()

