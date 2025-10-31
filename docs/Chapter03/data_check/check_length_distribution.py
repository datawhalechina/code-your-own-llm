"""
检查数据文件中文本的长度分布

用法:
    python -m nanochat.check_length_distribution
"""

import pyarrow.parquet as pq
import os
import sys


def check_length_distribution(data_path):
    """检查文本长度分布"""
    if not os.path.exists(data_path):
        print(f"❌ 错误: 文件不存在 {data_path}")
        return
    
    print(f"📊 正在分析: {data_path}")
    print()
    
    # 读取数据
    table = pq.read_table(data_path)
    texts = table['text'].to_pylist()
    
    # 统计长度
    lengths = [len(text) for text in texts]
    
    # 打印统计信息
    avg_length = sum(lengths) / len(lengths)
    min_length = min(lengths)
    max_length = max(lengths)
    
    print(f"📈 长度统计:")
    print(f"  总样本数: {len(texts):,}")
    print(f"  平均长度: {avg_length:.0f} 字符")
    print(f"  最短: {min_length} 字符")
    print(f"  最长: {max_length} 字符")
    print()
    
    # 分桶统计
    buckets = {
        "< 50": 0,
        "50-100": 0,
        "100-500": 0,
        "500-1000": 0,
        "1000-2000": 0,
        "2000-5000": 0,
        "> 5000": 0
    }
    
    for length in lengths:
        if length < 50:
            buckets["< 50"] += 1
        elif length < 100:
            buckets["50-100"] += 1
        elif length < 500:
            buckets["100-500"] += 1
        elif length < 1000:
            buckets["500-1000"] += 1
        elif length < 2000:
            buckets["1000-2000"] += 1
        elif length < 5000:
            buckets["2000-5000"] += 1
        else:
            buckets["> 5000"] += 1
    
    print(f"📊 长度分布:")
    for bucket, count in buckets.items():
        percentage = 100 * count / len(texts)
        bar = "█" * int(percentage / 2)
        print(f"  {bucket:>12} 字符: {bar} {percentage:5.1f}% ({count:,})")
    print()
    
    # 绘制分布图 (如果安装了 matplotlib)
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('文本长度 (字符数)', fontsize=12)
        plt.ylabel('样本数量', fontsize=12)
        plt.title('文本长度分布', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        
        output_file = 'length_distribution.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ 分布图已保存到: {output_file}")
    except ImportError:
        print("💡 提示: 安装 matplotlib 可以生成可视化图表")
        print("   pip install matplotlib")
    
    # 质量评估
    print()
    print("=" * 60)
    print("📋 数据质量评估:")
    print("=" * 60)
    
    if avg_length < 500:
        print("⚠️  平均长度偏短 (< 500 字符)")
        print("   建议: 增加更长的文本样本")
    elif avg_length > 2000:
        print("⚠️  平均长度偏长 (> 2000 字符)")
        print("   建议: 可以考虑切分长文本")
    else:
        print("✅ 平均长度合理 (500-2000 字符)")
    
    short_ratio = buckets["< 50"] / len(texts)
    if short_ratio > 0.1:
        print(f"⚠️  超短文本过多 ({short_ratio*100:.1f}%)")
        print("   建议: 过滤掉太短的文本")
    else:
        print("✅ 超短文本比例正常")
    
    long_ratio = buckets["> 5000"] / len(texts)
    if long_ratio > 0.1:
        print(f"⚠️  超长文本较多 ({long_ratio*100:.1f}%)")
        print("   建议: 可能需要切分")
    else:
        print("✅ 超长文本比例正常")


def main():
    print("=" * 60)
    print("文本长度分布检查工具")
    print("=" * 60)
    print()
    
    # 默认检查路径
    default_path = os.path.expanduser("~/.cache/nanochat/base_data/shard_00000.parquet")
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = default_path
    
    check_length_distribution(data_path)


if __name__ == "__main__":
    main()

