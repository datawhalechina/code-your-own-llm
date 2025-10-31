"""
把你自己的文本数据转换成训练用的格式

用法:
    python -m nanochat.convert_custom_data
"""

import pyarrow.parquet as pq
import pyarrow as pa
import os
import glob


def convert_from_single_file(input_file, output_dir, min_length=50):
    """从单个文本文件读取并转换"""
    texts = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) > min_length:
                texts.append(line)
    
    print(f"✅ 从 {input_file} 读取 {len(texts):,} 条文本")
    return texts


def convert_from_directory(input_dir, output_dir, min_length=50):
    """从多个文本文件读取并转换"""
    texts = []
    pattern = os.path.join(input_dir, "*.txt")
    
    for file_path in glob.glob(pattern):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            if len(content) > min_length:
                texts.append(content)
    
    print(f"✅ 从 {input_dir} 读取 {len(texts):,} 条文本")
    return texts


def save_to_parquet(texts, output_dir, shard_size=100):
    """保存文本到 Parquet 格式"""
    os.makedirs(output_dir, exist_ok=True)
    
    num_shards = (len(texts) + shard_size - 1) // shard_size
    
    for i in range(num_shards):
        start = i * shard_size
        end = min(start + shard_size, len(texts))
        shard_texts = texts[start:end]
        
        # 创建表格
        table = pa.Table.from_pydict({
            'text': shard_texts
        })
        
        # 保存
        output_path = os.path.join(output_dir, f"shard_{i:05d}.parquet")
        pq.write_table(
            table,
            output_path,
            row_group_size=1024,
            compression='zstd',
            compression_level=3
        )
        print(f"  保存: {output_path} ({len(shard_texts)} 条)")
    
    print(f"\n✅ 转换完成! 数据保存在: {output_dir}")


def main():
    print("=" * 60)
    print("自定义数据转换工具")
    print("=" * 60)
    
    # 配置参数
    # 方法1: 从单个文件读取 (取消注释下面这行)
    # input_file = "my_data.txt"
    
    # 方法2: 从目录读取多个文件 (取消注释下面这行)
    input_dir = "my_texts"
    
    # 输出目录
    output_dir = os.path.expanduser("~/.cache/nanochat/my_data")
    
    # 最小文本长度
    min_length = 50
    
    # 每个分片的大小
    shard_size = 100
    
    # 执行转换
    # texts = convert_from_single_file(input_file, output_dir, min_length)
    texts = convert_from_directory(input_dir, output_dir, min_length)
    
    if len(texts) == 0:
        print("❌ 错误: 没有找到符合条件的文本!")
        return
    
    save_to_parquet(texts, output_dir, shard_size)


if __name__ == "__main__":
    main()

