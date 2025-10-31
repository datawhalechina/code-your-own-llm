"""
数据检查和处理工具集

这个包提供了一系列用于检查和处理训练数据的工具：

- check_data: 验证 Parquet 数据文件的完整性
- check_length_distribution: 检查文本长度分布
- check_content_quality: 随机抽样检查内容质量
- check_char_distribution: 检查字符分布统计
- convert_custom_data: 转换自定义文本数据为 Parquet 格式

用法示例：
    python -m data_check.check_data
    python -m data_check.check_length_distribution
    python -m data_check.check_content_quality
    python -m data_check.check_char_distribution
    python -m data_check.convert_custom_data
"""

__version__ = "0.1.0"
__all__ = [
    "check_data",
    "check_length_distribution",
    "check_content_quality",
    "check_char_distribution",
    "convert_custom_data",
]

