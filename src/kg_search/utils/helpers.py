"""
通用工具函数
"""

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Iterator

import tiktoken


def generate_id(prefix: str = "") -> str:
    """
    生成唯一ID

    Args:
        prefix: ID前缀

    Returns:
        唯一标识符
    """
    unique_id = uuid.uuid4().hex[:12]
    return f"{prefix}_{unique_id}" if prefix else unique_id


def generate_hash_id(content: str) -> str:
    """
    基于内容生成哈希ID

    Args:
        content: 内容字符串

    Returns:
        哈希ID
    """
    return hashlib.md5(content.encode()).hexdigest()[:16]


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    计算文本的token数量

    Args:
        text: 输入文本
        model: 模型名称

    Returns:
        token数量
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def load_json_file(file_path: str | Path) -> dict[str, Any] | list[Any]:
    """
    加载JSON文件

    Args:
        file_path: 文件路径

    Returns:
        解析后的JSON数据
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_file(file_path: str | Path) -> list[dict[str, Any]]:
    """
    加载JSONL文件

    Args:
        file_path: 文件路径

    Returns:
        JSON对象列表
    """
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def save_json_file(data: Any, file_path: str | Path, indent: int = 2) -> None:
    """
    保存数据到JSON文件

    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: 缩进空格数
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def chunk_list(lst: list[Any], chunk_size: int) -> Iterator[list[Any]]:
    """
    将列表分块

    Args:
        lst: 输入列表
        chunk_size: 块大小

    Yields:
        分块后的列表
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """
    扁平化嵌套字典

    Args:
        d: 嵌套字典
        parent_key: 父键
        sep: 分隔符

    Returns:
        扁平化后的字典
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def extract_nested_value(d: dict[str, Any], key_path: str, sep: str = ".") -> Any:
    """
    从嵌套字典中提取值

    Args:
        d: 嵌套字典
        key_path: 键路径，如 "basic_info.name"
        sep: 分隔符

    Returns:
        提取的值，如果路径不存在则返回None
    """
    keys = key_path.split(sep)
    value = d
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value
