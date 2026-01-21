"""
GraphRAG适配层

封装GraphRAG库的核心功能，与自研系统集成
"""

from .adapter import GraphRAGAdapter
from .config import GraphRAGConfig
from .extractors import GraphRAGEntityExtractor, GraphRAGRelationExtractor
from .searchers import GraphRAGGlobalSearcher, GraphRAGLocalSearcher

__all__ = [
    "GraphRAGAdapter",
    "GraphRAGConfig",
    "GraphRAGEntityExtractor",
    "GraphRAGRelationExtractor",
    "GraphRAGGlobalSearcher",
    "GraphRAGLocalSearcher",
]
