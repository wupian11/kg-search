"""
日志配置模块

使用 structlog 进行结构化日志记录
"""

import logging
import sys
from typing import Any

import structlog


def setup_logging(log_level: str = "INFO", json_format: bool = False) -> None:
    """
    配置日志系统

    Args:
        log_level: 日志级别
        json_format: 是否使用JSON格式输出
    """
    # 配置标准库logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # 配置structlog处理器
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        # 生产环境：JSON格式
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # 开发环境：彩色控制台输出
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        结构化日志记录器
    """
    return structlog.get_logger(name)
