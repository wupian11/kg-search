# KG-Search: 文博领域智能检索系统

基于 **GraphRAG** 和 **向量数据库** 的智能检索系统，专为文博领域数字文物库设计。

## ✨ 特性

- 🔍 **混合检索**: 结合向量相似度搜索和知识图谱遍历
- 🧠 **GraphRAG**: 支持 Local Search 和 Global Search
- 📊 **知识图谱**: 自动提取文物实体和关系
- 🗄️ **双存储**: ChromaDB + Neo4j
- 📄 **多格式支持**: JSON/JSONL/Markdown/TXT
- 🚀 **API服务**: FastAPI RESTful API
- 🐳 **容器化部署**: Docker Compose

## 📦 快速开始

```bash
# Docker Compose 启动
cp docker/.env.example docker/.env
cd docker && docker-compose up -d

# 或本地开发
pip install -e .
python scripts/seed_data.py
python scripts/build_index.py --data-dir ./data/raw
kg-search
```

详细文档请参阅 [docs/README.md](docs/README.md)
