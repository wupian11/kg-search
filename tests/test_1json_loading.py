"""
测试 data/1.json 文物数据的加载和提取流程

使用前10个文物数据作为测试集，验证：
1. JSON文件加载
2. 文档内容提取
3. 实体抽取（不需要LLM的结构化提取）
4. 完整调用链路
"""

import json
from pathlib import Path

import pytest

from kg_search.ingestion.loaders import JSONLoader
from kg_search.ingestion.loaders.base import Document
from kg_search.ingestion.chunkers import StructureChunker, RecursiveChunker
from kg_search.ingestion import IngestionPipeline
from kg_search.extraction.entity_extractor import EntityExtractor, Entity
from kg_search.utils import get_logger, setup_logging

# 设置日志
setup_logging(log_level="DEBUG")
logger = get_logger(__name__)

# 数据文件路径
DATA_FILE = Path(__file__).parent.parent / "data" / "1.json"

# 文物数据的字段映射（适配1.json的结构）
ARTIFACT_FIELD_MAPPING = {
    "artifact_id": "caseId",
    "artifact_name": "caseName",
    "dynasty": "descAge",  # 年代描述
    "period": "descAge",
    "material": "descMaterial",  # 材质
    "dimensions": "descSize",  # 尺寸
    "museum": "govInstitutionName",  # 管理机构
    "location": "addresses.0.cityName",  # 城市
    "description": "reserveStatusDesc",  # 保存状态描述
    "technique": None,  # 此数据集没有工艺字段
    "culture": "assetsClassifyName",  # 资产分类
}


class ArtifactJSONLoader(JSONLoader):
    """文物数据专用加载器"""
    
    def __init__(self):
        """初始化加载器，使用文物数据的字段映射"""
        super().__init__(field_mapping=ARTIFACT_FIELD_MAPPING)
    
    def _create_document(self, data: dict, source: str) -> Document:
        """
        从文物数据创建文档
        
        重写以适配1.json的特殊结构
        """
        from kg_search.utils import extract_nested_value, generate_id
        
        # 提取字段值
        extracted = {}
        for doc_field, json_path in self.field_mapping.items():
            if json_path is None:
                continue
            # 处理数组索引路径
            value = self._extract_value_with_array_index(data, json_path)
            if value is not None:
                extracted[doc_field] = value
        
        # 构建文档内容
        content = self._build_content_for_artifact(extracted, data)
        
        # 生成ID
        doc_id = extracted.get("artifact_id") or generate_id("artifact")
        
        return Document(
            id=doc_id,
            content=content,
            metadata=data,  # 保存原始完整数据
            source=source,
            doc_type="artifact",
            artifact_id=extracted.get("artifact_id"),
            artifact_name=extracted.get("artifact_name"),
            dynasty=extracted.get("dynasty"),
            period=extracted.get("period"),
            material=extracted.get("material"),
            technique=extracted.get("technique"),
            location=extracted.get("location"),
            museum=extracted.get("museum"),
            culture=extracted.get("culture"),
            dimensions=extracted.get("dimensions"),
            description=extracted.get("description"),
        )
    
    def _extract_value_with_array_index(self, data: dict, key_path: str) -> any:
        """提取值，支持数组索引，如 addresses.0.cityName"""
        keys = key_path.split(".")
        value = data
        for key in keys:
            if value is None:
                return None
            # 检查是否是数组索引
            if key.isdigit():
                idx = int(key)
                if isinstance(value, list) and len(value) > idx:
                    value = value[idx]
                else:
                    return None
            elif isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
    
    def _build_content_for_artifact(self, extracted: dict, raw_data: dict) -> str:
        """构建文物的文档内容"""
        parts = []
        
        if name := extracted.get("artifact_name"):
            parts.append(f"文物名称：{name}")
        
        if culture := extracted.get("culture"):
            parts.append(f"文物分类：{culture}")
        
        # 从原始数据获取更多信息
        if asset_types := raw_data.get("assetsTypes"):
            for at in asset_types:
                if at_name := at.get("name"):
                    parts.append(f"类型：{at_name}")
                if sub_name := at.get("subName"):
                    parts.append(f"子类型：{sub_name}")
        
        if period := extracted.get("period"):
            parts.append(f"年代：{period}")
        
        if material := extracted.get("material"):
            parts.append(f"材质：{material}")
        
        if dimensions := extracted.get("dimensions"):
            parts.append(f"尺寸：{dimensions}")
        
        if location := extracted.get("location"):
            parts.append(f"地点：{location}")
        
        if museum := extracted.get("museum"):
            parts.append(f"管理机构：{museum}")
        
        # 添加登记原因
        if register_reason := raw_data.get("registerReason"):
            parts.append(f"登记原因：{register_reason}")
        
        # 添加描述
        if description := extracted.get("description"):
            parts.append(f"描述：{description}")
        
        return "\n".join(parts)


def load_first_n_artifacts(n: int = 10) -> list[dict]:
    """加载前N个文物数据"""
    logger.info(f"Loading first {n} artifacts from {DATA_FILE}")
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        artifacts = data[:n]
    else:
        artifacts = data.get("artifacts", data.get("items", []))[:n]
    
    logger.info(f"Loaded {len(artifacts)} artifacts")
    return artifacts


class TestArtifactJSONLoading:
    """测试文物JSON数据加载"""
    
    @pytest.fixture
    def test_artifacts(self) -> list[dict]:
        """获取前10个测试文物数据"""
        return load_first_n_artifacts(10)
    
    @pytest.fixture
    def artifact_loader(self) -> ArtifactJSONLoader:
        """创建文物加载器"""
        return ArtifactJSONLoader()
    
    def test_load_first_10_artifacts(self, test_artifacts):
        """测试加载前10个文物数据"""
        logger.info("=" * 60)
        logger.info("测试: 加载前10个文物数据")
        logger.info("=" * 60)
        
        assert len(test_artifacts) == 10, f"Expected 10 artifacts, got {len(test_artifacts)}"
        
        for i, artifact in enumerate(test_artifacts, 1):
            case_id = artifact.get("caseId", "N/A")
            case_name = artifact.get("caseName", "N/A")
            logger.info(f"文物 {i}: ID={case_id}, 名称={case_name}")
        
        logger.info(f"成功加载 {len(test_artifacts)} 个文物数据")
    
    def test_json_loader_with_data(self, test_artifacts, artifact_loader, tmp_path):
        """测试JSON加载器处理文物数据"""
        logger.info("=" * 60)
        logger.info("测试: JSON加载器处理文物数据")
        logger.info("=" * 60)
        
        # 将测试数据写入临时文件
        test_file = tmp_path / "test_artifacts.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_artifacts, f, ensure_ascii=False, indent=2)
        
        logger.info(f"临时测试文件: {test_file}")
        
        # 使用加载器加载
        documents = artifact_loader.load(test_file)
        
        assert len(documents) == 10, f"Expected 10 documents, got {len(documents)}"
        
        logger.info(f"成功创建 {len(documents)} 个文档对象")
        
        # 打印每个文档的详细信息
        for i, doc in enumerate(documents, 1):
            logger.info("-" * 40)
            logger.info(f"文档 {i}:")
            logger.info(f"  ID: {doc.id}")
            logger.info(f"  名称: {doc.artifact_name}")
            logger.info(f"  分类: {doc.culture}")
            logger.info(f"  年代: {doc.period}")
            logger.info(f"  材质: {doc.material}")
            logger.info(f"  地点: {doc.location}")
            logger.info(f"  管理机构: {doc.museum}")
            logger.info(f"  内容长度: {len(doc.content)} 字符")
        
        return documents
    
    def test_document_content_generation(self, test_artifacts, artifact_loader, tmp_path):
        """测试文档内容生成"""
        logger.info("=" * 60)
        logger.info("测试: 文档内容生成")
        logger.info("=" * 60)
        
        # 写入临时文件并加载
        test_file = tmp_path / "test_artifacts.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_artifacts, f, ensure_ascii=False, indent=2)
        
        documents = artifact_loader.load(test_file)
        
        # 验证第一个文档的内容
        first_doc = documents[0]
        logger.info(f"第一个文档的完整内容:")
        logger.info("-" * 40)
        logger.info(first_doc.content)
        logger.info("-" * 40)
        
        # 验证内容包含关键字段
        assert first_doc.artifact_name in first_doc.content, "文档内容应包含文物名称"
        
        # 验证元数据保存完整
        assert "caseId" in first_doc.metadata, "元数据应包含caseId"
        assert "caseName" in first_doc.metadata, "元数据应包含caseName"
        
        logger.info("文档内容生成验证通过")


class TestChunkingPipeline:
    """测试分块流程"""
    
    @pytest.fixture
    def documents(self, tmp_path) -> list[Document]:
        """准备测试文档"""
        artifacts = load_first_n_artifacts(10)
        loader = ArtifactJSONLoader()
        
        test_file = tmp_path / "test_artifacts.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(artifacts, f, ensure_ascii=False, indent=2)
        
        return loader.load(test_file)
    
    def test_structure_chunking(self, documents):
        """测试结构化分块"""
        logger.info("=" * 60)
        logger.info("测试: 结构化分块")
        logger.info("=" * 60)
        
        chunker = StructureChunker(
            chunk_size=512,
            chunk_overlap=0,
            max_tokens=500,
            strategy="auto",
        )
        
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)
            logger.info(f"文档 {doc.artifact_name}: 生成 {len(chunks)} 个块")
        
        logger.info(f"总共生成 {len(all_chunks)} 个块")
        
        # 打印前5个块的内容
        for i, chunk in enumerate(all_chunks[:5], 1):
            logger.info(f"块 {i} (文档: {chunk.document_id}):")
            logger.info(f"  内容: {chunk.content[:100]}...")
        
        return all_chunks
    
    def test_recursive_chunking(self, documents):
        """测试递归分块"""
        logger.info("=" * 60)
        logger.info("测试: 递归分块")
        logger.info("=" * 60)
        
        chunker = RecursiveChunker(
            chunk_size=256,
            chunk_overlap=50,
            max_tokens=300,
        )
        
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)
            logger.info(f"文档 {doc.artifact_name}: 生成 {len(chunks)} 个块")
        
        logger.info(f"总共生成 {len(all_chunks)} 个块")
        
        return all_chunks


class TestStructuredEntityExtraction:
    """测试结构化实体提取（不需要LLM）"""
    
    @pytest.fixture
    def documents(self, tmp_path) -> list[Document]:
        """准备测试文档"""
        artifacts = load_first_n_artifacts(10)
        loader = ArtifactJSONLoader()
        
        test_file = tmp_path / "test_artifacts.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(artifacts, f, ensure_ascii=False, indent=2)
        
        return loader.load(test_file)
    
    def test_extract_structured_entities(self, documents):
        """测试从文档结构中提取实体"""
        logger.info("=" * 60)
        logger.info("测试: 结构化实体提取")
        logger.info("=" * 60)
        
        # 创建一个Mock LLM客户端（实际测试中不使用LLM）
        class MockLLMClient:
            async def chat_completion(self, **kwargs):
                return json.dumps({"entities": []})
        
        extractor = EntityExtractor(llm_client=MockLLMClient())
        
        all_entities = []
        for doc in documents:
            # 直接调用结构化提取方法
            entities = extractor._extract_structured_entities(doc)
            all_entities.extend(entities)
            
            logger.info(f"文档 '{doc.artifact_name}' 提取的实体:")
            for entity in entities:
                logger.info(f"  - {entity.type}: {entity.name}")
        
        logger.info("-" * 40)
        logger.info(f"总共提取 {len(all_entities)} 个实体")
        
        # 统计实体类型
        entity_types = {}
        for entity in all_entities:
            type_name = entity.type.value if hasattr(entity.type, 'value') else str(entity.type)
            entity_types[type_name] = entity_types.get(type_name, 0) + 1
        
        logger.info("实体类型统计:")
        for type_name, count in sorted(entity_types.items()):
            logger.info(f"  {type_name}: {count}")
        
        return all_entities


class TestFullPipeline:
    """测试完整的加载-分块-提取流程"""
    
    def test_full_pipeline_with_10_artifacts(self, tmp_path):
        """测试完整流程"""
        logger.info("=" * 60)
        logger.info("测试: 完整的加载-分块-提取流程")
        logger.info("=" * 60)
        
        # 1. 加载数据
        logger.info("步骤1: 加载文物数据")
        artifacts = load_first_n_artifacts(10)
        
        test_file = tmp_path / "test_artifacts.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(artifacts, f, ensure_ascii=False, indent=2)
        
        # 2. 使用自定义加载器
        logger.info("步骤2: 创建文档对象")
        loader = ArtifactJSONLoader()
        documents = loader.load(test_file)
        logger.info(f"  创建了 {len(documents)} 个文档")
        
        # 3. 分块
        logger.info("步骤3: 文档分块")
        chunker = StructureChunker(
            chunk_size=512,
            max_tokens=500,
            strategy="auto",
        )
        
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)
        logger.info(f"  生成了 {len(all_chunks)} 个块")
        
        # 4. 结构化实体提取
        logger.info("步骤4: 结构化实体提取")
        
        class MockLLMClient:
            async def chat_completion(self, **kwargs):
                return json.dumps({"entities": []})
        
        extractor = EntityExtractor(llm_client=MockLLMClient())
        
        all_entities = []
        for doc in documents:
            entities = extractor._extract_structured_entities(doc)
            all_entities.extend(entities)
        logger.info(f"  提取了 {len(all_entities)} 个实体")
        
        # 5. 输出摘要
        logger.info("=" * 60)
        logger.info("处理摘要")
        logger.info("=" * 60)
        logger.info(f"  输入文件: {test_file}")
        logger.info(f"  文物数量: {len(artifacts)}")
        logger.info(f"  文档数量: {len(documents)}")
        logger.info(f"  块数量: {len(all_chunks)}")
        logger.info(f"  实体数量: {len(all_entities)}")
        
        # 输出每个文物的处理结果
        logger.info("-" * 40)
        logger.info("各文物处理详情:")
        for i, doc in enumerate(documents, 1):
            doc_chunks = [c for c in all_chunks if c.document_id == doc.id]
            doc_entities = [e for e in all_entities if e.source_doc_id == doc.id]
            logger.info(f"{i}. {doc.artifact_name}")
            logger.info(f"   - 块数: {len(doc_chunks)}")
            logger.info(f"   - 实体数: {len(doc_entities)}")
        
        # 验证
        assert len(documents) == 10
        assert len(all_chunks) > 0
        assert len(all_entities) > 0
        
        logger.info("=" * 60)
        logger.info("完整流程测试通过!")
        logger.info("=" * 60)


if __name__ == "__main__":
    # 直接运行时执行简单测试
    import asyncio
    
    setup_logging(log_level="DEBUG")
    
    print("=" * 60)
    print("直接运行: 测试 1.json 加载和提取流程")
    print("=" * 60)
    
    # 加载数据
    artifacts = load_first_n_artifacts(10)
    print(f"\n加载了 {len(artifacts)} 个文物数据\n")
    
    # 打印每个文物的基本信息
    for i, artifact in enumerate(artifacts, 1):
        print(f"{i}. {artifact.get('caseName', 'N/A')}")
        print(f"   ID: {artifact.get('caseId', 'N/A')}")
        print(f"   分类: {artifact.get('assetsClassifyName', 'N/A')}")
        print(f"   材质: {artifact.get('descMaterial', 'N/A')}")
        print(f"   年代: {artifact.get('descAge', 'N/A')}")
        print()
    
    # 测试加载器
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(artifacts, f, ensure_ascii=False)
        temp_file = f.name
    
    loader = ArtifactJSONLoader()
    documents = loader.load(temp_file)
    
    print(f"\n创建了 {len(documents)} 个文档对象\n")
    
    # 打印第一个文档的详细信息
    print("=" * 60)
    print("第一个文档详情:")
    print("=" * 60)
    doc = documents[0]
    print(f"ID: {doc.id}")
    print(f"名称: {doc.artifact_name}")
    print(f"分类: {doc.culture}")
    print(f"年代: {doc.period}")
    print(f"材质: {doc.material}")
    print(f"地点: {doc.location}")
    print(f"管理机构: {doc.museum}")
    print(f"\n内容:")
    print("-" * 40)
    print(doc.content)
    print("-" * 40)
    
    # 测试实体提取
    class MockLLMClient:
        async def chat_completion(self, **kwargs):
            return json.dumps({"entities": []})
    
    extractor = EntityExtractor(llm_client=MockLLMClient())
    
    print("\n" + "=" * 60)
    print("结构化实体提取结果:")
    print("=" * 60)
    
    for doc in documents[:3]:  # 只打印前3个文档的实体
        entities = extractor._extract_structured_entities(doc)
        print(f"\n文档: {doc.artifact_name}")
        for entity in entities:
            type_str = entity.type.value if hasattr(entity.type, 'value') else str(entity.type)
            print(f"  - [{type_str}] {entity.name}")
    
    # 清理临时文件
    import os
    os.unlink(temp_file)
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
