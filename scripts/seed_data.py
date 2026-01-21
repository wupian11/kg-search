#!/usr/bin/env python
"""
示例数据导入脚本

导入示例文物数据用于测试
"""

import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kg_search.config import get_settings

# 示例文物数据（嵌套JSON格式）
SAMPLE_ARTIFACTS = [
    {
        "id": "artifact_001",
        "basic_info": {
            "name": "四羊方尊",
            "dynasty": "商代",
            "period": "公元前14世纪-前11世纪",
            "culture": "商文化",
        },
        "physical_info": {
            "material": "青铜",
            "technique": "分铸法",
            "dimensions": "高58.3厘米，重34.5公斤",
        },
        "provenance": {
            "excavation_site": "湖南宁乡",
            "excavation_year": "1938年",
            "museum": "中国国家博物馆",
        },
        "description": "四羊方尊是商代晚期青铜礼器，属于祭祀用的酒器。方尊四角各有一只卷角羊，羊头伸出器外，羊角弯曲，造型生动。器身装饰有云雷纹、夔龙纹等，工艺精湛，是商代青铜艺术的杰出代表。",
    },
    {
        "id": "artifact_002",
        "basic_info": {
            "name": "司母戊鼎",
            "dynasty": "商代",
            "period": "公元前14世纪-前11世纪",
            "culture": "商文化",
        },
        "physical_info": {
            "material": "青铜",
            "technique": "分铸法、浑铸法",
            "dimensions": "高133厘米，口长110厘米，口宽79厘米，重832.84公斤",
        },
        "provenance": {
            "excavation_site": "河南安阳殷墟",
            "excavation_year": "1939年",
            "museum": "中国国家博物馆",
        },
        "description": "司母戊鼎是迄今世界上出土最大、最重的青铜礼器，是商王为祭祀其母戊而制。鼎身呈长方形，口沿有厚实的双耳，四足为圆柱形。器身装饰有饕餮纹、夔龙纹。内壁铸有'司母戊'三字铭文。",
    },
    {
        "id": "artifact_003",
        "basic_info": {
            "name": "玉琮",
            "dynasty": "新石器时代",
            "period": "公元前3300年-前2200年",
            "culture": "良渚文化",
        },
        "physical_info": {
            "material": "玉",
            "technique": "琢磨",
            "dimensions": "高8.9厘米，射径17.1-17.6厘米",
        },
        "provenance": {
            "excavation_site": "浙江余杭反山",
            "excavation_year": "1986年",
            "museum": "浙江省博物馆",
        },
        "description": "良渚文化玉琮是良渚文化最具代表性的玉器类型。外方内圆，象征天圆地方。器表刻有神人兽面纹，是良渚先民的神徽标志。玉质温润，工艺精细，体现了良渚文化高度发达的玉器制作技术。",
    },
    {
        "id": "artifact_004",
        "basic_info": {
            "name": "曾侯乙编钟",
            "dynasty": "战国",
            "period": "公元前433年",
            "culture": "楚文化",
        },
        "physical_info": {
            "material": "青铜",
            "technique": "分铸法",
            "dimensions": "全套65件，总重量约2567公斤",
        },
        "provenance": {
            "excavation_site": "湖北随州擂鼓墩",
            "excavation_year": "1978年",
            "museum": "湖北省博物馆",
        },
        "description": "曾侯乙编钟是战国早期曾国国君曾侯乙的墓葬出土乐器，是目前发现数量最多、保存最完好、音域最宽的青铜编钟。全套编钟分三层悬挂，可演奏完整的五声、六声及七声音阶乐曲，是中国古代音乐文化的杰出代表。",
    },
    {
        "id": "artifact_005",
        "basic_info": {
            "name": "秦始皇兵马俑",
            "dynasty": "秦代",
            "period": "公元前210年左右",
            "culture": "秦文化",
        },
        "physical_info": {
            "material": "陶",
            "technique": "模制、塑形、彩绘",
            "dimensions": "将军俑高约197厘米，一般武士俑高约180厘米",
        },
        "provenance": {
            "excavation_site": "陕西西安临潼",
            "excavation_year": "1974年",
            "museum": "秦始皇帝陵博物院",
        },
        "description": "秦始皇兵马俑是秦始皇陵的陪葬坑，被誉为'世界第八大奇迹'。已发现三个俑坑，出土陶俑、陶马约8000件。每个陶俑神态各异，形象逼真，展现了秦代高超的雕塑艺术和严密的军事编制。",
    },
]


def create_sample_data():
    """创建示例数据文件"""
    settings = get_settings()

    # 创建数据目录
    raw_dir = Path(settings.raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 保存JSON文件
    sample_file = raw_dir / "sample_artifacts.json"
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump({"artifacts": SAMPLE_ARTIFACTS}, f, ensure_ascii=False, indent=2)

    print(f"示例数据已创建: {sample_file}")
    print(f"包含 {len(SAMPLE_ARTIFACTS)} 件文物数据")

    # 同时创建JSONL格式
    jsonl_file = raw_dir / "sample_artifacts.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for artifact in SAMPLE_ARTIFACTS:
            f.write(json.dumps(artifact, ensure_ascii=False) + "\n")

    print(f"JSONL格式数据已创建: {jsonl_file}")

    # 创建Markdown示例
    md_content = """# 三星堆青铜大立人

## 基本信息
- 朝代：商代晚期
- 年代：公元前1200年左右
- 文化：三星堆文化

## 物理特征
- 材质：青铜
- 工艺：分铸法
- 尺寸：通高262厘米，人像高180厘米

## 出处
- 出土地点：四川广汉三星堆遗址
- 发现年份：1986年
- 收藏机构：三星堆博物馆

## 描述
三星堆青铜大立人是三星堆遗址出土的最大青铜人像，被誉为"铜像之王"。立人头戴高冠，身着华服，双手夸张地环握，似乎握有某种象征权力的器物。面部特征独特，具有三星堆文化特有的纵目造型。整座雕像造型奇特，工艺精湛，是古蜀文明的杰出代表。
"""

    md_file = raw_dir / "sanxingdui.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Markdown示例已创建: {md_file}")


def main():
    """主函数"""
    create_sample_data()
    print("\n示例数据创建完成！")
    print("运行以下命令构建索引：")
    print("  python scripts/build_index.py --data-dir ./data/raw")


if __name__ == "__main__":
    main()
