"""
年代标准化工具

支持多种年代格式的解析和标准化
"""

import re

from kg_search.utils import get_logger

logger = get_logger(__name__)


# 日治时期年号转西元（基准年）
JAPANESE_ERA_BASE = {
    "明治": 1868,
    "大正": 1912,
    "昭和": 1926,
    "平成": 1989,
    "令和": 2019,
}

# 民国纪年基准
MINGUO_BASE = 1911

# 中国朝代关键词
DYNASTY_KEYWORDS = {
    "清代": r"清|同治|光緒|光绪|宣統|宣统|道光|咸豐|咸丰|嘉慶|嘉庆|乾隆|雍正|康熙|順治|顺治",
    "日治時期": r"日治|明治|大正|昭和",
    "民國": r"民[國国]",
    "明代": r"明|洪武|永樂|永乐|宣德|正統|正统|景泰|天順|天顺|成化|弘治|正德|嘉靖|隆慶|隆庆|萬曆|万历|泰昌|天啟|天启|崇禎|崇祯",
    "宋代": r"宋|建隆|乾德|開寶|开宝|太平興國|太平兴国|雍熙|端拱|淳化|至道|咸平|景德|大中祥符|天禧|乾興|乾兴|天聖|天圣|明道|景祐|寶元|宝元|康定|慶曆|庆历|皇祐|至和|嘉祐|治平|熙寧|熙宁|元豐|元丰|元祐|紹聖|绍圣|元符|建中靖國|建中靖国|崇寧|崇宁|大觀|大观|政和|重和|宣和|靖康|建炎|紹興|绍兴|隆興|隆兴|乾道|淳熙|紹熙|绍熙|慶元|庆元|嘉泰|開禧|开禧|嘉定|寶慶|宝庆|紹定|绍定|端平|嘉熙|淳祐|寶祐|宝祐|開慶|开庆|景定|咸淳|德祐|景炎|祥興|祥兴",
    "唐代": r"唐|武德|貞觀|贞观|永徽|顯慶|显庆|龍朔|龙朔|麟德|乾封|總章|总章|咸亨|上元|儀鳳|仪凤|調露|调露|永隆|開耀|开耀|永淳|弘道|嗣聖|嗣圣|文明|光宅|垂拱|永昌|載初|载初|天授|如意|長壽|长寿|延載|延载|證聖|证圣|天冊萬歲|天册万岁|萬歲登封|万岁登封|萬歲通天|万岁通天|神功|聖曆|圣历|久視|久视|大足|長安|长安|神龍|神龙|景龍|景龙|唐隆|景雲|景云|太極|太极|延和|先天|開元|开元|天寶|天宝|至德|乾元|上元|寶應|宝应|廣德|广德|永泰|大曆|大历|建中|興元|兴元|貞元|贞元|永貞|永贞|元和|長慶|长庆|寶曆|宝历|大和|開成|开成|會昌|会昌|大中|咸通|乾符|廣明|广明|中和|光啟|光启|文德|龍紀|龙纪|大順|大顺|景福|乾寧|乾宁|光化|天復|天复|天祐|天祐",
    "漢代": r"漢|汉|西漢|西汉|東漢|东汉",
    "戰國": r"戰國|战国",
    "春秋": r"春秋",
    "商代": r"商|殷",
    "周代": r"周|西周|東周|东周",
    "秦代": r"秦",
    "三國": r"三國|三国|魏|蜀|吳|吴",
    "晉代": r"晉|晋|西晉|西晋|東晉|东晋",
    "南北朝": r"南北朝|南朝|北朝|劉宋|刘宋|南齊|南齐|梁|陳|陈|北魏|東魏|东魏|西魏|北齊|北齐|北周",
    "隋代": r"隋",
    "五代十國": r"五代|十國|十国|後梁|后梁|後唐|后唐|後晉|后晋|後漢|后汉|後周|后周",
    "元代": r"元|至元|大德|延祐|至治|泰定|天順|天顺|致和|天曆|天历|至順|至顺|元統|元统|至元|至正",
    "新石器時代": r"新石器",
    "舊石器時代": r"舊石器|旧石器",
    "青銅時代": r"青銅|青铜",
}


def normalize_year_to_ce(age_str: str) -> tuple[int | None, int | None]:
    """
    将各种年代格式标准化为西元年份

    支持格式:
    - 西元1932年, 公元前1600年
    - 日治明治39年, 日治昭和7年
    - 民國65年
    - 清同治九年(1870)
    - 1949
    - 1906-1932 (年份范围)

    Args:
        age_str: 年代描述字符串

    Returns:
        (year_start, year_end) - 年份范围，单一年份时 year_end 为 None
    """
    if not age_str:
        return None, None

    # 清理字符串
    age_str = age_str.strip()

    # 1. 尝试匹配带括号的西元年份 如 "清同治九年(1870)"
    bracket_match = re.search(r"\((\d{3,4})\)", age_str)
    if bracket_match:
        return int(bracket_match.group(1)), None

    # 2. 尝试匹配西元纪年 如 "西元1932年"
    ce_match = re.search(r"西元\s*(\d{3,4})\s*年", age_str)
    if ce_match:
        return int(ce_match.group(1)), None

    # 3. 尝试匹配公元纪年（含公元前）
    ce_match2 = re.search(r"公元(前)?(\d{3,4})年", age_str)
    if ce_match2:
        year = int(ce_match2.group(2))
        if ce_match2.group(1):  # 公元前
            return -year, None
        return year, None

    # 4. 尝试匹配日治时期年号
    jp_era_match = re.search(r"(明治|大正|昭和|平成|令和)\s*(\d{1,2})\s*年", age_str)
    if jp_era_match:
        era = jp_era_match.group(1)
        year_num = int(jp_era_match.group(2))
        if era in JAPANESE_ERA_BASE:
            # 日本年号第1年对应基准年
            return JAPANESE_ERA_BASE[era] + year_num - 1, None

    # 5. 尝试匹配民国纪年
    minguo_match = re.search(r"民[國国]\s*(\d{1,3})\s*年", age_str)
    if minguo_match:
        year_num = int(minguo_match.group(1))
        return MINGUO_BASE + year_num, None

    # 6. 尝试匹配纯数字年份
    pure_year_match = re.search(r"^(\d{4})$", age_str.strip())
    if pure_year_match:
        return int(pure_year_match.group(1)), None

    # 7. 尝试匹配年份范围 如 "1906-1932"
    range_match = re.search(r"(\d{4})\s*[-~至到]\s*(\d{4})", age_str)
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))

    # 无法解析
    logger.debug(f"无法解析年代: {age_str}")
    return None, None


def extract_dynasty(age_str: str) -> str | None:
    """
    从年代描述中提取朝代/时期

    Args:
        age_str: 年代描述字符串

    Returns:
        朝代名称，无法识别返回 None
    """
    if not age_str:
        return None

    for dynasty_name, pattern in DYNASTY_KEYWORDS.items():
        if re.search(pattern, age_str):
            return dynasty_name

    return None


def get_dynasty_range(dynasty: str) -> tuple[int | None, int | None]:
    """
    获取朝代的大致年份范围

    Args:
        dynasty: 朝代名称

    Returns:
        (start_year, end_year) 年份范围
    """
    dynasty_ranges = {
        "新石器時代": (-10000, -2000),
        "舊石器時代": (-2500000, -10000),
        "青銅時代": (-2000, -500),
        "商代": (-1600, -1046),
        "周代": (-1046, -256),
        "春秋": (-770, -476),
        "戰國": (-475, -221),
        "秦代": (-221, -207),
        "漢代": (-206, 220),
        "三國": (220, 280),
        "晉代": (265, 420),
        "南北朝": (420, 589),
        "隋代": (581, 618),
        "唐代": (618, 907),
        "五代十國": (907, 979),
        "宋代": (960, 1279),
        "元代": (1271, 1368),
        "明代": (1368, 1644),
        "清代": (1644, 1912),
        "日治時期": (1895, 1945),
        "民國": (1912, None),  # 至今
    }

    return dynasty_ranges.get(dynasty, (None, None))
