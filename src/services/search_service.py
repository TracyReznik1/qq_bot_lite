import logging
import re
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlparse

from src.config import config
from src.services.url_fetch_service import fetch_url

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

logger = logging.getLogger("qq-bot")
QUERY_MAX_CHARS = 120
SUMMARY_MAX_CHARS = 500
SEARCH_FETCH_RESULT_LIMIT = 3
PAGE_EXCERPT_MAX_CHARS = 1200
SEARCH_CONTEXT_MAX_CHARS = 4500
COMPRESSED_PAGE_EXCERPT_MAX_CHARS = 450
HIGH_VALUE_SOURCE_TYPES = {"官方/文档", "机构/公共来源"}
MEDIUM_VALUE_SOURCE_TYPES = {"代码/项目源", "百科", "新闻/媒体"}


@dataclass(frozen=True)
class SearchResult:
    ok: bool
    status: str
    text: str


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _retrieval_date() -> str:
    return date.today().isoformat()


def _query_intent(query: str) -> tuple[str, str]:
    text = str(query or "").casefold()
    current_markers = (
        "最新",
        "最近",
        "当前",
        "现在",
        "今天",
        "实时",
        "新闻",
        "价格",
        "版本",
        "发布",
        "更新",
        "趋势",
        "舆论",
        "评价",
        "热度",
        "爆火",
        "走红",
        "current",
        "latest",
        "today",
        "news",
        "price",
        "version",
        "release",
        "update",
        "trend",
        "trending",
        "viral",
        "public opinion",
    )
    if any(marker in text for marker in current_markers):
        return "current", "high"
    return "general", "normal"


def _truncate_text(text: str, limit: int) -> str:
    text = _collapse_spaces(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    if " " in cut:
        word_cut = cut.rsplit(" ", 1)[0].rstrip()
        if len(word_cut) >= limit // 2:
            cut = word_cut
    return cut.rstrip("，,。；;：:")


def _source_domain(url: str) -> str:
    try:
        return (urlparse(str(url or "")).hostname or "").lower()
    except ValueError:
        return ""


def _source_type(url: str) -> str:
    try:
        parsed = urlparse(str(url or ""))
    except ValueError:
        return "未知"
    domain = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    if not domain:
        return "未知"
    if (
        domain.startswith(("docs.", "developer.", "developers."))
        or domain in {"docs.github.com", "learn.microsoft.com", "developer.mozilla.org"}
        or "/docs" in path
        or "/documentation" in path
    ):
        return "官方/文档"
    if domain == "github.com" or domain.endswith(".github.com"):
        return "代码/项目源"
    if domain.endswith(".gov") or domain.endswith(".gov.cn") or domain.endswith(".edu") or domain.endswith(".edu.cn"):
        return "机构/公共来源"
    if domain.endswith("wikipedia.org") or domain.endswith("wikidata.org"):
        return "百科"
    if (
        domain == "reddit.com"
        or domain.endswith(".reddit.com")
        or domain in {"x.com", "twitter.com", "weibo.com", "zhihu.com", "tieba.baidu.com"}
        or domain.endswith(".twitter.com")
        or domain.endswith(".weibo.com")
        or domain.endswith(".zhihu.com")
    ):
        return "社交/论坛"
    if domain in {
        "apnews.com",
        "bbc.com",
        "bloomberg.com",
        "nytimes.com",
        "reuters.com",
        "thepaper.cn",
        "xinhua.net",
    } or domain.endswith(".reuters.com"):
        return "新闻/媒体"
    return "普通网页"


def _source_priority(source_type: str) -> str:
    if source_type in HIGH_VALUE_SOURCE_TYPES:
        return "high"
    if source_type in MEDIUM_VALUE_SOURCE_TYPES:
        return "medium"
    return "low"


def _source_priority_score(source_type: str) -> int:
    priority = _source_priority(source_type)
    if priority == "high":
        return 3
    if priority == "medium":
        return 2
    if source_type == "普通网页":
        return 1
    return 0


def _source_priority_summary() -> str:
    return "high=官方/文档或机构公共来源；medium=代码/项目源、百科、新闻/媒体；low=普通网页或社交论坛线索"


def _query_terms(query: str) -> list[str]:
    text = str(query or "").casefold()
    terms = re.findall(r"[a-z0-9][a-z0-9._-]*|[\u4e00-\u9fff]{2,}", text)
    stop_terms = {
        "latest",
        "current",
        "today",
        "news",
        "最新",
        "最近",
        "当前",
        "现在",
    }
    cleaned = []
    seen = set()
    for term in terms:
        term = term.strip("._-")
        if len(term) < 2 or term in stop_terms or term in seen:
            continue
        seen.add(term)
        cleaned.append(term)
    return cleaned


def _result_relevance(query_terms: list[str], result: dict[str, str]) -> tuple[str, list[str]]:
    if not query_terms:
        return "medium", []
    text = (
        " ".join(
            str(result.get(key) or "")
            for key in ("title", "body", "href", "source_domain", "page_title", "page_excerpt")
        )
    ).casefold()
    matched = [term for term in query_terms if term in text]
    if not matched:
        return "low", []
    if len(matched) == len(query_terms) or len(matched) >= 2:
        return "high", matched
    return "medium", matched


def _relevance_score(relevance: str) -> int:
    if relevance == "high":
        return 3
    if relevance == "medium":
        return 2
    return 0


def _relevance_counts(results: list[dict[str, str]]) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "low": 0}
    for result in results:
        relevance = result.get("relevance") or "medium"
        if relevance not in counts:
            relevance = "medium"
        counts[relevance] += 1
    return counts


def _relevance_count_summary(results: list[dict[str, str]]) -> str:
    counts = _relevance_counts(results)
    return f"high {counts['high']}；medium {counts['medium']}；low {counts['low']}"


def _relevance_quality(results: list[dict[str, str]]) -> str:
    counts = _relevance_counts(results)
    if counts["high"] >= 2:
        return "strong"
    if counts["high"] >= 1 or counts["medium"] >= 2:
        return "medium"
    return "weak"


def _relevance_quality_score(quality: str) -> int:
    if quality == "strong":
        return 3
    if quality == "medium":
        return 2
    return 1


def _raw_relevance_quality(query: str, results: list[dict[str, str]]) -> str:
    terms = _query_terms(query)
    scored_results = []
    for result in results:
        relevance, _matched = _result_relevance(terms, result)
        scored = dict(result)
        scored["relevance"] = relevance
        scored_results.append(scored)
    return _relevance_quality(scored_results)


def _relevance_summary(query_terms: list[str]) -> str:
    if not query_terms:
        return "未提取到稳定关键词；主要按来源优先级和原始排序判断。"
    return f"基于查询词 {'、'.join(query_terms)} 在标题、摘要、链接和正文摘录中的命中情况；high 优先用于回答，low 只作弱线索。"


def _query_specificity(query_terms: list[str]) -> str:
    if len(query_terms) >= 3:
        return "high"
    if len(query_terms) == 2:
        return "medium"
    if not query_terms:
        return "low"
    term = query_terms[0]
    if re.search(r"\d", term) or (re.search(r"[a-z]", term) and len(term) >= 6):
        return "medium"
    return "low"


def _query_specificity_hint(specificity: str) -> str:
    if specificity == "high":
        return "搜索词包含多个稳定限定词，结果可按相关性和来源质量综合使用。"
    if specificity == "medium":
        return "搜索词有一定限定性；回答时仍需注意同名实体或语境差异。"
    return "搜索词较短或过泛；回答时需要说明可能存在歧义，必要时请用户补充限定词。"


def search_query_specificity(query: str) -> str:
    return _query_specificity(_query_terms(query))


def search_query_specificity_score(query: str) -> int:
    specificity = search_query_specificity(query)
    if specificity == "high":
        return 3
    if specificity == "medium":
        return 2
    return 1


def _fetch_line_value(text: str, label: str) -> str:
    prefix = f"{label}："
    for line in str(text or "").splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _fetch_excerpt(text: str) -> str:
    marker = "正文摘录："
    raw_text = str(text or "")
    if marker not in raw_text:
        return ""
    return _truncate_text(raw_text.split(marker, 1)[1], PAGE_EXCERPT_MAX_CHARS)


def _source_type_summary(results: list[dict[str, str]]) -> str:
    counts: dict[str, int] = {}
    for result in results:
        source_type = result.get("source_type") or "未知"
        counts[source_type] = counts.get(source_type, 0) + 1
    return "；".join(f"{source_type} {count}" for source_type, count in counts.items())


def _evidence_quality(results: list[dict[str, str]], fetch_successes: int) -> str:
    source_types = {result.get("source_type") or "" for result in results}
    if fetch_successes >= 2 and source_types & HIGH_VALUE_SOURCE_TYPES:
        return "strong"
    if fetch_successes >= 1 or source_types & HIGH_VALUE_SOURCE_TYPES:
        return "medium"
    return "weak"


def _answer_guidance(quality: str, relevance_quality: str = "medium") -> str:
    if relevance_quality == "weak":
        return "当前结果相关性偏弱，只能说明没有可靠确认；不要把低相关结果包装成直接答案。"
    if quality == "strong":
        return "优先依据正文摘录和官方/机构来源；不同来源冲突时说明冲突和不确定。"
    if quality == "medium":
        return "结合正文摘录、标题和摘要回答；缺少可靠来源支持的细节要说明不确定。"
    return "证据偏弱，只能作为线索；不要断言未被来源支持的细节。"


def _citable_sources(results: list[dict[str, str]]) -> str:
    sources = []
    citable_results = []
    for index, result in enumerate(results, 1):
        source_type = result.get("source_type") or ""
        has_excerpt = bool(result.get("page_excerpt"))
        if not has_excerpt and source_type not in HIGH_VALUE_SOURCE_TYPES:
            continue
        citable_results.append((index, result))

    citable_results.sort(
        key=lambda item: (
            -_source_priority_score(item[1].get("source_type") or ""),
            item[0],
        )
    )
    for index, result in citable_results:
        title = result.get("title") or result.get("page_title") or "无标题"
        domain = result.get("source_domain") or "未知"
        sources.append(f"[{index}] {_truncate_text(title, 80)}（{domain}）")
        if len(sources) >= 3:
            break
    if not sources:
        return "无强引用来源；只能概括搜索结果并说明不确定。"
    return "；".join(sources)


def _citable_source_count(results: list[dict[str, str]]) -> int:
    count = 0
    for result in results:
        source_type = result.get("source_type") or ""
        if result.get("page_excerpt") or source_type in HIGH_VALUE_SOURCE_TYPES:
            count += 1
    return count


def _domain_counts(results: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        domain = result.get("source_domain") or _source_domain(result.get("href") or "")
        domain = domain or "未知"
        counts[domain] = counts.get(domain, 0) + 1
    return counts


def _domain_coverage_summary(results: list[dict[str, str]]) -> str:
    counts = _domain_counts(results)
    total = sum(counts.values())
    if not total:
        return "0 个来源域名"
    dominant_domain, dominant_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return f"{len(counts)} 个来源域名；主要域名 {dominant_domain} {dominant_count}/{total}"


def _domain_diversity(results: list[dict[str, str]]) -> str:
    counts = _domain_counts(results)
    total = sum(counts.values())
    if not total:
        return "unknown"
    if len(counts) == 1 and total > 1:
        return "single_domain"
    if len(counts) < min(3, total):
        return "limited"
    return "diverse"


def _cross_check_summary(results: list[dict[str, str]], fetch_attempts: int, fetch_successes: int) -> str:
    citable_count = _citable_source_count(results)
    source_type_count = len({result.get("source_type") or "未知" for result in results})
    domain_count = len(_domain_counts(results))
    fetch_failures = max(fetch_attempts - fetch_successes, 0)
    return f"{citable_count} 个可引用来源；{source_type_count} 类来源；{fetch_failures} 个结果页面未读到正文；{domain_count} 个来源域名"


def _risk_hint(
    fetch_attempts: int,
    fetch_successes: int,
    relevance_quality: str = "medium",
    domain_diversity: str = "diverse",
) -> str:
    if relevance_quality == "weak":
        return "搜索结果与查询词匹配弱，只能作为线索，不要用来支撑关键事实。"
    if fetch_attempts > fetch_successes:
        return "未读到正文的普通网页只能作为线索，不要用来支撑关键事实。"
    if domain_diversity == "single_domain":
        return "多个结果来自同一域名，不能当作独立交叉验证；关键事实仍需其他来源确认。"
    return "当前可引用来源可支撑基础回答；关键结论仍需优先依据正文摘录。"


def _searchable_result_text(result: dict[str, str]) -> str:
    return " ".join(
        str(result.get(key) or "")
        for key in ("title", "body", "published_date", "page_title", "page_excerpt")
    )


def _normalize_date(value: str) -> str:
    parts = re.split(r"[-/.]", value)
    if len(parts) != 3:
        return value
    year, month, day = parts
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _normalize_published_date(value: object) -> str:
    text = _collapse_spaces(value)
    if not text:
        return ""
    match = re.search(r"(?<!\d)20\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])(?!\d)", text)
    if match:
        return _normalize_date(match.group(0))
    return _truncate_text(text, 80)


def _published_date_from_item(item: dict) -> str:
    for key in ("date", "published_date", "publishedDate", "published", "updated", "updated_date"):
        if key in item:
            published_date = _normalize_published_date(item.get(key))
            if published_date:
                return published_date
    return ""


def _published_date_summary(results: list[dict[str, str]]) -> str:
    dated = [result.get("published_date") or "" for result in results if result.get("published_date")]
    total = len(results)
    standard_dates = [value for value in dated if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", value)]
    if standard_dates:
        return f"{len(dated)}/{total} 个结果含日期；最新 {max(standard_dates)}；最旧 {min(standard_dates)}"
    if dated:
        return f"{len(dated)}/{total} 个结果含日期；未提取到标准日期"
    return f"0/{total} 个结果含日期"


def _standard_published_dates(results: list[dict[str, str]]) -> list[date]:
    dates = []
    for result in results:
        value = result.get("published_date") or ""
        if not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", value):
            continue
        try:
            dates.append(date.fromisoformat(value))
        except ValueError:
            continue
    return dates


def _freshness_risk(query: str, results: list[dict[str, str]]) -> str:
    _intent, freshness = _query_intent(query)
    if freshness != "high":
        return ""
    published_dates = _standard_published_dates(results)
    if not published_dates:
        return "unknown_dates；高时效问题缺少结果发布时间，不能确认是最新信息。"
    try:
        retrieved_on = date.fromisoformat(_retrieval_date())
    except ValueError:
        return "unknown_retrieval_date；检索时间无法解析，高时效问题需继续确认。"

    latest = max(published_dates)
    age_days = (retrieved_on - latest).days
    if age_days > 30:
        return f"stale；最新发布时间 {latest.isoformat()} 距检索时间 {age_days} 天，高时效问题需继续确认。"
    if len(published_dates) < len(results):
        return "partial_dates；部分结果缺少发布时间，高时效问题需优先引用带日期来源。"
    return "current；带日期结果覆盖较好，回答仍需结合检索时间表达。"


def _conflict_values(results: list[dict[str, str]]) -> list[str]:
    dates: set[str] = set()
    versions: set[str] = set()
    for result in results:
        text = _searchable_result_text(result)
        for match in re.findall(r"\b20\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])\b", text):
            dates.add(_normalize_date(match))
        for match in re.findall(r"(?i)\bv\d+(?:\.\d+){1,3}\b", text):
            versions.add(match.lower())

    conflicts: list[str] = []
    if len(dates) > 1:
        conflicts.extend(sorted(dates))
    if len(versions) > 1:
        conflicts.extend(sorted(versions))
    return conflicts


def normalize_search_query(query: str) -> str:
    query = _collapse_spaces(query)
    if not query:
        return ""

    query = re.sub(r"^/(?:search|s)(?:\s+|$)", "", query, flags=re.IGNORECASE).strip()
    has_reason_intent = any(marker in query for marker in ("为什么", "为啥", "为何", "怎么回事", "原因"))
    prefix_patterns = [
        r"^(?:帮我|麻烦你|麻烦|请|可以|能不能|能帮我|给我)?\s*(?:查一下|搜一下|搜索一下|查询一下|查查|搜搜|搜索|查询|查|搜)\s*",
        r"^(?:你知道|知道|请问|问一下)\s*",
        r"^(?:please\s+)?(?:search\s+for|look\s+up|find\s+out|tell\s+me\s+about)\s+",
        r"^(?:what|who|why|how)\s+(?:is|are|was|were|did|does|do)\s+",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in prefix_patterns:
            cleaned = re.sub(pattern, "", query, flags=re.IGNORECASE).strip()
            if cleaned != query:
                query = cleaned
                changed = True

    replacements = [
        "是什么梗",
        "是什么意思",
        "是什么",
        "是谁",
        "为什么",
        "为啥",
        "为何",
        "怎么回事",
        "到底",
        "这么",
        "吗",
        "呀",
        "啊",
        "呢",
    ]
    for phrase in replacements:
        query = query.replace(phrase, " ")
    if not re.search(r"https?://", query, flags=re.IGNORECASE):
        query = re.sub(r"([A-Za-z0-9])([\u4e00-\u9fff])", r"\1 \2", query)
        query = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9])", r"\1 \2", query)
    query = _collapse_spaces(query).strip(" 。！？!，,；;：:")
    if has_reason_intent and query and "原因" not in query:
        query = _collapse_spaces(f"{query} 原因")
    return _truncate_text(query, QUERY_MAX_CHARS)


def normalize_search_items(items: list[dict], max_results: int | None = None) -> list[dict[str, str]]:
    limit = max_results if max_results is not None else config.search_max_results
    limit = max(int(limit or 1), 1)
    normalized = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        title = _collapse_spaces(item.get("title") or "")
        body = _truncate_text(item.get("body") or item.get("content") or item.get("snippet") or "", SUMMARY_MAX_CHARS)
        href = _collapse_spaces(item.get("href") or item.get("url") or "")
        if not href:
            continue
        if not title and not body:
            continue

        url_key = href.rstrip("/").lower()
        if url_key and url_key in seen_urls:
            continue

        title_key = title.lower()
        if title_key and title_key in seen_titles:
            continue

        if url_key:
            seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        normalized_item = {"title": title, "body": body, "href": href}
        published_date = _published_date_from_item(item)
        if published_date:
            normalized_item["published_date"] = published_date
        normalized.append(normalized_item)
        if len(normalized) >= limit:
            break

    return normalized


def format_search_failure(search_type: str, status: str, query: str, message: str) -> str:
    intent, freshness = _query_intent(query)
    return (
        f"搜索状态：{status}\n"
        f"搜索类型：{search_type}\n"
        f"搜索词：{query or '无'}\n"
        f"检索时间：{_retrieval_date()}\n"
        f"搜索意图：{intent}\n"
        f"时效性要求：{freshness}\n"
        f"说明：{message}"
    )


def format_search_success(search_type: str, query: str, entries: list[str], metadata: list[str] | None = None) -> str:
    intent, freshness = _query_intent(query)
    lines = [
        "搜索状态：success",
        f"搜索类型：{search_type}",
        f"搜索词：{query}",
        f"检索时间：{_retrieval_date()}",
        f"搜索意图：{intent}",
        f"时效性要求：{freshness}",
        f"结果数：{len(entries)}",
    ]
    if metadata:
        lines.extend(metadata)
    lines.append("")
    for index, entry in enumerate(entries, 1):
        lines.append(f"{index}. {str(entry or '').strip()}")
    return "\n\n".join(lines)


def _preferred_fetch_indexes(results: list[dict[str, str]]) -> set[int]:
    candidates = []
    for index, result in enumerate(results):
        href = result.get("href") or ""
        if not href.lower().startswith(("http://", "https://")):
            continue
        candidates.append(
            (
                -_relevance_score(result.get("relevance") or ""),
                -_source_priority_score(result.get("source_type") or ""),
                index,
            )
        )
    candidates.sort()
    return {index for _relevance, _priority, index in candidates[:SEARCH_FETCH_RESULT_LIMIT]}


def _enrich_search_results(query: str, results: list[dict[str, str]]) -> list[dict[str, str]]:
    enriched_results = []
    terms = _query_terms(query)

    for result in results:
        enriched = dict(result)
        href = enriched.get("href") or ""
        domain = _source_domain(href)
        enriched["source_domain"] = domain or "未知"
        enriched["source_type"] = _source_type(href)
        enriched["source_priority"] = _source_priority(enriched["source_type"])
        relevance, matched_terms = _result_relevance(terms, enriched)
        enriched["relevance"] = relevance
        if matched_terms:
            enriched["matched_terms"] = ", ".join(matched_terms)
        enriched_results.append(enriched)

    fetch_indexes = _preferred_fetch_indexes(enriched_results)
    for index, enriched in enumerate(enriched_results):
        href = enriched.get("href") or ""
        if index in fetch_indexes:
            try:
                fetch_result = fetch_url(href)
                fetch_text = str(getattr(fetch_result, "text", "") or "")
                fetch_ok = bool(getattr(fetch_result, "ok", False))
                fetch_status = str(
                    getattr(fetch_result, "status", "")
                    or _fetch_line_value(fetch_text, "获取状态")
                    or "unknown"
                )
            except Exception:
                logger.debug("Search result page fetch failed: %s", href)
                fetch_text = ""
                fetch_ok = False
                fetch_status = "request_error"

            enriched["page_fetch_status"] = fetch_status
            enriched["page_fetch_ok"] = "true" if fetch_ok else "false"
            if fetch_ok:
                page_title = _fetch_line_value(fetch_text, "标题")
                page_excerpt = _fetch_excerpt(fetch_text)
                if page_title:
                    enriched["page_title"] = _truncate_text(page_title, 160)
                if page_excerpt:
                    enriched["page_excerpt"] = page_excerpt
            else:
                message = _fetch_line_value(fetch_text, "说明")
                if message:
                    enriched["page_fetch_message"] = _truncate_text(message, 180)

    return enriched_results


def _format_result_entry(index: int, result: dict[str, str], page_excerpt_limit: int) -> str:
    title = result.get("title") or "无标题"
    body = result.get("body") or "无摘要"
    href = result.get("href") or ""
    lines = [
        f"引用编号：[{index}]",
        title,
        f"摘要：{body}",
        f"链接：{href}",
        f"来源域名：{result.get('source_domain') or '未知'}",
        f"来源类型：{result.get('source_type') or '普通网页'}",
        f"来源优先级：{result.get('source_priority') or _source_priority(result.get('source_type') or '')}",
        f"相关性：{result.get('relevance') or 'medium'}",
    ]
    published_date = result.get("published_date") or ""
    if published_date:
        lines.insert(4, f"发布时间：{published_date}")
    matched_terms = result.get("matched_terms") or ""
    if matched_terms:
        lines.append(f"命中词：{matched_terms}")
    fetch_status = result.get("page_fetch_status") or ""
    if fetch_status:
        lines.append(f"页面读取状态：{fetch_status}")
    page_title = result.get("page_title") or ""
    if page_title and page_title != title:
        lines.append(f"页面标题：{page_title}")
    page_excerpt = result.get("page_excerpt") or ""
    if page_excerpt:
        lines.append(f"正文摘录：{_truncate_text(page_excerpt, page_excerpt_limit)}")
    fetch_message = result.get("page_fetch_message") or ""
    if fetch_message:
        lines.append(f"页面读取说明：{fetch_message}")
    return "\n".join(lines)


def _format_entries(results: list[dict[str, str]], page_excerpt_limit: int) -> list[str]:
    return [
        _format_result_entry(index, result, page_excerpt_limit)
        for index, result in enumerate(results, 1)
    ]


def _with_compression_metadata(metadata: list[str]) -> list[str]:
    compressed = list(metadata)
    insert_at = next(
        (
            index
            for index, line in enumerate(compressed)
            if line.startswith("回答建议：")
        ),
        len(compressed),
    )
    compressed[insert_at:insert_at] = [
        "结果压缩：enabled",
        "压缩说明：长正文摘录已压缩以控制上下文长度",
    ]
    return compressed


def _format_success(query: str, results: list[dict[str, str]]) -> str:

    fetch_successes = sum(1 for result in results if result.get("page_fetch_ok") == "true")
    fetch_attempts = sum(1 for result in results if result.get("page_fetch_status"))
    quality = _evidence_quality(results, fetch_successes)
    relevance_quality = _relevance_quality(results)
    conflicts = _conflict_values(results)
    terms = _query_terms(query)
    query_specificity = _query_specificity(terms)
    domain_diversity = _domain_diversity(results)
    freshness_risk = _freshness_risk(query, results)
    metadata = [
        f"页面读取数：{fetch_successes}/{fetch_attempts}" if fetch_attempts else "页面读取数：0/0",
        f"证据质量：{quality}",
        f"来源类型汇总：{_source_type_summary(results)}",
        f"来源优先级说明：{_source_priority_summary()}",
        f"相关性说明：{_relevance_summary(terms)}",
        f"查询具体度：{query_specificity}",
        f"查询提示：{_query_specificity_hint(query_specificity)}",
        f"相关性质量：{relevance_quality}",
        f"相关性汇总：{_relevance_count_summary(results)}",
        f"发布时间覆盖：{_published_date_summary(results)}",
        f"域名覆盖：{_domain_coverage_summary(results)}",
        f"域名集中风险：{domain_diversity}",
        "引用方式：回答关键事实时可用 [1]、[2] 等编号标注依据；不要引用页面读取失败且没有正文摘录的普通网页。",
        f"可引用来源：{_citable_sources(results)}",
        f"交叉验证：{_cross_check_summary(results, fetch_attempts, fetch_successes)}",
        f"风险提示：{_risk_hint(fetch_attempts, fetch_successes, relevance_quality, domain_diversity)}",
        f"回答建议：{_answer_guidance(quality, relevance_quality)}",
    ]
    if freshness_risk:
        metadata.insert(10, f"时效性风险：{freshness_risk}")
    if conflicts:
        metadata.extend(
            [
                f"疑似冲突：版本/日期存在多个不同值：{'、'.join(conflicts)}",
                "冲突处理：不要直接合并冲突信息；优先引用官方/机构来源，并说明其他来源可能过时或不一致。",
            ]
        )
    entries = _format_entries(results, PAGE_EXCERPT_MAX_CHARS)
    text = format_search_success("web", query, entries, metadata=metadata)
    if len(text) <= SEARCH_CONTEXT_MAX_CHARS:
        return text

    compressed_entries = _format_entries(results, COMPRESSED_PAGE_EXCERPT_MAX_CHARS)
    compressed_metadata = _with_compression_metadata(metadata)
    return format_search_success("web", query, compressed_entries, metadata=compressed_metadata)


def _tavily_search(query: str) -> list[dict]:
    if TavilyClient is None or not config.tavily_api_key:
        return []
    try:
        client = TavilyClient(api_key=config.tavily_api_key, proxies=config.proxies)
        response = client.search(
            query,
            search_depth="basic",
            max_results=config.search_max_results,
        )
        results = []
        for item in response.get("results", []):
            title = item.get("title") or ""
            content = item.get("content") or ""
            url = item.get("url") or ""
            if title and content:
                results.append({"title": title, "body": content[:500], "href": url})
        return results
    except Exception:
        logger.debug("Tavily search failed, falling back to ddgs")
        raise


def _ddgs_search(query: str) -> list[dict]:
    if DDGS is None:
        return []
    last_error = None
    for use_proxy in (config.proxy_url or None, None):
        try:
            with DDGS(proxy=use_proxy, timeout=config.request_timeout) as ddgs:
                return list(ddgs.text(query, max_results=config.search_max_results))
        except Exception as error:
            last_error = error
            if use_proxy is None:
                logger.debug("ddgs search failed")
    if last_error is not None:
        raise last_error
    return []


def search(query: str) -> SearchResult:
    query = normalize_search_query(query)
    if not query:
        return SearchResult(
            ok=False,
            status="empty_query",
            text=format_search_failure("web", "empty_query", query, "没有可搜索的关键词。"),
        )

    tavily_available = TavilyClient is not None and bool(config.tavily_api_key)
    ddgs_available = DDGS is not None
    if not tavily_available and not ddgs_available:
        return SearchResult(
            ok=False,
            status="missing_dependency",
            text=format_search_failure("web", "missing_dependency", query, "搜索组件未安装。需要 ddgs 或 tavily-python。"),
        )

    provider_attempts = 0
    provider_errors = 0
    results = []
    normalized_results: list[dict[str, str]] = []

    if tavily_available:
        provider_attempts += 1
        try:
            results = _tavily_search(query)
            normalized_results = normalize_search_items(results)
        except Exception:
            provider_errors += 1
            logger.debug("Tavily search failed")

    should_try_ddgs = not normalized_results
    if normalized_results and ddgs_available:
        should_try_ddgs = _raw_relevance_quality(query, normalized_results) == "weak"

    # 优先 Tavily；失败、未配置或结果明显弱相关时尝试 ddgs
    if should_try_ddgs and ddgs_available:
        provider_attempts += 1
        try:
            ddgs_results = _ddgs_search(query)
            ddgs_normalized = normalize_search_items(ddgs_results)
            if not normalized_results:
                results = ddgs_results
                normalized_results = ddgs_normalized
            elif ddgs_normalized and (
                _relevance_quality_score(_raw_relevance_quality(query, ddgs_normalized))
                > _relevance_quality_score(_raw_relevance_quality(query, normalized_results))
            ):
                results = ddgs_results
                normalized_results = ddgs_normalized
        except Exception:
            provider_errors += 1
            logger.debug("ddgs search failed")

    if normalized_results:
        enriched_results = _enrich_search_results(query, normalized_results)
        return SearchResult(ok=True, status="success", text=_format_success(query, enriched_results))

    if provider_attempts and provider_errors == provider_attempts:
        return SearchResult(
            ok=False,
            status="provider_error",
            text=format_search_failure("web", "provider_error", query, "搜索服务暂时不可用，请稍后再试。"),
        )
    return SearchResult(
        ok=False,
        status="no_results",
        text=format_search_failure("web", "no_results", query, "没有搜到有用结果。"),
    )


def web_search(query: str) -> str:
    return search(query).text


def has_search_results(search_result: SearchResult) -> bool:
    return search_result.ok
