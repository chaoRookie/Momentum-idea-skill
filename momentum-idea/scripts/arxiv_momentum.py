#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arxiv_momentum.py —— 用 arXiv 看一个方向最近几个月的势头。

ICLR 一年才出一次结果，看不到最近半年的变化；arXiv 每天都有新论文，正好补上。
做法：按月统计「标题或摘要里出现关键词」的论文数，再除以同期同类目的论文总数（大盘），
得到月度份额；比较最近 6 个月和之前 6 个月，判断在升温、持平还是降温。

注意：arXiv 统计的是预印本，不区分录用 / 拒稿，所以只能看热度，看不了录用红利。
只用 Python 标准库；arXiv 要求请求间隔 ≥ 3 秒，12 个月约需 1.5 分钟，完整月份的结果会缓存。

示例：
  python3 arxiv_momentum.py --direction gnn
  python3 arxiv_momentum.py --direction 大模型Agent --months 18
  python3 arxiv_momentum.py --name "时间序列预测" --phrases "time series forecasting"
  python3 arxiv_momentum.py --direction gnn --and-phrases "large language model" LLM   # 交叉：GNN × LLM
  python3 arxiv_momentum.py --direction rag --cats cs.CL cs.IR --json
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
DIRECTIONS_FILE = SKILL_DIR / "data" / "directions.json"
API = "https://export.arxiv.org/api/query"
USER_AGENT = "momentum-idea-skill/1.0 (+https://github.com/chaoRookie/Momentum-idea-skill)"
DEFAULT_CATS = ["cs.LG", "cs.AI", "cs.CL", "cs.CV"]
DELAY = 3.5  # arXiv API 使用条款：请求间隔至少 3 秒
TOTAL_RE = re.compile(r"<opensearch:totalResults[^>]*>(\d+)</opensearch:totalResults>")


def cache_file():
    base = os.environ.get("MOMENTUM_IDEA_CACHE")
    root = Path(base).expanduser() if base else Path.home() / ".cache" / "momentum-idea"
    d = root / "arxiv"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        d = Path(tempfile.gettempdir()) / "momentum-idea" / "arxiv"
        d.mkdir(parents=True, exist_ok=True)
    return d / "counts.json"


def load_cache():
    try:
        return json.loads(cache_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_cache(cache):
    cache_file().write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")


def load_directions():
    with open(DIRECTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["directions"]


def resolve_direction(query, directions):
    q = query.strip().lower().replace(" ", "")
    norm = lambda s: s.lower().replace(" ", "")
    exact = [d for d in directions if norm(d["id"]) == q or norm(d["name"]) == q
             or any(norm(a) == q for a in d.get("aliases", []))]
    if exact:
        return exact
    return [d for d in directions if q in norm(d["name"]) or any(q in norm(a) for a in d.get("aliases", []))]


def _or_group(phrases):
    terms = []
    for p in phrases:
        p = p.replace('"', "").strip()
        if p:
            terms.append('ti:"{0}" OR abs:"{0}"'.format(p))
    return "(" + " OR ".join(terms) + ")" if terms else None


def build_query(phrases, cats, and_groups=()):
    topic = " AND ".join(g for g in [_or_group(phrases)] + [_or_group(x) for x in and_groups] if g) or None
    cat = "(" + " OR ".join("cat:" + c for c in cats) + ")"
    return topic, cat


def month_range(until, months):
    """返回 [(yyyy, mm), ...]，截止到 until（含），共 months 个月。"""
    y, m = until
    out = []
    for _ in range(months):
        out.append((y, m))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return list(reversed(out))


def last_day(y, m):
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return (nxt - date(y, m, 1)).days


_last_call = [0.0]


def fetch_total(search_query, retries=2):
    # max_results=0 会让 arXiv 报 500，所以取 1 条，只读 totalResults
    params = urllib.parse.urlencode({"search_query": search_query, "start": 0, "max_results": 1})
    url = API + "?" + params
    wait = 5.0
    for attempt in range(retries + 1):
        gap = time.time() - _last_call[0]
        if gap < DELAY:
            time.sleep(DELAY - gap)
        _last_call[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8", "replace")
            m = TOTAL_RE.search(body)
            if m:
                return int(m.group(1))
            err = "返回内容里没有 totalResults"
        except urllib.error.HTTPError as e:
            err = "HTTP {}".format(e.code)
        except (urllib.error.URLError, OSError) as e:
            err = str(e)
        if attempt < retries:
            time.sleep(wait)
            wait *= 2
    raise RuntimeError(err)


def count_month(topic, cat, y, m, cache, today):
    stamp = "submittedDate:[{0}{1:02d}010000 TO {0}{1:02d}{2:02d}2359]".format(y, m, last_day(y, m))
    complete = (y, m) < (today.year, today.month)
    out = {}
    for key, q in (("topic", "{} AND {} AND {}".format(topic, cat, stamp)), ("base", "{} AND {}".format(cat, stamp))):
        h = hashlib.sha1(q.encode("utf-8")).hexdigest()
        if complete and h in cache:
            out[key] = cache[h]
            continue
        out[key] = fetch_total(q)
        if complete:
            cache[h] = out[key]
    return out["topic"], out["base"]


def summarize(rows):
    n = len(rows)
    half = n // 2
    first, second = rows[:n - half], rows[n - half:]
    def agg(part):
        t = sum(r["topic"] for r in part)
        b = sum(r["base"] for r in part)
        return t, b, (t / b if b else 0.0)
    t1, b1, s1 = agg(first)
    t2, b2, s2 = agg(second)
    rel = (s2 - s1) / s1 if s1 else None
    if rel is None:
        trend = "数据不足"
    elif rel >= 0.15:
        trend = "升温"
    elif rel <= -0.15:
        trend = "降温"
    else:
        trend = "持平"
    return {
        "early_months": "{}–{}".format(first[0]["month"], first[-1]["month"]),
        "recent_months": "{}–{}".format(second[0]["month"], second[-1]["month"]),
        "early_count": t1, "recent_count": t2,
        "count_growth": (t2 - t1) / t1 if t1 else None,
        "base_growth": (b2 - b1) / b1 if b1 else None,
        "early_share": s1, "recent_share": s2, "share_change_rel": rel, "trend": trend,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="arXiv 月度势头：方向论文数 ÷ 同类目论文总数")
    ap.add_argument("-d", "--direction", help="内置方向 id、中文名或别名")
    ap.add_argument("--name", help="自定义方向的显示名")
    ap.add_argument("--phrases", nargs="+", help="自定义关键词（英文）")
    ap.add_argument("--and-phrases", nargs="+", action="append", default=[], metavar="PHRASE",
                    help="再加一组必须同时命中的关键词（可重复），用于交叉方向")
    ap.add_argument("--cats", nargs="+", default=DEFAULT_CATS,
                    help="arXiv 类目，默认 cs.LG cs.AI cs.CL cs.CV；机器人加 cs.RO，检索加 cs.IR，金融加 q-fin.CP q-fin.TR 等")
    ap.add_argument("--months", type=int, default=12, help="统计最近几个完整月份，默认 12")
    ap.add_argument("--until", help="截止月份 YYYY-MM，默认上个月")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.phrases:
        name, phrases = args.name or " / ".join(args.phrases), args.phrases
    elif args.direction:
        found = resolve_direction(args.direction, load_directions())
        if len(found) != 1:
            names = "、".join(d["id"] for d in found) if found else "无"
            print("❌ 方向「{}」匹配结果：{}。请写得更具体，或用 --phrases 自定义关键词。".format(args.direction, names), file=sys.stderr)
            return 2
        name, phrases = found[0]["name"], found[0]["phrases"]
    else:
        print("❌ 请指定 --direction 或 --phrases", file=sys.stderr)
        return 2
    if args.and_phrases:
        name = args.name or " × ".join([name] + [" / ".join(g) for g in args.and_phrases])

    today = date.today()
    if args.until:
        y, m = (int(x) for x in args.until.split("-"))
    else:
        y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    months = month_range((y, m), max(2, args.months))
    topic, cat = build_query(phrases, args.cats, args.and_phrases)
    cache = load_cache()
    rows = []
    print("⏳ 正在查询 arXiv（{} 个月 × 2 次请求，每次间隔 {} 秒）…".format(len(months), DELAY), file=sys.stderr, flush=True)
    try:
        for (yy, mm) in months:
            t, b = count_month(topic, cat, yy, mm, cache, today)
            rows.append({"month": "{}-{:02d}".format(yy, mm), "topic": t, "base": b, "share": t / b if b else 0.0})
            save_cache(cache)
    except RuntimeError as e:
        print("❌ arXiv 暂时不可用（{}），可能被限流或网络受限。\n"
              "   · 过几分钟重试（已查到的月份会缓存）；\n"
              "   · 或改用网页搜索估计近期热度，并在报告里注明\"近期势头未用 arXiv 数据核实\"。".format(e), file=sys.stderr)
        return 3

    summary = summarize(rows)
    if args.json:
        print(json.dumps({"name": name, "phrases": phrases, "and": args.and_phrases, "cats": args.cats,
                          "rows": rows, "summary": summary},
                         ensure_ascii=False, indent=1))
        return 0

    pct = lambda x, d=2: "—" if x is None else "{:.{}f}%".format(100 * x, d)
    sgn = lambda x: "—" if x is None else "{:+.0f}%".format(100 * x)
    print("## 📈 arXiv 月度势头：{}".format(name))
    print()
    also = "".join("，且同时含 " + " / ".join("`{}`".format(p) for p in g) for g in args.and_phrases)
    print("- 口径：标题或摘要含 {}{}；类目 {}；按首次提交月份统计".format(
        " / ".join("`{}`".format(p) for p in phrases), also, " ∪ ".join(args.cats)))
    print("- arXiv 是预印本，只反映热度，看不了录用红利")
    print()
    print("| 月份 | 方向论文数 | 同类目总数 | 份额 |")
    print("|---|---|---|---|")
    for r in rows:
        print("| {} | {} | {} | {} |".format(r["month"], r["topic"], r["base"], pct(r["share"])))
    print()
    s = summary
    print("### 结论：最近 {} 个月 vs 之前 {} 个月 → **{}**".format(
        len(rows) // 2, len(rows) - len(rows) // 2, s["trend"]))
    print("- 论文数 {} → {}（{}），同类目总数变化 {}".format(
        s["early_count"], s["recent_count"], sgn(s["count_growth"]), sgn(s["base_growth"])))
    print("- 份额 {} → {}（相对变化 {}；±15% 以内算持平）".format(
        pct(s["early_share"]), pct(s["recent_share"]), sgn(s["share_change_rel"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
