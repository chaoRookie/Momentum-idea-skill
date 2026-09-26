#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iclr_momentum.py —— 用 ICLR 2017–2026 的录用 / 拒稿记录，算一个研究方向的"科研 momentum"。

指标沿用《做科研越来越像炒股：4 万篇 ICLR 论文回测》（鸭基米德，PaperWeekly 转载）：
  1. 份额          = 该方向论文数 ÷ 当年已决论文总数
  2. 份额年度变化   = 今年份额 − 去年份额（百分点），就是科研圈的 momentum
  3. 相对录用率差   = 该方向录用率 − 同年不做该方向的论文录用率（百分点，附 95% 区间）
在此基础上再给：增长去向（近 N 年新增的论文有多少落在录用里）、处境标签、数据等级。

数据：GitHub 上公开的 papercopilot/paperlists（ICLR 每篇论文的标题、摘要、状态）。
口径：已决论文 = 录用 + 拒稿；撤稿、desk reject 不计入；2017–2018 的 "Invite to Workshop
      Track" 算作未进主会（拒稿）；一篇论文可以同时属于多个方向。
只用 Python 标准库。首次运行会下载约 300MB 原始数据，压缩缓存后约 20MB，之后几秒出结果。

示例：
  python3 iclr_momentum.py --list
  python3 iclr_momentum.py --direction gnn
  python3 iclr_momentum.py --direction 图神经网络 --samples 8
  python3 iclr_momentum.py --name "时间序列预测" --phrases "time series forecasting" "temporal forecasting"
  python3 iclr_momentum.py --direction gnn --and-phrases "large language model" LLM   # 交叉：GNN × LLM
  python3 iclr_momentum.py --phrases financial stock portfolio --and-phrases "LLM agent" agentic
  python3 iclr_momentum.py --all                 # 扫描全部内置方向（找方向模式）
  python3 iclr_momentum.py --all --offline       # 不联网，用预计算快照
  python3 iclr_momentum.py --direction llm --json
"""

import argparse
import gzip
import json
import math
import os
import random
import re
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from statistics import NormalDist

SKILL_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SKILL_DIR / "data"
DIRECTIONS_FILE = DATA_DIR / "directions.json"
PRECOMPUTED_FILE = DATA_DIR / "iclr_precomputed.json"

YEARS = list(range(2017, 2027))
RAW_URL = "https://raw.githubusercontent.com/papercopilot/paperlists/main/iclr/iclr{year}.json"
MEDIA_URL = "https://media.githubusercontent.com/media/papercopilot/paperlists/main/iclr/iclr{year}.json"
USER_AGENT = "momentum-idea-skill/1.0 (+https://github.com/chaoRookie/Momentum-idea-skill)"
LFS_SIGNATURE = b"version https://git-lfs"
Z95 = NormalDist().inv_cdf(0.975)

LEVEL_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴", "black": "⚫", "na": "⚪"}
LEVEL_NAME = {"green": "较易", "yellow": "中等", "red": "困难", "black": "极难", "na": "数据不足"}
DIM_NAME = {
    "crowding": "①拥挤度",
    "momentum": "②势头",
    "premium": "③录用红利",
    "growth": "④增长去向",
}
LABELS = {
    "early_premium": ("🌱", "早期红利", "份额在涨，录用率也显著高于同年其他论文：越早进场越好"),
    "crowded_no_premium": ("🔥", "拥挤无红利", "份额在涨，但录用率和其他论文看不出差别：趋势还在，红利已被吃完"),
    "warming_unclear": ("🌤️", "升温中·红利不明", "份额在涨，但方向还小、样本不够，录用红利看不清"),
    "premium_to_discount": ("📉", "红利变折价", "前几年做这个方向录用率更高，现在反而更低：论文翻倍，红利变成折价"),
    "discount": ("📉", "录用折价", "做这个方向的录用率显著低于同年其他论文：审稿人对它更挑剔"),
    "growth_in_rejects": ("🧱", "增长落在拒稿", "论文数在涨，但新增的几乎都被拒了"),
    "lagging_market": ("🐢", "跑输大盘", "论文数还在涨，但涨得比大会慢、份额在缩（\"凉了\"最常见的一种）：你以为的冷门，其实人不少"),
    "shrinking": ("🥶", "真萎缩", "论文数真的在减少"),
    "steady": ("➖", "平稳", "没有明显的升温、退潮或红利信号"),
}
LABEL_PRIORITY = [
    "shrinking", "growth_in_rejects", "premium_to_discount", "discount", "early_premium",
    "crowded_no_premium", "warming_unclear", "lagging_market", "steady",
]


# ----------------------------------------------------------------------------
# 数据：下载、缓存、读取
# ----------------------------------------------------------------------------

def cache_dir():
    base = os.environ.get("MOMENTUM_IDEA_CACHE")
    root = Path(base).expanduser() if base else Path.home() / ".cache" / "momentum-idea"
    d = root / "iclr"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        d = Path(tempfile.gettempdir()) / "momentum-idea" / "iclr"
        d.mkdir(parents=True, exist_ok=True)
    return d


def classify_status(status):
    """把 papercopilot 的状态映射成 A（录用）/ R（拒稿）/ None（不计入）。"""
    s = (status or "").strip().lower()
    if not s or "withdraw" in s or "desk" in s:
        return None
    if "reject" in s or "workshop" in s:
        return "R"
    if any(k in s for k in ("accept", "oral", "poster", "spotlight", "talk", "top-", "notable")):
        return "A"
    return None


def _download(url, dest, timeout=600):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _meta_path():
    return cache_dir() / "meta.json"


def load_meta():
    try:
        return json.loads(_meta_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def build_year_cache(year):
    """下载某一年的原始列表，只保留已决论文的 id / 标题 / 摘要 / 结果，压缩存盘。"""
    path = cache_dir() / "iclr{}.jsonl.gz".format(year)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.json"
        _download(RAW_URL.format(year=year), raw)
        with open(raw, "rb") as f:
            head = f.read(64)
        if head.startswith(LFS_SIGNATURE):  # 大文件放在 Git LFS 里，要换 media 地址
            _download(MEDIA_URL.format(year=year), raw)
        with open(raw, "r", encoding="utf-8") as f:
            papers = json.load(f)
    status_counts = {}
    kept = 0
    tmp_path = path.with_name(path.name + ".tmp")
    with gzip.open(tmp_path, "wt", encoding="utf-8") as out:
        for p in papers:
            status = p.get("status") or ""
            status_counts[status] = status_counts.get(status, 0) + 1
            decision = classify_status(status)
            if decision is None:
                continue
            row = {"i": p.get("id") or "", "t": p.get("title") or "", "a": p.get("abstract") or "", "d": decision}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    os.replace(tmp_path, path)
    meta = load_meta()
    meta[str(year)] = {"fetched": date.today().isoformat(), "decided": kept, "status_counts": status_counts}
    _meta_path().write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def load_years(years, refresh=False):
    papers_by_year = {}
    for year in years:
        path = cache_dir() / "iclr{}.jsonl.gz".format(year)
        if refresh or not path.exists():
            print("⬇️  正在下载 ICLR {} 论文列表（只需一次）…".format(year), file=sys.stderr, flush=True)
            build_year_cache(year)
        rows = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                p["text"] = p["t"] + "\n" + p["a"]
                p["low"] = p["text"].lower()
                rows.append(p)
        papers_by_year[year] = rows
    return papers_by_year


# ----------------------------------------------------------------------------
# 方向：内置词表、关键词匹配
# ----------------------------------------------------------------------------

def load_directions():
    with open(DIRECTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["directions"]


def _norm(text):
    return text.lower().replace(" ", "")


def resolve_direction(query, directions):
    q = _norm(query)
    exact = [d for d in directions if _norm(d["id"]) == q or _norm(d["name"]) == q
             or any(_norm(a) == q for a in d.get("aliases", []))]
    if exact:
        return exact
    return [d for d in directions if q in _norm(d["name"]) or any(q in _norm(a) for a in d.get("aliases", []))]


_SPLIT = re.compile(r"[\s\-_/]+")


def _word_pattern(word, last):
    """含大写字母的单词（LLM、AI、NeRF）按大小写精确匹配，其余单词不区分大小写；最后一个词兼容复数。"""
    body = re.escape(word) + (r"(?:s|es)?" if last and word[-1].isalpha() else "")
    return body if any(ch.isupper() for ch in word) else "(?i:" + body + ")"


def phrase_to_regex(phrase):
    """词间空格/连字符等价，末尾兼容复数；大小写规则见 _word_pattern。
    返回 (预筛用的小写片段, 正则)：先用 in 做廉价的子串预筛，命中了再跑正则。"""
    words = [w for w in _SPLIT.split(phrase.strip()) if w]
    if not words:
        raise ValueError("关键词不能为空")
    body = r"[\s\-]+".join(_word_pattern(w, i == len(words) - 1) for i, w in enumerate(words))
    pattern = r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])"
    needle = max(words, key=len).lower()
    return needle, re.compile(pattern)


class Matcher:
    """include 里任一命中，且每个 and 组里各有一个命中，且 exclude 一个都不命中。"""

    def __init__(self, phrases, exclude=(), regexes=(), and_groups=()):
        self.include = [phrase_to_regex(p) for p in phrases] + [("", re.compile(r, re.IGNORECASE)) for r in regexes]
        self.and_groups = [[phrase_to_regex(p) for p in g] for g in and_groups if g]
        self.exclude = [phrase_to_regex(p) for p in exclude]
        if not self.include:
            raise ValueError("至少需要一个关键词")

    @staticmethod
    def _hit(rules, paper):
        low, text = paper["low"], paper["text"]
        return any(needle in low and rx.search(text) for needle, rx in rules)

    def match(self, paper):
        return (self._hit(self.include, paper)
                and all(self._hit(g, paper) for g in self.and_groups)
                and not self._hit(self.exclude, paper))


def matcher_for(direction):
    return Matcher(direction.get("phrases", []), direction.get("exclude", []), direction.get("regex", []),
                   direction.get("and", []))


# ----------------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------------

def totals_by_year(papers_by_year):
    return {y: (len(ps), sum(1 for p in ps if p["d"] == "A")) for y, ps in papers_by_year.items()}


def count_by_year(matcher, papers_by_year, collect=False):
    counts, matched = {}, {}
    for y, ps in papers_by_year.items():
        n = a = 0
        hits = []
        for p in ps:
            if matcher.match(p):
                n += 1
                if p["d"] == "A":
                    a += 1
                if collect:
                    hits.append(p)
        counts[y] = (n, a)
        matched[y] = hits
    return counts, matched


def compute_rows(counts, totals, z=Z95):
    """逐年算份额、份额年变化、相对录用率差。区间用 Agresti–Caffo（小样本更稳，大样本与普通区间几乎一样）。"""
    rows, prev_share = [], None
    for y in sorted(totals):
        N, A = totals[y]
        n, a = counts.get(y, (0, 0))
        share = n / N if N else 0.0
        row = {
            "year": y, "n": n, "accepted": a, "rejected": n - a, "N": N, "A": A, "share": share,
            "share_delta": None if prev_share is None else share - prev_share,
            "acc_rate": None, "other_acc_rate": None, "gap": None, "ci_low": None, "ci_high": None, "se": None,
        }
        prev_share = share
        m = N - n
        if n > 0 and m > 0:
            p1, p2 = a / n, (A - a) / m
            q1, q2 = (a + 1) / (n + 2), (A - a + 1) / (m + 2)
            se = math.sqrt(q1 * (1 - q1) / (n + 2) + q2 * (1 - q2) / (m + 2))
            center = q1 - q2
            row.update(acc_rate=p1, other_acc_rate=p2, gap=p1 - p2, se=se,
                       ci_low=center - z * se, ci_high=center + z * se)
        rows.append(row)
    return rows


def diagnose(rows, window=3, hot_cut=None, hot_cut_rel=None):
    by_year = {r["year"]: r for r in rows}
    latest = rows[-1]
    base = by_year.get(latest["year"] - window, rows[0])
    n1, n0 = latest["n"], base["n"]
    a1, a0 = latest["accepted"], base["accepted"]
    d = {
        "latest_year": latest["year"], "base_year": base["year"],
        "count_ratio": (n1 / n0) if n0 else None,
        "conf_ratio": latest["N"] / base["N"] if base["N"] else None,
        "share_latest": latest["share"], "share_base": base["share"],
        "delta_n": n1 - n0, "delta_accepted": a1 - a0,
        "growth_to_accept": ((a1 - a0) / (n1 - n0)) if n1 - n0 > 0 else None,
        "overall_acc_latest": latest["A"] / latest["N"] if latest["N"] else None,
        "small_sample": n1 < 30,
        "hot_cut": hot_cut,
        "hot_cut_rel": hot_cut_rel,
    }
    sd = latest["share_delta"] or 0.0
    gap, lo, hi = latest["gap"], latest["ci_low"], latest["ci_high"]
    prev = by_year.get(latest["year"] - 1)
    prev_share = prev["share"] if prev else 0.0
    # "升温"：份额年变化为正，且至少 +0.1 个百分点或份额相对涨 10%（排除 +0.01 这种噪声）
    warming = sd > 0 and (sd >= 0.001 or (prev_share > 0 and sd / prev_share >= 0.10))
    window_rows = [by_year[y] for y in range(base["year"], latest["year"]) if y in by_year]
    past_premium = any(r["gap"] is not None and r["n"] >= 10 and r["gap"] >= 0.03 for r in window_rows)

    labels = []
    if n0 > 0 and n1 < n0:
        labels.append("shrinking")
    if d["delta_n"] >= 20 and d["growth_to_accept"] is not None and d["growth_to_accept"] < 0.10:
        labels.append("growth_in_rejects")
    if gap is not None and n1 >= 10:
        if past_premium and (hi < 0 or gap <= -0.03):
            labels.append("premium_to_discount")
        elif hi < 0:
            labels.append("discount")
    if gap is not None and warming and lo > 0:
        labels.append("early_premium")
    if gap is not None and warming and lo <= 0 and not {"premium_to_discount", "discount"} & set(labels):
        labels.append("crowded_no_premium" if latest["share"] >= 0.03 else "warming_unclear")
    lag_window = n0 > 0 and n1 > n0 and latest["share"] < base["share"]
    lag_recent = prev is not None and n1 > prev["n"] and sd <= -0.005
    if lag_window or lag_recent:
        labels.append("lagging_market")
    if not labels:
        labels.append("steady")
    d["warming"] = warming
    prev_s = prev["share"] if prev else 0.0
    d["share_delta_rel"] = (sd / prev_s) if prev_s > 0 else None
    d["hot_abs"] = bool(hot_cut is not None and sd > 0 and sd >= hot_cut)
    d["hot_rel"] = bool(hot_cut_rel is not None and d["share_delta_rel"] is not None and sd > 0
                        and d["share_delta_rel"] >= hot_cut_rel and n1 >= 30)
    labels.sort(key=LABEL_PRIORITY.index)
    d["labels"] = labels
    d["primary_label"] = labels[0]
    d["levels"] = levels(latest, base, d, hot_cut)
    return d


def levels(latest, base, d, hot_cut):
    lv = {}
    share = latest["share"]
    lv["crowding"] = "red" if share > 0.10 else ("yellow" if share >= 0.03 else "green")

    sd = latest["share_delta"]
    if "shrinking" in d["labels"]:
        lv["momentum"] = "black"
    elif sd is None:
        lv["momentum"] = "na"
    elif sd <= -0.005:
        lv["momentum"] = "red"
    elif d.get("hot_abs") or d.get("hot_rel"):
        lv["momentum"] = "green"
    else:
        lv["momentum"] = "yellow"

    gap, lo, hi = latest["gap"], latest["ci_low"], latest["ci_high"]
    if gap is None or latest["n"] < 10:
        lv["premium"] = "na"
    elif lo > 0:
        lv["premium"] = "green"
    elif hi < 0:
        lv["premium"] = "black" if gap <= -0.10 else "red"
    else:
        lv["premium"] = "yellow"

    if d["delta_n"] > 0:
        r = d["growth_to_accept"]
        if r < 0:
            lv["growth"] = "black"
        elif r < 0.10:
            lv["growth"] = "red"
        elif d["overall_acc_latest"] is not None and r < d["overall_acc_latest"]:
            lv["growth"] = "yellow"
        else:
            lv["growth"] = "green"
    elif latest["n"] == 0:
        lv["growth"] = "na"
    elif latest["accepted"] >= base["accepted"]:
        lv["growth"] = "green"
    elif (latest["acc_rate"] or 0) >= (base["acc_rate"] or 0):
        lv["growth"] = "yellow"
    else:
        lv["growth"] = "red"
    return lv


def phrase_hits(spec, matcher, papers):
    """每个关键词在给定论文里命中几篇、其中几篇只靠它命中（主关键词组）。"""
    labels = list(spec.get("phrases", [])) + ["regex:" + r for r in spec.get("regex", [])]
    rules = matcher.include
    out = []
    for i, (label, (needle, rx)) in enumerate(zip(labels, rules)):
        hit = only = 0
        for p in papers:
            if needle in p["low"] and rx.search(p["text"]):
                hit += 1
                if not any(n2 in p["low"] and r2.search(p["text"]) for j, (n2, r2) in enumerate(rules) if j != i):
                    only += 1
        out.append((label, hit, only))
    return out


def hot_cutoff(share_deltas):
    """文章回测的 hot 定义：当年份额增长最快的四分之一方向。返回这一门槛（份额年变化）。"""
    vals = sorted(v for v in share_deltas if v is not None)
    if not vals:
        return None
    k = len(vals) - int(math.ceil(len(vals) / 4.0))
    return vals[max(0, k)]


def bonferroni_z(m, alpha=0.05):
    return NormalDist().inv_cdf(1 - alpha / (2 * max(1, m)))


# ----------------------------------------------------------------------------
# 预计算快照（离线兜底）
# ----------------------------------------------------------------------------

def load_precomputed():
    with open(PRECOMPUTED_FILE, "r", encoding="utf-8") as f:
        pre = json.load(f)
    totals = {int(y): tuple(v) for y, v in pre["totals"].items()}
    counts = {k: {int(y): tuple(v) for y, v in c.items()} for k, c in pre["directions"].items()}
    return pre, totals, counts


def build_precomputed(papers_by_year, directions):
    totals = totals_by_year(papers_by_year)
    out = {
        "generated": date.today().isoformat(),
        "source": "papercopilot/paperlists (ICLR 2017-2026)，按 data/directions.json 计算",
        "definition": "每年 [论文数, 录用数]；已决论文 = 录用 + 拒稿（不含撤稿 / desk reject）",
        "totals": {str(y): list(v) for y, v in totals.items()},
        "directions": {},
    }
    for d in directions:
        counts, _ = count_by_year(matcher_for(d), papers_by_year)
        out["directions"][d["id"]] = {str(y): list(v) for y, v in counts.items()}
    PRECOMPUTED_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return PRECOMPUTED_FILE


# ----------------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------------

def pct(x, digits=1):
    return "—" if x is None else "{:.{}f}%".format(100 * x, digits)


def pp(x, digits=1):
    return "—" if x is None else "{:+.{}f}".format(100 * x, digits)


def ratio(x):
    return "—" if x is None else "×{:.2f}".format(x)


def gap_cell(r):
    if r["gap"] is None:
        return "—"
    s = "{}（{}, {}）".format(pp(r["gap"]), pp(r["ci_low"]), pp(r["ci_high"]))
    return s + (" ⚠️样本少" if r["n"] < 30 else "")


def label_text(key):
    emoji, name, _ = LABELS[key]
    return "{} {}".format(emoji, name)


def render_direction(title, spec, rows, diag, samples, source_note, show_years=None):
    out = []
    out.append("## 📊 ICLR momentum：{}".format(title))
    out.append("")
    out.append("- 口径：{}".format(spec))
    out.append("- 已决论文 = 录用 + 拒稿（不含撤稿、desk reject，所以这里的录用率会比官方公布的高）；一篇论文可同时属于多个方向")
    out.append("- 数据：{}".format(source_note))
    out.append("")
    out.append("| 年份 | 论文数 | 录用 | 拒稿 | 份额 | 份额年变化 | 录用率 | 其他论文录用率 | 相对录用率差（95% 区间） |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    first = next((i for i, r in enumerate(rows) if r["n"] > 0), len(rows))
    for r in rows[max(0, min(first, len(rows) - 4)):]:
        if show_years and r["year"] not in show_years:
            continue
        out.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["year"], r["n"], r["accepted"], r["rejected"], pct(r["share"], 2), pp(r["share_delta"], 2),
            pct(r["acc_rate"]), pct(r["other_acc_rate"]), gap_cell(r)))
    out.append("")
    latest = rows[-1]
    y0, y1 = diag["base_year"], diag["latest_year"]
    out.append("### 诊断（{} → {}）".format(y0, y1))
    out.append("- 论文数 {}，大会整体 {}；份额 {} → {}".format(
        ratio(diag["count_ratio"]), ratio(diag["conf_ratio"]), pct(diag["share_base"], 2), pct(diag["share_latest"], 2)))
    if diag["delta_n"] > 0 and diag["delta_accepted"] > diag["delta_n"]:
        out.append("- 论文只多了 {} 篇，录用却多了 {} 篇（拒稿减少了 {} 篇）：增长全落在录用里".format(
            diag["delta_n"], diag["delta_accepted"], diag["delta_accepted"] - diag["delta_n"]))
    elif diag["delta_n"] > 0:
        out.append("- 新增 {} 篇，其中录用 {:+d} 篇（占新增的 {}）；{} 年大会整体录用率 {}（已决口径）".format(
            diag["delta_n"], diag["delta_accepted"], pct(diag["growth_to_accept"]), y1, pct(diag["overall_acc_latest"])))
    else:
        out.append("- 论文数变化 {:+d} 篇，录用数变化 {:+d} 篇".format(diag["delta_n"], diag["delta_accepted"]))
    if latest["gap"] is not None:
        if latest["ci_low"] > 0:
            verdict = "显著为正：选这个方向目前有录用红利"
        elif latest["ci_high"] < 0:
            verdict = "显著为负：做这个方向的录用率反而更低"
        else:
            verdict = "区间跨零：和其他论文的录用率看不出差别"
        out.append("- {} 年相对录用率差 {}（{}, {}）→ {}".format(
            y1, pp(latest["gap"]), pp(latest["ci_low"]), pp(latest["ci_high"]), verdict))
    if diag.get("hot_cut") is not None and latest["share_delta"] is not None:
        if diag["hot_abs"]:
            verdict = "属于 hot 组（文章口径）"
        elif diag["hot_rel"]:
            verdict = "按相对增幅属于 hot 组（小方向适用）"
        else:
            verdict = "不属于 hot 组"
        rel_txt = ""
        if diag.get("hot_cut_rel") is not None and diag.get("share_delta_rel") is not None:
            rel_txt = "；相对增幅 {:+.0f}%（前 25% 门槛 ≥ {:+.0f}%，要求 ≥ 30 篇）".format(
                100 * diag["share_delta_rel"], 100 * diag["hot_cut_rel"])
        out.append("- 份额年变化 {}（28 个内置方向里前 25% 的门槛 ≥ {}）{} → {}".format(
            pp(latest["share_delta"], 2), pp(diag["hot_cut"], 2), rel_txt, verdict))
    out.append("- 处境标签：**{}**{}".format(
        label_text(diag["primary_label"]),
        "（另：{}）".format("、".join(label_text(k) for k in diag["labels"][1:])) if len(diag["labels"]) > 1 else ""))
    out.append("  - {}".format(LABELS[diag["primary_label"]][2]))
    lv = diag["levels"]
    out.append("- 数据等级：" + " ｜ ".join("{} {}{}".format(DIM_NAME[k], LEVEL_EMOJI[lv[k]], LEVEL_NAME[lv[k]]) for k in DIM_NAME))
    n_latest = rows[-1]["n"]
    conf = ("样本少（最新一年 {} 篇 < 30），结论只能当方向性参考".format(n_latest) if diag["small_sample"]
            else "样本量够（最新一年 {} 篇）".format(n_latest))
    out.append("- 可信度：{}；关键词准不准要看下面的命中样例；以上都是历史关联，不是因果".format(conf))
    if samples:
        out.append("")
        out.append("### 命中样例（{} 年随机抽取，用来检查关键词有没有误伤）".format(y1))
        for p in samples:
            tag = "录用" if p["d"] == "A" else "拒稿"
            link = " — https://openreview.net/forum?id={}".format(p["i"]) if p.get("i") else ""
            out.append("- [{}] {}{}".format(tag, p["t"].strip().replace("\n", " "), link))
    return "\n".join(out)


def render_scan(results, window, m, source_note):
    y1 = results[0]["rows"][-1]["year"]
    y0 = y1 - window
    zb = bonferroni_z(m)
    out = []
    out.append("## 🔍 ICLR {} 内置方向扫描（{} 个方向）".format(y1, m))
    out.append("")
    out.append("- 数据：{}".format(source_note))
    out.append("- hot = 当年份额增长最快的前 25%（文章回测口径）")
    out.append("- ✱ = 当年相对录用率差在 {} 个方向的多重比较校正后仍显著不为零（检验的是当年的差距，和文章检验\"下降趋势是否显著\"不是同一个检验）".format(m))
    out.append("- 录用率按已决论文算（不含撤稿、desk reject），会比官方公布的高")
    out.append("")
    out.append("| 方向 | {} 论文数 | 份额 | 份额年变化 | 相对录用率差（95% 区间） | {}→{} 新增里录用占比 | 处境标签 |".format(y1, y0, y1))
    out.append("|---|---|---|---|---|---|---|")
    for res in sorted(results, key=lambda x: -x["rows"][-1]["n"]):
        r, d = res["rows"][-1], res["diag"]
        hot = " ⬆️hot" if (r["share_delta"] is not None and d["hot_cut"] is not None
                           and r["share_delta"] > 0 and r["share_delta"] >= d["hot_cut"]) else ""
        star = ""
        if r["gap"] is not None and r["se"]:
            if abs(r["gap"]) > zb * r["se"]:
                star = " ✱"
        growth = pct(d["growth_to_accept"]) if d["delta_n"] > 0 else "论文数{:+d}".format(d["delta_n"])
        out.append("| {} | {} | {} | {}{} | {}{} | {} | {} |".format(
            res["name"], r["n"], pct(r["share"], 2), pp(r["share_delta"], 2), hot, gap_cell(r), star,
            growth, " ".join(LABELS[k][0] for k in d["labels"]) + " " + LABELS[d["primary_label"]][1]))
    out.append("")
    out.append("标签图例：" + "；".join("{} {}".format(v[0], v[1]) for k, v in LABELS.items()))
    out.append("")
    out.append("提醒：格子里是份额和录用率的历史关联，不是科学价值；热门方向给的是\"少亏一点\"，不是保送。")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="ICLR 科研 momentum：份额 / 份额年变化 / 相对录用率差")
    ap.add_argument("--list", action="store_true", help="列出内置方向")
    ap.add_argument("-d", "--direction", help="内置方向 id、中文名或别名，如 gnn / 图神经网络")
    ap.add_argument("--name", help="自定义方向的显示名")
    ap.add_argument("--phrases", nargs="+", help="自定义方向的关键词（英文，匹配标题+摘要）")
    ap.add_argument("--and-phrases", nargs="+", action="append", default=[], metavar="PHRASE",
                    help="再加一组关键词，必须同时命中（组内任一即可；可重复使用来叠加多组），用于交叉方向")
    ap.add_argument("--exclude", nargs="+", default=[], help="命中这些词就排除")
    ap.add_argument("--regex", nargs="+", default=[], help="高级：额外的正则表达式（不区分大小写）")
    ap.add_argument("--all", action="store_true", help="扫描全部内置方向")
    ap.add_argument("--window", type=int, default=3, help="增长去向的回看年数，默认 3（即 2023→2026）")
    ap.add_argument("--samples", type=int, default=5, help="展示多少篇命中论文样例，默认 5")
    ap.add_argument("--phrase-hits", action="store_true", help="列出每个关键词在最新一年各命中多少篇，用来删掉误伤多的词")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offline", action="store_true", help="不联网，用 data/iclr_precomputed.json（仅内置方向）")
    ap.add_argument("--refresh", action="store_true", help="重新下载数据")
    ap.add_argument("--prefetch", action="store_true", help="只下载并缓存数据")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--build-precomputed", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def fail(msg, code=2):
    print("❌ " + msg, file=sys.stderr)
    sys.exit(code)


def network_hint(err):
    return ("下载 ICLR 数据失败：{}\n"
            "   · 内置方向可以加 --offline，用预计算快照（数据截至快照日期）；\n"
            "   · 自定义方向离线算不了：改用网页搜索 + references/iclr-snapshot.md 里文章的数据做判断，并在报告里注明。"
            ).format(err)


def main(argv=None):
    args = parse_args(argv)
    directions = load_directions()

    if args.list:
        for d in directions:
            print("{:<24} {:<18} 关键词：{}".format(d["id"], d["name"], " | ".join(d["phrases"])))
        return 0

    if args.prefetch:
        try:
            load_years(YEARS, refresh=args.refresh)
        except (urllib.error.URLError, OSError) as e:
            fail(network_hint(e))
        print("✅ ICLR 2017–2026 已缓存到 {}".format(cache_dir()))
        return 0

    custom = bool(args.phrases or args.regex)
    if args.and_phrases and not (custom or args.direction):
        fail("--and-phrases 要和 --phrases 或 --direction 一起用（它是\"同时还要命中\"的那一组）")
    if not (args.all or args.direction or custom or args.build_precomputed):
        fail("请指定 --direction、--phrases 或 --all（--list 查看内置方向）")

    # 选定要算的方向
    targets = []
    if args.all:
        targets = [(d["id"], d["name"], d) for d in directions]
    elif args.direction and not custom:
        found = resolve_direction(args.direction, directions)
        if not found:
            fail("没找到内置方向「{}」。用 --list 查看，或者用 --phrases 自定义关键词。".format(args.direction))
        if len(found) > 1:
            fail("「{}」匹配到多个方向：{}，请写得更具体".format(args.direction, "、".join(d["id"] for d in found)))
        d = found[0]
        if args.and_phrases or args.exclude:
            spec = dict(d)
            spec["and"] = args.and_phrases
            spec["exclude"] = d.get("exclude", []) + args.exclude
            name = args.name or " × ".join([d["name"]] + [" / ".join(g) for g in args.and_phrases])
            targets = [(d["id"] + "+", name, spec)]
        else:
            targets = [(d["id"], d["name"], d)]
    elif custom:
        name = args.name or args.direction or " × ".join(
            [" / ".join(args.phrases or args.regex)] + [" / ".join(g) for g in args.and_phrases])
        spec = {"id": "custom", "name": name, "phrases": args.phrases or [], "exclude": args.exclude,
                "regex": args.regex, "and": args.and_phrases}
        targets = [("custom", name, spec)]

    # 取数
    samples_by_id, counts_by_id, phrase_hits_by_id = {}, {}, {}
    if args.offline:
        if custom or args.and_phrases or args.exclude:
            fail("自定义关键词、交叉方向需要原始数据，离线算不了。请联网运行，或改用网页搜索 + 文章快照。")
        pre, totals, pre_counts = load_precomputed()
        for key, _, _ in targets:
            counts_by_id[key] = pre_counts[key]
        source_note = "预计算快照（{} 生成，{}），离线模式无命中样例".format(pre["generated"], pre["source"])
        hot_deltas_source = pre_counts
    else:
        try:
            papers = load_years(YEARS, refresh=args.refresh)
        except (urllib.error.URLError, OSError) as e:
            fail(network_hint(e))
        if args.build_precomputed:
            path = build_precomputed(papers, directions)
            print("✅ 已写入 {}".format(path))
            return 0
        totals = totals_by_year(papers)
        meta = load_meta()
        fetched = sorted({v.get("fetched", "?") for v in meta.values()})
        source_note = "papercopilot/paperlists 的 ICLR 2017–2026 列表（抓取于 {}）".format("、".join(fetched) or "未知")
        rng = random.Random(args.seed)
        for key, _, spec in targets:
            matcher = matcher_for(spec)
            counts, matched = count_by_year(matcher, papers, collect=args.samples > 0 or args.phrase_hits)
            counts_by_id[key] = counts
            latest = matched.get(max(YEARS), [])
            samples_by_id[key] = rng.sample(latest, min(args.samples, len(latest))) if args.samples > 0 else []
            if args.phrase_hits and not args.all:
                phrase_hits_by_id[key] = phrase_hits(spec, matcher, latest)
        hot_deltas_source = None

    # hot 门槛：28 个内置方向最新一年份额年变化的前 25%
    if hot_deltas_source is None:
        if args.all:
            hot_deltas_source = counts_by_id
        else:
            try:
                _, _, hot_deltas_source = load_precomputed()
            except (OSError, ValueError, KeyError):
                hot_deltas_source = {}
    deltas, rel_deltas = [], []
    for c in hot_deltas_source.values():
        r = compute_rows(c, totals)
        deltas.append(r[-1]["share_delta"])
        prev_share = r[-2]["share"] if len(r) > 1 else 0.0
        rel_deltas.append(r[-1]["share_delta"] / prev_share if prev_share > 0 else None)
    cut = hot_cutoff(deltas)
    cut_rel = hot_cutoff(rel_deltas)

    results = []
    for key, name, spec in targets:
        rows = compute_rows(counts_by_id[key], totals)
        diag = diagnose(rows, window=args.window, hot_cut=cut, hot_cut_rel=cut_rel)
        results.append({"id": key, "name": name, "spec": spec, "rows": rows, "diag": diag,
                        "samples": samples_by_id.get(key, [])})

    if args.json:
        payload = {"source": source_note, "window": args.window, "hot_cut": cut, "hot_cut_rel": cut_rel, "results": [
            {"id": r["id"], "name": r["name"], "phrases": r["spec"].get("phrases", []),
             "and": r["spec"].get("and", []), "exclude": r["spec"].get("exclude", []), "rows": r["rows"],
             "diagnosis": r["diag"],
             "samples": [{"title": p["t"], "decision": p["d"], "openreview_id": p["i"]} for p in r["samples"]]}
            for r in results]}
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 0

    if args.all:
        print(render_scan(results, args.window, len(results), source_note))
        return 0

    r = results[0]
    spec = r["spec"]
    words = " / ".join("`{}`".format(p) for p in spec.get("phrases", []) + spec.get("regex", []))
    also = "".join("，且同时命中 " + " / ".join("`{}`".format(p) for p in g) for g in spec.get("and", []))
    excl = "；排除 " + " / ".join("`{}`".format(p) for p in spec.get("exclude", [])) if spec.get("exclude") else ""
    spec_text = "标题 + 摘要匹配 {}{}{}（含大写字母的单词按大小写精确匹配，其余单词不区分大小写）".format(words, also, excl)
    print(render_direction("{}（{}）".format(r["name"], r["id"]), spec_text, r["rows"], r["diag"], r["samples"], source_note))
    hits = phrase_hits_by_id.get(r["id"])
    if hits:
        print()
        print("### 各关键词命中（{} 年，在上面命中的论文里）".format(r["rows"][-1]["year"]))
        print("| 关键词 | 命中 | 只靠它命中 |")
        print("|---|---|---|")
        for label, hit, only in hits:
            print("| `{}` | {} | {} |".format(label, hit, only))
        print()
        print("只靠某个词命中、样例又不对题的，就是该删的词。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
