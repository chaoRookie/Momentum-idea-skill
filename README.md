# momentum-idea：找科研方向 + 预评估投稿难度

一个 Claude Skill（Claude Code 和 claude.ai 都能用）。它用「科研 momentum」框架帮你：

1. 🔍 **找方向**：扫描 28 个 AI 方向的冷热和录用红利，结合你的兴趣、持有期和资源，给出候选对比表；
2. ⚖️ **评估方向**：对一个具体方向或 idea，给出"方向本身"和"对你来说"两层难度、推荐投稿档位和切入建议。

框架和核心问题来自 PaperWeekly 转载的《做科研越来越像炒股：4 万篇 ICLR 论文回测，追热点是对的》（作者：鸭基米德，新加坡国立大学）：

> 选方向时，比"热不热"更该问的是：这个方向的增长，落在录用里，还是落在拒稿里？

## 它怎么判断

**数据部分**（脚本自动计算，指标定义与文章一致）：

| 指标 | 定义 |
|---|---|
| 份额 | 该方向论文数 ÷ 当年 ICLR 已决论文总数 |
| 份额年变化 | 今年份额 − 去年份额，即科研圈的 momentum |
| 相对录用率差 | 该方向录用率 − 同年其他论文录用率（附 95% 区间），即有没有录用红利 |
| 增长去向 | 近 3 年新增的论文里，有多少落在录用里 |

再据此给每个方向打上处境标签：🌱 早期红利 / 🔥 拥挤无红利 / 🌤️ 升温中·红利不明 / 📉 红利变折价（及只算趋势的一档）/ 📉 录用折价 / 🧱 增长落在拒稿 / 🐢 跑输大盘 / 🥶 真萎缩。标签只在 95% 区间支持时下定论，区间跨零的只算趋势。

**经验部分**：审稿门槛（搜索近期录用论文和审稿意见），以及你的算力、技能、**持有期**（离必须出成果还有多久）和导师支持。

**输出**：两层难度等级（🟢🟡🔴⚫，不给总分）+ 推荐投稿档位（顶会主会 / 次一档 / workshop / 期刊）+ 切入建议 + 可信度，并生成一份 Markdown 报告。

## 安装

### Claude Code

在这个仓库里打开 Claude Code 就能直接用：`.claude/skills/momentum-idea` 是指向 `momentum-idea/` 的链接，会自动加载（云端会话也一样）。

想在所有项目里都能用，装到个人目录：

```bash
git clone https://github.com/chaoRookie/Momentum-idea-skill.git
mkdir -p ~/.claude/skills
cp -r Momentum-idea-skill/momentum-idea ~/.claude/skills/
```

只想在某个项目里用，就复制到该项目的 `.claude/skills/` 下。

### claude.ai

1. 打包：`cd Momentum-idea-skill && zip -r momentum-idea.zip momentum-idea`
2. 在 claude.ai 设置里的 Skills 页面上传这个 zip（需要开启代码执行功能）。

claude.ai 的沙箱可能下载不了数据，这时 skill 会自动改用内置的预计算快照和文章数据，并在报告里注明。

## 使用示例

直接用自然语言问就行：

- "帮我评估一下：金融时间序列预测 + 大模型，现在投稿难不难？"
- "GNN 现在还值得做吗？"
- "我是大三学生，8 个月后要准备复试，只有一张 4090，帮我找几个合适的方向"
- "Agent 和 RAG 我该选哪个？"

第一次使用时，它会一次性问你 5 个问题（阶段、持有期、算力、经验、导师），在 Claude Code 里会存到 `~/.momentum-idea/profile.md`，之后不用重复回答。

也可以直接运行脚本：

```bash
python3 momentum-idea/scripts/iclr_momentum.py --list                 # 28 个内置方向
python3 momentum-idea/scripts/iclr_momentum.py --direction gnn        # 单个方向
python3 momentum-idea/scripts/iclr_momentum.py --all                  # 全部方向扫描
python3 momentum-idea/scripts/iclr_momentum.py --name "时间序列预测" --phrases "time series forecasting"
python3 momentum-idea/scripts/iclr_momentum.py --direction gnn --and-phrases "large language model" LLM   # 交叉方向：GNN × LLM
python3 momentum-idea/scripts/iclr_momentum.py --all --offline        # 不联网
python3 momentum-idea/scripts/arxiv_momentum.py --direction gnn       # arXiv 近 12 个月势头
```

只依赖 Python 3.8+ 标准库。首次运行会下载约 300MB 的 ICLR 论文列表，压缩缓存到 `~/.cache/momentum-idea/`（约 20MB），之后一次全方向扫描约 15 秒。

## 数据与致谢

- 框架、指标和文章数据：鸭基米德《做科研越来越像炒股：4 万篇 ICLR 论文回测，追热点是对的》（PaperWeekly 转载）
- ICLR 2017–2026 录用/拒稿记录：[papercopilot/paperlists](https://github.com/papercopilot/paperlists)
- 近期势头：[arXiv API](https://info.arxiv.org/help/api/index.html)（请遵守其每 3 秒一次的请求间隔）

原文没有公开关键词表。本 skill 重建了 28 个方向的词表，并用文章公布的数字校准：LLM 各年论文数误差在 3% 以内，相对录用率差与文章相差不超过 0.2 个百分点；28 个方向份额年变化的平均偏差都 ≤ 0.2 个百分点。完整对照见 [`references/iclr-snapshot.md`](momentum-idea/references/iclr-snapshot.md)。

## 局限

- 只有 ICLR 公开拒稿，所以"录用红利"只能在 ICLR 上算；其他会议的结论只能参考。
- 关键词匹配有误伤和漏检，报告会列出命中样例供你检查。
- 所有结论都是历史关联，不是因果；热度也不代表科学价值。
- arXiv 可能对共享 IP 限流，失败时 skill 会改用网页搜索并注明。

## 目录结构

```
momentum-idea/
├── SKILL.md                     # 主流程
├── references/
│   ├── momentum-framework.md    # 文章框架与处境标签
│   ├── difficulty-rubric.md     # 9 个方面的等级规则、总评、投稿档位、切入策略
│   ├── report-template.md       # 报告模板
│   └── iclr-snapshot.md         # 文章数据快照 + 重算对照
├── data/
│   ├── directions.json          # 28 个方向的关键词表
│   └── iclr_precomputed.json    # 预计算结果（离线兜底）
└── scripts/
    ├── iclr_momentum.py         # ICLR 三指标、标签、等级
    └── arxiv_momentum.py        # arXiv 月度势头
```

更新数据：`python3 momentum-idea/scripts/iclr_momentum.py --prefetch --refresh` 重新下载；改了关键词表后，运行 `python3 momentum-idea/scripts/iclr_momentum.py --build-precomputed` 重新生成离线快照。
