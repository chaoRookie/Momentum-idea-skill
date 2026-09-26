---
name: momentum-idea
description: 用「科研 momentum」框架帮用户寻找合适的 AI/ML 科研方向，并预评估某个方向现在的投稿难度。基于 4 万篇 ICLR 2017–2026 录用/拒稿数据回测的三个指标（份额、份额年变化、相对录用率差），再结合个人条件（阶段、持有期、算力、经验、导师），给出两层难度、推荐投稿档位和切入建议。只要用户在找科研方向或 idea、纠结选题、问某个方向还能不能做、卷不卷、凉了没、好不好中、投稿难不难、该投哪个会，想比较几个方向，或提到 momentum、方向冷热、录用红利、追热点，都应该使用本 skill，即使用户没有明说要评估。Also use it whenever someone wants to find an AI research direction, check whether a topic is crowded or still has an acceptance premium, estimate paper submission difficulty, or pick a target venue.
---

# momentum-idea：找科研方向 + 预评估投稿难度

核心问题来自《做科研越来越像炒股：4 万篇 ICLR 论文回测》（鸭基米德，PaperWeekly）：

> 选方向时，比"热不热"更该问的是：这个方向的增长，落在录用里，还是落在拒稿里？

本 skill 用数据回答"这个方向热不热、还有没有录用红利"，再结合用户的个人条件回答"以你的资源能不能做出来、该投哪"。它不追问用户：个人信息只在开头问一轮，之后直接给结论。

下文的 `<SKILL_DIR>` 指本 skill 所在的目录（SKILL.md 所在的文件夹）。

## 第 0 步：拿到个人档案

"对你来说的难度"要用到 5 项个人信息。

1. 先找现成的：Claude Code 里读 `~/.momentum-idea/profile.md`；claude.ai 里看对话、项目说明或记忆里有没有这些信息。
2. 没有就**一次性**问下面几个问题，一条消息问完，不追问；用户说"跳过"就用默认值：
   1. 阶段：本科 / 硕士 / 博士 / 工作，几年级？
   2. 持有期：离你最近的硬目标（投稿截止、毕业、复试、保研材料）还有多久？
   3. 算力：比如"一张 4090""实验室 4×A100""只有 API 额度"
   4. 相关经验：会什么、复现过什么、发过什么
   5. 导师或合作者：有没有人在这个方向带你？
   6. （仅找方向模式）感兴趣的领域或关键词
3. 用户跳过的项用默认值：本科生；持有期 6 个月；一张 24GB 消费级显卡；会 PyTorch、没有一作论文；没人带。报告里凡是用到默认值的地方都标"（假设）"。
4. Claude Code 里把答案写进 `~/.momentum-idea/profile.md`，下次直接用；用户说情况变了就更新：

```markdown
# momentum-idea 个人档案（更新于 YYYY-MM-DD）
- 阶段：
- 持有期：
- 算力：
- 相关经验：
- 导师或合作者：
- 兴趣方向：
```

如果用户只是想快速看数据（比如"LLM 现在还有红利吗"），可以先给方向本身的结论，再补一句"告诉我你的情况，可以算对你来说的难度"。

## 第 1 步：判断模式

- **评估模式**：用户给了具体方向或 idea（"我想做 X""X 还能做吗""帮我评估这个 idea"）。
- **找方向模式**：没有具体方向（"我该做什么方向""推荐几个方向"），或者给了几个候选要比较。
- 分不清时，先按评估模式处理用户提到的方向，结尾再提供找方向模式。

## 第 2 步（评估模式）

1. **定关键词**
   - 先看能不能对上内置方向：`python3 <SKILL_DIR>/scripts/iclr_momentum.py --list`
   - 对不上就自己拟 3–6 个英文关键词：全称、常见缩写（写成大写，会按大小写精确匹配）、1–2 个同义说法。避开 "learning""model" 这种泛词，否则会匹配到大半个会议。
   - 具体 idea 往往太细、样本不够：同时准备"上一级方向"（给市场背景）和"细分方向"（看竞争密度）两组关键词，分别跑。
2. **跑 ICLR 数据**（首次运行会下载约 300MB 并缓存，之后几秒出结果）
   ```bash
   python3 <SKILL_DIR>/scripts/iclr_momentum.py --direction gnn
   python3 <SKILL_DIR>/scripts/iclr_momentum.py --name "时间序列预测" --phrases "time series forecasting" "time-series forecasting"
   ```
   检查输出末尾的命中样例：如果明显有误伤（比如 "agent" 匹配到强化学习里的 agent），用 `--exclude` 或更具体的词重跑一次。需要结构化结果时加 `--json`。
3. **跑 arXiv 近期势头**：ICLR 一年才出一次结果，看不到最近半年。方向很新，或 ICLR 最新一年样本少时尤其要跑（约 1.5 分钟）：
   ```bash
   python3 <SKILL_DIR>/scripts/arxiv_momentum.py --direction gnn --months 12
   ```
4. **⑤ 审稿门槛**：按 `references/difficulty-rubric.md` 第 3 节做 3–5 次针对性搜索，记下依据和链接。
5. **⑥–⑨ 对你来说**：对照个人档案，按 `references/difficulty-rubric.md` 第 4 节定等级。
6. **总评、投稿档位、切入建议**：按 `references/difficulty-rubric.md` 第 5–7 节；联网核对最新的 CCF 目录和持有期内的截稿日期。
7. **出报告**：用 `references/report-template.md` 的模板 A。

## 第 2 步（找方向模式）

1. 档案里没有兴趣方向就问（和第 0 步合并成一轮）。
2. 扫描全部内置方向：`python3 <SKILL_DIR>/scripts/iclr_momentum.py --all`
3. 把用户兴趣和热门方向做交叉（比如"金融 × Agent""金融 × 时间序列"），用 `--phrases` 跑 2–4 个交叉方向。交叉点往往就是"热门方向里的小切口"。
4. 按持有期和处境标签筛选（规则见 `references/difficulty-rubric.md` 第 4 节 ⑧）：持有期短的优先 🌱 和 🔥 里的小切口，避开 🧱 🥶；🐢 📉 只在用户有明确优势时保留。再按兴趣和资源排序。
5. 给出 5–8 个候选的对比表，前 3 名各用三句话说清楚，按模板 B 出报告；最后邀请用户挑一个做完整评估。

## 数据跑不了怎么办

- **ICLR 数据下载失败**：内置方向加 `--offline`，改用 `data/iclr_precomputed.json`（2026-09-26 计算）；自定义方向离线算不了，就用 `references/iclr-snapshot.md` 里最接近的方向作参照，加上网页搜索，并写明是近似。
- **arXiv 失败**（被限流时常见 406 之类的错误）：过几分钟重试；或者用网页搜索估计近期热度，在报告里写"近期势头未用 arXiv 核实"。
- **claude.ai 里不能跑脚本或不能联网**：用 `references/iclr-snapshot.md` + 网页搜索，可信度记为"低"。
- 不管用哪种兜底，都在报告里写清楚数据来源和截止日期。

## 输出

先在对话里给简短结论，再给完整报告：

```markdown
**一句话结论**：这个方向目前是 🔥 拥挤无红利：趋势还在，录用红利已经被吃完了……
| | 方向本身 | 对你来说 |
|---|---|---|
| 难度 | 🔴 困难 | 🟡 中等 |
**推荐投稿**：……（B 计划：……）
**切入建议**：1. …… 2. ……
📄 完整报告：momentum-report-xxx-20260926.md
```

- Claude Code：报告写成当前目录下的 Markdown 文件。
- claude.ai：生成可下载的 Markdown 文件；做不到就直接在对话里给出完整报告。
- 比较多个方向时附对比表。

## 判断原则

- **数字要有出处**：只用脚本输出、文章快照或带链接的网页数据，拿不到就写"未核实"。一个编出来但看着合理的数字，会直接误导用户选题。
- **不给总分**：文章特别强调"区间跨零就是看不出差别"。用等级 + 理由 + 可信度，不用一个看似精确的分数。
- **说清口径和边界**：都是 ICLR 上的历史关联，不是因果；热度不代表科学价值；ICLR 以外的会议只能参考。
- **追热点是"少亏一点"，不是保送**：推荐热门方向时也要说清楚这一点。
- **语气**：用户多半正在为选方向焦虑。坦诚说出难点，同时给出能走的路（更小的切口、更合适的档位、第一周能做的事），而不是只说"别做"。用用户的语言回答，可以适当用 emoji。

## 参考文件

- `references/momentum-framework.md`：文章框架（momentum / crash / value / 持有期、三个指标、处境标签）。第一次用本 skill，或需要向用户解释"为什么这么判断"时读。
- `references/difficulty-rubric.md`：9 个方面的等级标准、总评规则、投稿档位、切入策略。每次评估都要用。
- `references/report-template.md`：两套报告模板。出报告前读。
- `references/iclr-snapshot.md`：文章原始数据 + 本 skill 重算对照。兜底，或需要引用原文数字时读。
- `data/directions.json`：28 个内置方向的关键词表。
