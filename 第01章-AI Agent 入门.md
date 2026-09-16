# 第 1 章 AI Agent 入门

> 来源：《深入理解 AI Agent：设计原理与工程实践》（bojieli/ai-agent-book，Apache-2.0）
> 原文：book/chapter1.md ｜ 配套实验：chapter1/ 共 4 个（另含跨章引用 7-1/7-2）｜ 难度：🟢 入门级

## 一、本章一句话主旨

从实践出发建立理解 AI Agent 的基础框架：**Agent = 大脑（LLM）+ 眼睛（上下文）+ 手脚（工具）**；当模型能力逐渐商品化后，真正构成产品差异的是模型之外的 **Harness 工程**（上下文管理、工具接口、约束、验证、纠正）。

## 二、核心概念与公式

| 概念 | 原文术语 | 说明 |
|------|----------|------|
| 最小工程公式 | `Agent = LLM + 上下文 + 工具` | 加号为工程组件组合，非 RL 形式化定义；只描述 Agent 边界内实现，**不含 Environment** |
| 生产形态改写 | `Agent = Model + Harness`；`Harness = 上下文管理 + 工具接口 + 约束 + 验证 + 纠正`；`Agent ↔ Environment` | 最小 Demo 只需 Model + 能构造上下文/暴露工具的 Harness；生产系统再加约束、验证、纠正 |
| 组件映射 | 大脑=LLM/Policy；眼睛=上下文构造/Observation&History；手脚=工具与适配器/Action Interface | 三组件可映射到 RL 的策略与交互接口，但不严格等同 |
| 上下文五部分 | System Prompt、Tool Definitions、User Messages、Assistant Messages、Tool Results | 前两项为**静态前缀**，后三项为随交互增长的**动态轨迹** |
| 核心循环 | ReAct（Reasoning + Acting） | 思考（Thought）→ 行动（Action）→ 观察（Observation）循环，直到任务完成 |
| 模型即 Agent | Model as Agent | 先进模型经 RL 将工具调用**决策策略**内化为原生能力 |
| 学习三路径 | 上下文适应 / 外部产物 / 模型参数 | 不同时间尺度的协同机制 |
| Harness 五要素 | Context / Tools / Constrain / Verify / Correct | |

## 三、分节精读

### 3.1 现代 Agent = LLM + 上下文 + 工具

图 1-1 给出外层 Agent↔Environment 闭环与内层 Model–Harness 结构：Model 负责策略决策，Harness 环绕模型做上下文构造、工具暴露、循环与状态维护、权限验证。

> Agent = 大脑 + 眼睛 + 手脚。大脑负责思考和决策，眼睛接收环境提供的观察，手脚将决策转化为作用于环境的行动。

**观察空间与动作空间**：Harness 把环境观察转为上下文、把模型行动转为工具调用。原文强调：

> 在底层模型固定时，提升 Agent 任务表现最主要的系统工程手段，往往就是重新定义或扩展观察空间与动作空间。

Manus 合并 Coding/Deep Research/Computer Use 三者空间；OpenClaw 经消息渠道把接口延伸到用户数字生活。

**工具（手脚）**按互动方向分五类：感知工具、执行工具、协作工具、事件触发工具、用户沟通工具。工具调用（Tool Calling / Function Calling）四步：① 在上下文声明可用工具；② 模型自主判断是否/调用哪个/传何参数；③ 结果追加上下文；④ 模型决定下一步。设计原则：**通用基础能力用于组合与探索，专用工具用于约束高风险与强业务规则操作**（如支付、删除需专用且可审计），受限代码解释器须在隔离沙盒运行。

**LLM（大脑）**独特能力是**内部思考**——行动前先规划推演，不改外部环境却提升后续行动质量。

> 模型即 Agent：Harness 这个词原指马具……把马的力量引导到正确方向。模型是那匹强大但不可预测的马，Harness 是把能力引导成可靠任务执行的工程外壳。

引用 Rich Sutton《苦涩的教训》（[^ch1-1]）：约束验证会被模型逐步内化，但过程慢。

**学习机制三路径**：① 上下文适应（任务内，立即可调、不改持久状态）；② 外部产物（跨任务，知识文档/Prompt/Skill/程序，可审计）；③ 模型参数（训练周期，内化高维难显式表达的能力）。三者是协同而非互斥。

**上下文（眼睛）**由五个部分构成（见第二节表格）。实验 1-1（★★ 消融）结论：

> 工具定义缺失 → 无法调用工具，可能编造格式工整答案；工具执行结果缺失 → 闭环丧失，Agent“盲目”重试至预算耗尽；思考过程缺失 → 若能从工具结果重建则几乎无代价；历史消息缺失 → 冗余操作、重复犯错。

> “给出了回答”不等于“完成了任务”，上下文残缺时典型的失败不是报错退出，而是一个看上去毫无破绽的答案。

### 3.2 ReAct 循环

> Agent 执行任务的核心模式叫做 ReAct。实际循环包含三个环节：模型先思考当前该做什么，然后调用工具行动，再观察工具返回结果并继续思考下一步。这个“想→做→看→想→做→看”的循环不断重复，直到任务完成。

轨迹（trajectory）= 静态前缀（系统提示词 + 工具定义）+ 动态消息历史（用户消息 + 模型回复 + 工具结果）。最小运行骨架：

```python
trajectory = [user_request]
repeat:
    context = stable_prefix + trajectory
    decision = Model(context)
    trajectory.append(decision)
    if decision has no tool call:
        return decision.answer
    for call in decision.tool_calls:       # independent calls may run in parallel
        validated_call = Harness.validate(call)
        observation = Environment.execute(validated_call)
        trajectory.append(observation)
```

多币种收入汇总实例仅用 **3 次迭代、4 次工具调用**完成。基本设计中上下文只增不改，使系统可解释、可调试，轨迹可沉淀知识库或用于 RL 训练，形成从经验学习的闭环。

实验 1-2（★ Kimi K3 原生 Agent）：约 2.8 万亿参数 MoE，100 万 token 窗口、原生视觉、常开 thinking mode；RL 内化**工具调用决策策略**（何时调用、调哪个、传何参），工具执行由外部 Formula 服务端脚本引擎提供；长链工具调用稳定（200～300 次）。关键澄清（[^ch1-2]）：RL 内化的是决策，而非工具执行机制；编排循环从客户端移到服务端。

实验 1-3（★ GPT-5.6 原生 Deep Research）：采用 **Freeform Tool Calling**（`type:"custom"` 允许原始文本参数，非 JSON）；配合 Responses API 的 `web_search` 与 `code_interpreter` 内置工具实现“搜索→阅读→分析”迭代；引入模型层**意图澄清过程**。可用等价托管工具（阿里云百炼 qwen3.7-plus、Kimi K3 Formula）复现。

### 3.3 Harness 工程：模型之外的竞争力

基础 ReAct 暴露脆弱点：幻觉、选错工具、遇错无法自恢复。生产公式与五要素表（节选自原文）：

| 功能 | 职责与核心原则 | 实际例子 | 详见 |
|------|--------------|---------|------|
| Context | 提供感知信息；信息要充分 | 系统提示词、知识库、Agent 状态栏 | 二、三章 |
| Tools | 提供观察与行动；接口要清晰 | MCP 工具、代码解释器 | 四章 |
| Constrain | 设定行为边界；故障安全默认值 | Claude Code 每工具默认需授权 | 四章 |
| Verify | 自动判断结果对错；只看结构化数据 | Linter、类型系统 | 五、六章 |
| Correct | 发现问题自动修正或回退；不暴露中间态 | 静默重试、熔断机制 | 二、五章 |

> 行业正在从“能做事”向“可靠地做事”转变，Harness 工程因此成为 Agent 系统的核心竞争力。

**工程范式演进五阶段**（层层包含）：提示工程 → 上下文工程 → Harness 工程 → Loop 工程 → Graph 工程（2026-07）。LangChain 在 Terminal Bench 2.0 从 52.8% 升至 66.5%，改变的是 Harness 而非模型。

**构建有效 Agent 三原则**：① 保持简单（每多一层抽象都是新盲区）；② 保持透明（显示规划/日志/轨迹）；③ 设计好工具接口 **ACI（Agent-Computer Interface）**——从 Agent 视角而非程序员视角设计。模糊接口会被模型放大成系统性错误，对应制造业**防呆（Poka-yoke）**。

**选择模型**：闭源（OpenAI/Anthropic 领先、成本高）vs 开源（差距 6 月内、可私有化、工具调用差异大需测试）；**绝大多数 Agent 需要支持 Reasoning 的模型**；还需评估输出速度、多模态硬性要求。

**编排模式**：工作流（Workflow，预定义代码路径、确定性）如订机票 4 固定节点（核实身份→搜索航班→完成付款→确认预订）、文生图（LLM 改写 + 生图）；自主 Agent（执行路径由环境反馈实时决定，即 ReAct，需停止条件防死循环）。二者常混合——关键合规流程用工作流，灵活决策切自主。实验 1-4（★）对照文生图工作流路线与原生路线，说明 **Harness 给能力短板打的补丁会随模型变强被内化**。

**主流框架对比**（节选）：

| 框架/平台 | 核心定位 | 编排模式 | 适用场景 |
|----------|----------|----------|----------|
| Codex Harness | Codex 开源运行时 | 自主 | Coding Agent、嵌入自有产品 |
| Claude Agent SDK | 生产级框架 | 自主 | 复杂自主任务 |
| LangChain/LangGraph | 通用 LLM 框架 | 工作流+自主 | 链式思考、多步骤工作流 |
| n8n | 可视化自动化 | 工作流+自主 | 业务自动化、非技术团队 |
| Dify | LLM 应用平台 | 工作流+对话 | 企业级 RAG |
| CrewAI | 角色化多 Agent | Multi-Agent | 团队式任务分解 |
| OpenClaw | 开源全能个人 Agent | 自主+事件驱动 | 个人助理、多平台集成 |
| DeepSeek Harness | Agent 自进化框架 | 一切皆插件 | Agent 开发者 |
| Pi | 极简 Coding Agent | 自主 | Agent 开发者 |

**护栏与安全性（三层）**：上下文层（管能看到什么：相关性/安全分类器、内容审核、基于规则保护；含 Constitutional Classifiers——规则驱动合成数据 + 上下文联合判断 + 两级筛查）；执行层（管能做什么：工具风险评级、输出检查 PII/验证）；数据层（管世界最终被改成什么样：行级安全、约束校验器、受控视图）。人工干预触发：① 超过失败阈值；② 高风险/不可逆操作（如大额退款）。

**Harness 五要素与“构建”部分对应**：上下文管理→第二、三章；工具接口与约束→第四章；验证与纠正→第五、六章；安全为横切关注点挂三层护栏。

### 3.4 贯穿全书的设计模式

1. **提议者—审核者（Proposer-Reviewer）**：产出与评判由不共享上下文的角色分别承担，前提“自审不可靠”。
2. **渐进式披露（Progressive Disclosure）**：先给可检索目录，再按需加载细节，同时优化上下文预算与选择精度。
3. **只增不改（Append-only）**：状态以追加演进，换来可缓存、可重放、可审计。
4. **边界集 + 保留集（Boundary Set + Retention Set）**：修改须同时在“应当改变的样本”与“不应影响的样本”上验证。
5. **最小 diff + 可回滚**：每次修改尽量小、带来源、可单独回滚，使归因成为可能。

## 四、关键洞察与反直觉点

- **扩展眼睛和手脚（观察/动作空间）常比换更强模型更有效**：许多“需要更聪明模型”的问题其实是接口问题。
- **上下文决定能力上限**：“给出了回答”≠“完成了任务”；残缺上下文的失败常是无破绽的错误答案而非报错。
- **模型越强，Harness 越关键**：模型内化的是决策策略，约束/验证/纠正仍需工程外壳；Harness 补丁随模型变强被内化。
- **学习三路径是协同而非互斥**：上下文负责临场适应，外部产物负责可控积累，参数负责内化难显式表达的能力。
- **从工作流到自主 Agent 是降风险顺序**：先提示词→再工作流→最后自主。

## 五、重要表格 / 数据 / 公式

- 核心公式：`Agent = LLM + 上下文 + 工具`；`Agent = Model + Harness`，`Harness = 上下文管理 + 工具接口 + 约束 + 验证 + 纠正`。
- 上下文五部分：System Prompt / Tool Definitions（静态前缀）+ User Messages / Assistant Messages / Tool Results（动态轨迹）。
- 五类 Agent 产品对照（眼睛/手脚/策略）：Coding Agent、Deep Research、Browser Use、豆包手机助手、Pine AI 个人办事——动作空间均“开放式”，且能内部思考、持续交互。
- 消融实验 1-1 四组件缺失后果（见 3.1）。
- ReAct 实例数据：3 次迭代、4 次工具调用完成多币种汇总。

## 六、配套实验清单

| 实验目录 | 要证明的机制 | 运行入口 |
|----------|--------------|----------|
| chapter1/context/（1-1） | 系统性消融展示上下文各组件重要性；支持阿里云百炼/Qwen、SiliconFlow、字节 Doubao、Kimi 等多提供商 | `chapter1/context/`，配置 API Key 后运行（Starter 推荐从此开始） |
| chapter1/web-search-agent/（1-2） | Kimi K3 模型即 Agent，原生深度搜索、多轮搜索与信息整合 | `chapter1/web-search-agent/` |
| chapter1/search-codegen/（1-3） | 模型自主多轮搜索 + 服务端代码执行 Deep Research 闭环；先澄清意图再执行 | `chapter1/search-codegen/` |
| chapter1/image-gen-workflow/（1-4） | 具体/宽泛需求 × 工作流（kimi-k3 改写+通义万相）与原生（Gemini/GPT-Image 2）双路线对照 | `chapter1/image-gen-workflow/` |
| learning-from-experience/（7-1,7-2，跨章） | Q-learning 与官方 Kimi K3 双臂实测（原属第 7 章，列于本章 README） | 见对应章节目录 |

> 注：各实验 “Run first” 具体命令（如 `python main.py`、provider 配置）在各子目录 README 内，本级 README 仅给出目录、类型（✅ 可独立运行）与验收要点，原文未在此处展开逐条命令。

## 七、本章小结与思考题要点

**本章小结要点**：Agent = 大脑+眼睛+手脚，三者缺一不可；扩展眼睛/手脚是最主要能力杠杆；上下文由静态前缀+动态轨迹构成，各组件不等价（消融实验）；Harness 是竞争力所在（约束/验证/纠正）；从工作流到自主 Agent 是降风险顺序；五个设计模式贯穿全书；安全是架构问题（三层护栏）；下一章深入上下文工程，RL 学术渊源留待第 8 章。

**思考题（共 10 道，原文难度标记）**：
1. ★★ 只增一项能力（更强模型/更丰富上下文/更多工具）选哪个？条件如何改变选择？
2. ★★★ ReAct 累计缓存读取量随轮数近似二次方增长，如何降低？
3. ★★ “模型即 Agent” 与 Harness 重要性上升如何共存？框架未来核心价值？
4. ★★ 除工具结果缺失外，还有哪些情况让 Agent 陷入重试循环？如何检测与终止？
5. ★ 用感知/行动/策略三维度分析一个日常 AI 产品并提改进。
6. ★★ 航班订票客服选工作流还是自主？能否混合？
7. ★★★ 工具多数低风险但特定参数组合高风险（如 `delete_file`），如何设计动态风险评估？
8. ★★ 受限动作空间（预定义选项）何时优于开放式？
9. ★★ 人工干预时用户不在线/响应慢/指令模糊，Agent 怎么办？
10. ★★★ 举一个会随模型进步而过时的 Agent 工程手段并说明。

## 八、术语表

| 术语 | 含义 |
|------|------|
| LLM | 大语言模型（Large Language Model），Agent 的大脑/决策核心 |
| ReAct | Reasoning + Acting，思考-行动-观察循环 |
| Harness | 环绕模型的运行与治理层（上下文/工具/约束/验证/纠正），喻为“马具” |
| MoE | Mixture of Experts 混合专家，Kimi K3 采用的稀疏激活架构 |
| ACI | Agent-Computer Interface，从 Agent 视角设计的工具接口 |
| 防呆 Poka-yoke | 丰田生产体系术语，用设计消除错误 |
| 消融实验 Ablation | 系统性移除组件以评估其贡献的实验 |
| 轨迹 Trajectory | Agent 执行中不断累积的消息历史 |
| 上下文适应/外部产物/模型参数 | 学习机制的三条时间尺度路径 |
| 工作流 Workflow | 预定义确定性代码路径编排 |
| 自主 Agent | 执行路径由环境反馈实时决定 |
| Constitutional Classifiers | Anthropic 宪法分类器，两级筛查的越狱防御 |

## 九、与其他章节的关联

- **第 2 章 上下文工程**：落实 Harness 的 Context 要素（提示工程、Agent 状态栏、上下文压缩、Agent Skills），也是本书最关键一章。
- **第 3 章**：用户记忆与知识库（跨会话的上下文管理 / 外部产物）。
- **第 4 章**：工具（Tools 要素、MCP、权限控制、主动工具发现）。
- **第 5、6 章**：Coding Agent 与通用 Agent 的验证与纠正（Verify/Correct）。
- **第 7 章**：从经验中学习（learning-from-experience，实验 7-1/7-2 已在本章 README 引用）。
- **第 8 章**：Agent 在强化学习中的学术渊源、传统 RL 与现代 LLM Agent 对比（本章小结预告）。
