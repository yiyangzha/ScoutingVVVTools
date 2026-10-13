# CMS VVV scouting 全流程任务交接（适用于 Codex / Claude 等 agent）

请接手下面的任务，先读取现有完整记录、要求和代码，再继续推进。用户需要实际完成分析全流程，不能只交付适配代码、方案或提交命令。本文同时记录已确认的决定、已完成的工作、尚未确认的问题和下一步建议。使用你当前环境支持的工具，不依赖某一种 agent 的专有工具或模式。

上一 agent 的最后一项指令是：停止修改和运行分析，只在 `tmp/prompt.md` 写好完整交接文档。本文就是这次交接的产物；上一 agent 没有在收到该指令后继续实施或运行分析。接手后按用户给你的启动指令恢复原全流程任务。历史计划中的“提问工具不可用、等待重启”等状态已被后来的恢复和回答取代，不能把它们当作当前阻塞。

## 1. 工作位置、原始目标和完成标准

- 仓库：`/eos/home-y/yiyangz/CMS_Run3_VVV_Scouting/ScoutingVVVTools_un`。
- 当前环境是 CERN lxplus；前次检查主机为 `lxplus948.cern.ch`。交接日期为 2026-10-01，时区 Europe/Zurich。
- 代码此前在其他 cluster 上运行过，后来增加了新的系统误差设置。要保留以前的 cluster 支持，增加 lxplus 适配，并在 lxplus 实际重新跑完整流程。
- **从 `src/sample.json` 注册的 MC 和 data 源数据出发，重新产生 ntuple，随后重新产生所有派生输入和最终结果。** 不能把以前的 ntuple、模型、JMS/JMR 拟合、系统误差 JSON 或其他旧产物当作本次新产物。
- 所有 MC 都可以使用；data 暂时只使用 `2024G_UParT_v6`。所有 MC 可用不等于可以把重复 dataset、inclusive/exclusive 重叠样本无条件相加；具体物理分组有不确定性必须问用户。
- 所有下游配置都必须对应本次新产生的输入。保留旧结果，建立新的独立输出路径并记录来源。
- Condor 必须通过 CERN `module load lxbatch/eossubmit` 使用。可以并行提交和运行，避免直接 local 运行受到资源限制。
- 对需要 GPU 的阶段，先调查可用方案并尝试合理的 batch GPU 路径；确实无法满足再问用户。不要未经确认就换模型或掩盖 GPU 没有真正启用。
- 完整交付至少包括：新 pileup 权重、ttbar 控制区 ntuple、新 JMS/JMR 拟合、五个分析树的 nominal 和八个 JES/JER/JMS/JMR variation ntuple、明确的混样/训练流程、每个分析树的模型及诊断、新信号区、QCD ABCD 背景估计、所有可评估 syst 比值、新 Data/MC 图、包含全部可评估已启用 nuisance 的 combine 预期显著度和预期限值。
- 五个分析树为 `boosted3`、`boosted2_resolved`、`boosted2_unresolved`、`boosted2_leptonic_1lep`、`boosted2_leptonic_2lep`；这与 classifier 的 process classes 是不同概念。
- 独立旧版 b-veto、触发效率等研究存在输入/定义差异，属于需要明确范围的问题；不能悄悄省略，也不能擅自用旧 data 跑。
- 必须监控并处理失败/held 作业，验证输入文件覆盖、batch/merge 完整性、nominal/variation 树、权重、模型与 SR 的一致性，以及 combine 输出。提交作业、退出码为零或生成了文件都不等于完成。

## 2. 用户的工作规则与授权

1. **有任何问题及时问，不猜。** 不懂、不确定、不清楚、奇怪、看起来不对、互相矛盾、要求无法满足、代码问题、物理缺陷等，都要及时提出。说明证据、影响、具体选项、建议及理由，一轮一轮问；得到答案后复述并记录。
2. **确保问题可见。** 优先使用当前平台可见、能等待用户回答的提问/选项工具。不要写死 Codex 的 Plan 模式或某个工具名称，也不要依赖用户看不到的异步卡片。若平台无法提供可靠的可见提问，清晰列出问题、选项和建议，然后停止执行等待答复，避免后续刷新把问题淹没。时间过去、工具返回空答案或用户没有看到，都不是确认。
3. 用户已明确授权本任务中的运行、编译、提交、重跑、必要的普通 cluster 适配和直接 bug 修复。正常执行不用反复问运行许可。此授权不会自动批准新的物理定义、功能或分析类别变更；这些必须先说明方案并由用户确认。
4. **认真阅读完整代码及数据流后再设计变更。** 不仅看 grep 片段。先复述理解，记录具体计划：改哪些文件/函数、行为、配置项、输出格式、下游影响和文档。对已经授权的常规执行/适配可继续；物理/功能改动的具体方案需要确认。
5. **持久记录。** log、plan、progress 和其他记录放在仓库 `./tmp/`，不纳入 git。及时记录进展、用户回答、新要求和失败原因。新用户要求通常应及时更新 `AGENTS.md`；代码变更要同步更新 `AGENTS.md` 和 `README.md`。本次最后指令限定上一 agent 只写本文，因此最新 SR 决定和交接要求先完整记录在本文，旧 progress 的下一步字段尚未更新。
6. **每个小阶段结束后，完整逐字逐句重新阅读整个 log、整个 plan、整个 progress 和全部要求。** 包括所有早期内容，不能只读摘要、grep 或新增部分；在 log 记录完成的检查点。输出被工具截断时必须补读。
7. **只在仓库内写入。** 所有环境、缓存、编译产物、临时文件、证书、日志、中间产物也只能放在当前仓库内。不得修改用户级配置或写外部路径。batch 环境若确有无法满足的路径限制，问用户，不悄悄放宽。
8. **不安装、不下载未明确授权的文件、工具、包、依赖或外部资源。** 唯一已明确批准的新增外部校准下载是下文的三份官方 PU ROOT 文件，已下载完成。其他缺失依赖（包括 CombinedLimit 等）需要具体说明来源、版本、动作并取得确认。使用现有软件优先。
9. 仓库还要求在打开远端 XRootD/DAS 输入前确认。用户已选择核实 MC 源元信息，DAS parent/McM 调查已执行；尚未打开任何远端 ROOT 事件文件。按已有授权和具体调查范围判断后续动作，有不确定性就问，不能把元信息查询扩张成不受限下载。
10. **不做 local tests。** 用户对本次实际分析运行/编译已有授权，但本地测试禁令仍有效。需要验证代码时安排合适的 batch worker 检查；不要执行本地测试或用“语法检查”规避该约束。
11. **不得删除任何文件或目录。** 禁止 `rm`、清理、unlink、替换前先删、删除临时文件等。保留旧结果和用户数据；尽量采用新增路径。不得恢复用户删掉的代码，除非确认。不得在两次读取之间覆盖用户编辑。
12. **Git 只读。** 可用 status/diff/log/show/check-ignore 等；禁止 commit、push、fetch、切分支、stash、reset、checkout、rebase 等改变历史或工作树的 git 操作。证书、中间产物、log 等不能跟踪。新增 `.gitignore` 规则通常需要确认；本次用户明确要求 `./tmp` 忽略，已增加 `/tmp/`。
13. 做最小、根因层面的修复。不要顺手重构、重排风格、加功能或改无关代码。发现无关问题要报告，不能擅自修。表达逻辑用肯定、直接的描述。
14. 跟用户用中文；代码注释、`AGENTS.md`、`README.md` 用英文并匹配现有结构风格。工具新增目录/配置/注册规则和 Python/C++ 具体风格请完整读取仓库 `AGENTS.md`。
15. 保持既有环境设计：C++ 转换使用 CMSSW（当前要求 `CMSSW_16_1_0_pre4` 的 correctionlib），Python 用 pixi 或 CMSSW，combine 用已有/获准建立且含构建好的 `HiggsAnalysis/CombinedLimit` 的 CMSSW 区域。
16. 对版本相关、变化较快的行为查官方资料并给出处，区分证据、推断和假设。不要用论坛猜测修物理。
17. 保持进度交流简洁、及时；最终汇报实际完成的结果、验证、未解决问题和风险。不能承诺已经完成未运行的工作。
18. **完整物理一致性。** 能评估的 syst 都要进入 combine；已启用 nuisance 缺少输入必须报错，不能悄悄给 1、漏掉或当成功。

## 3. 已确认的物理和流程决定

### Luminosity / data

- 预期灵敏度继续使用原始目标 **255.8 fb^-1**。
- 只用所列 v6 G data，原始 luminosity 是 **27.51 fb^-1**，用户要求归一/外推到 255.8。
- Data/MC 显示外推应同步缩放 data 误差，标明外推；不能把外推当成实际新增统计量。
- 曾进一步询问 JMS/JMR 控制区拟合是否也做外推，卡片返回空答案；该具体范围没有收到确认。保持这一事实清楚：控制区实际统计精度不能被无依据地提高。涉及实现范围时根据证据提出明确问题，不把空答案当许可。

### Pileup

- 保留原来的 **2024 C–I data pileup profile**，不是改成仅 G profile。
- 重新计算每个 MC 的 PU 权重。
- 用户明确允许获取 README 中的三份官方 `dataPileupHistogram-2024CDEFGHI_Golden-{66000,69200,72400}ub.root`。它们已经获取，见第 5 节。

### 重复 DY / MC 权重

- 相同 DY dataset 被注册为 `dytoll_ht400to800_mll120`（56.84）和 `dy_h400to800_m120`（175.3）。用户要求**先核实正确 cross section**，再选择 canonical name 并更新引用。两个数值都没有得到认可，不能凭大小或常识选。
- 对现有 MC 权重/归一化问题，用户选择：**先检查源 Runs 元信息、generator 正负权重、上游过滤和归一化，再提出全链路一致的修正方案给用户确认。** 这是调查授权，不是任意新权重定义的批准。
- 普通 tree entry 数、源生成统计、过滤效率和 signed sum of weights 可能不是同一个归一化量。必须根据真实生产元信息解决，不能只在单个脚本补一个 `genWeight`。

### AK4 cleaning（已批准的物理定义）

- 用户明确选择：resolved AK4 **只避开保留的两个 AK8**，不是避开所有超过 AK8 collection 阈值的 AK8。
- 此定义要落实到本次 fresh nominal 和所有 jet variation 转换及全部下游结果。
- 最小正确实现需要检查 collection ordering/索引/对象身份；不能用不可靠的 pT 等值判断代替选中的两个 jet。实现方案仍需完整记录。

### 信号区最优解（最后一次已收到的回答）

- 用户问：“哪个合理？哪个找到的才是最优解用哪个”。不能仅凭默认配置选 optimizer。
- 用户随后选择优先**最终含全部 syst 的 combine 灵敏度/预期显著度**，并补充：**如果计算时间太长或不现实，就算了，可以寻找近似等办法。** 因此不要求不计代价做不可实现的全空间精确搜索。
- 合理的候选执行方案：在相同输入、模型、score axes、约束和 luminosity 下比较两种现有优化器，再对有代表性的好解用含 syst 的 combine 复评。如果成本过高，可提出有依据的近似/分阶段方法。具体方法、计算预算、停止标准仍应透明说明；出现不确定性及时问。
- 两种现有代码的候选生成均有启发式限制。一般矩形版本即使 branch-and-bound 完全结束，也只证明其**有限生成候选集**的最优性；histogram 版有候选截断及后续局部 refinement。不能把“最好找到的解”“有限候选集精确最优”宣传成所有连续矩形的全局最优。
- 现有两种 optimizer 都优化统计-only 的 `sqrt(sum Z_i^2)`，这与最终含相关 syst 的 combine 显著度不等价。最新选择已解决优先目标，但尚未实施新的复评方案。

## 4. 持久记录和当前真实进度

先完整读取以下文件及当前 git diff。本文不能代替逐字读取要求：

- `AGENTS.md`
- `README.md`
- `tmp/lxplus_workflow_20261001_plan.md`
- `tmp/lxplus_workflow_20261001_log.md`
- `tmp/lxplus_workflow_20261001_progress.json`
- 本文 `tmp/prompt.md`

这些记录有完整历史，包括提问工具修复、重启、第一/第二轮回答和校准下载来源。历史“没有收到答案”“没有任何下载”描述对应当时状态，后来的明确记录已取代它们。最新 SR 回答及只写交接要求仅在本文；接手应把当前状态同步到正式记录，保留历史。

**截至交接：**

- 已做环境检查、证书备份、广泛完整源代码阅读、元信息调查和三份获准 PU 校准下载。
- **尚未修改任何 pipeline 实现代码；尚未编译；尚未运行 pipeline；尚未提交任何 campaign 作业；未做 local tests；未安装软件依赖；未打开远端 ROOT 事件输入。**
- tracked 修改仅 `.gitignore` 和 `AGENTS.md`。`tmp/` 全部忽略；证书在既有忽略的 `build/` 下。
- 源码审计还没有达到“仓库全部代码/配置均完整读完”的状态，不能直接宣称读完后跳过剩余项目。
- 还没有 fresh ntuple、PU 权重 CSV、fresh fit/model/SR/syst/combine 等结果；现有旧输出必须保留，不能被当作本次完成证据。

### 已读范围与剩余阅读

此前已完整阅读主数据流：`run.py`、`run.sh`、`src/sample.json`、`src/simple_json.h`、`weight.C`、`convert_branch.C`、`mix.C`、`train.py`、`model_io.py`、JMS/JMR fitter、theory/PU/四种 jet-syst 及共享/merge 脚本、两种 SR Python optimizer 和 OpenMP selector、`qcd_est.py`、`combine.C`、`data_mc.py`、`class_shapes.py`、`b_veto.C`、`ttbar_score.py`、conversion QA 与 hybrid 副本完整差异，以及独立 systematics/diagnostic 源文件和 README。大段截断内容此前做过补读。

主要 stage configs、CR branch/selection configs、训练 feature/selection configs 和若干 variants 已读。主 `convert/branch.json` 已通过完整 inputs 和所有 **234 个 distinct 完整 output 定义**的 catalog 阅读，但仍需检查精确的逐树顺序/依赖，不能称其 6035 行逐行读完。

明确剩余：

- 主 `convert/selection.json` 的后三种类别公式需要补充无截断完整阅读（`boosted2_unresolved`、两个 leptonic 类别）；前两类及 collection 定义已读。
- 主 branch config 的逐树完整 output 排序/映射。
- hybrid 和 b-veto-study 的 branch/selection variant 尚未完全闭合阅读。
- 根据真实文件 inventory 核实是否有其他遗漏，尤其 legacy/独立流程；不要仅依赖上一 agent 的口头覆盖列表。

## 5. 已准备的环境、证书、校准和元信息

### 证书

- 活动 proxy 原来在 `/tmp/x509up_u165165`。
- 用户明确要求的备份为仓库内 `build/credentials/grid_proxy_20261001`。目录权限 700，文件 600；备份与原文件 `cmp` 一致，git ignored，未打印内容。
- 本次运行使用备份的绝对路径设置 `X509_USER_PROXY`。不要依赖重新登录后可能变化的默认 proxy。
- 备份保持原来的 expiration，不会自动延长。最后记录约剩 7 天（`2026-10-01 19:01:35 UTC` 检查为 602278 秒）；接手要重新只读检查有效期，必要时及时问用户续证。
- 保护 credential 内容，维持目录 700 / 文件 600 权限；日志和 git 只记录必要的非敏感状态。

### eossubmit / software / GPU

- `eossubmit` 是 module，不是独立 executable 或名为 eossubmit 的 schedd。
- `/etc/modulefiles/lxbatch/eossubmit` 设置 `_myschedd_POOL=eossubmit`，collector 为 tweetybird03/04；加载后曾选择 `bigbird24.cern.ch`，当时用户 queue 为空。这只是当时快照，接手重新查。
- read-only inventory 曾显示 95 GPU nodes / 130 GPUs；不代表都空闲，也没有 GPU 作业 match/执行验证。
- `condor_submit`、`condor_q`、`condor_status`、`voms-proxy-info`、`root-config`、pixi executable、python3、dasgoclient、scram、curl、jq 可用；系统 `root-config` 为 6.40.04。
- CVMFS 上存在 gcc13 和 gcc14 的 `CMSSW_16_1_0_pre4`。gcc13 tool XML 显示 Python 3.9.14、correctionlib 2.7.0、XGBoost C++ 1.7.5；这不等于已经确认 Python XGBoost/CUDA 和所有 Python 依赖可用。
- 仓库初检没有本地 CMSSW、pixi environment、dataset。尚未确认 combine 软件可用，也未安装依赖。先检查中央现成软件；缺东西按精确动作询问。
- 不覆盖或删除 tracked binary `selections/convert/convert_branch`。新编译输出放仓库内已忽略的 `build/` 或明确 campaign 路径。

### 已下载的三份 PU 文件

目录：`tmp/lxplus_full_20261001/inputs/pileup/`。来源是 README 的官方链接：

`https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions24/PileUp/`

| 文件 | bytes | SHA256 |
| --- | ---: | --- |
| `dataPileupHistogram-2024CDEFGHI_Golden-66000ub.root` | 4470 | `02be96ef021370cb8c6e6d66e49bc40bf3e9e9c75711b7d7491c70d7c2f97454` |
| `dataPileupHistogram-2024CDEFGHI_Golden-69200ub.root` | 4466 | `5e379a2004788f693187f6a579a81f2530a27a00482e39b559337d026c40fb93` |
| `dataPileupHistogram-2024CDEFGHI_Golden-72400ub.root` | 4468 | `3b85a78a4a1ee929edccf9b785e5c50095022cdd74e5a29539fba4842dfcb253` |

通过备份证书 curl 获取，使用 CA 验证，没有 `--insecure`。只做了 ROOT signature/version/compression 检查（version 63009，compression 101）；histogram 语义/归一化内容验证还没在 worker 做。保留文件，不重新下载/覆盖。

### DY 调查证据

两个 alias 的 source 完全相同：

```text
/DYto2L-4Jets_Bin-HT-400to800-MLL-120_TuneCP5_13p6TeV_madgraphMLM-pythia8/yiyangz-ScoutingNano_MC_DYto2L-4Jets_Bin-HT-400to800-MLL-120_260713_v5-00000000000000000000000000000000/USER
```

- `dytoll_ht400to800_mll120`：sample_ID 38，xsection 56.84，raw_entries 1775774；当前训练引用它。
- `dy_h400to800_m120`：sample_ID 45，xsection 175.3，同样 raw_entries；ttbar CR 引用它。
- 56.84 与一个不同的 zjets_ht2500 注册值相同，疑似复制问题，但不能据此确认另一个值正确。
- DAS parent 查询成功，保存为 `tmp/lxplus_full_20261001/metadata/dy_parent_attempt_1.json` 和 `.stderr`。
- parent 是：

```text
/DYto2L-4Jets_Bin-HT-400to800-MLL-120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v4/MINIAODSIM
```

- 官方 McM public API response 保存为 `tmp/lxplus_full_20261001/metadata/dy_mcm_by_primary_attempt_1.json`。
- `GEN-RunIII2024Summer24MiniAODv6-00583` 对应上述 exact parent；上游 `GEN-RunIII2024Summer24wmLHEGS-00585` 注册 `cross_section=1`、`filter_efficiency=0.07`、`negative_weights_fraction=0`，返回的 validation 没有 measured cross section。这个 1 **不能直接当科学上正确的分析 cross section**，两个现有值都仍未验证。
- 后续可以调查生成 run 元信息，如 `GenRunInfoProduct.internalXSec` 和 filter/matching 产品，确认单位与生产过滤；`crossSection()` 可能优先 external LO，要避免误把占位值当结果。该读取方案尚未执行。
- 官方来源：`https://cms-pdmv-prod.web.cern.ch/mcm/public/restapi/requests`；具体 response 应结合 exact dataset/request 阅读。

## 6. 已发现的实现问题与物理风险（需要分类处理）

下面是源代码静态审计发现，**并非都已通过实际数据复现**。逐项区分直接 bug、待验证假设、物理定义和独立研究范围。普通明确 bug 可按已有授权最小修复；物理/功能变更仍问用户。不要把这一清单当已经获准修改全部代码。

### 提交、文件安全与完整性

- `run.py` 主要支持 SLURM，尚无 Condor backend；代理复制写 Purdue `/depot`，OpenMP probe 在仓库外建临时文件，编译会覆盖 tracked convert binary，local 退出会删除 binary。需要根因层面的安全路径/lxplus 适配，同时保留原 cluster 支持。
- 需要新 output namespace、来源/配置快照、job manifest、独立 logs、上游完成依赖，以及批次/文件覆盖检查。shared sample metadata 更新要序列化或采用正确的独立元信息汇总方式。
- conversion 遇到部分 remote-open/tree/schema failure 可以 skip 文件而 batch 成功；部分理论输入缺失填 unity。不能用“任务成功”替代完整性，已启用 syst 缺输入是错误。
- optional input branch 缺失时，buffer/bound 状态可能残留上一输入文件值；需核实并正确处理，不能制造假对象。
- PDF/scale/PS fixed buffer 大小与源计数缺少充分保护；需要真实 schema 和 count 验证。
- fresh namespace 可避免复用旧 resume products；现有 resume 验证并不绑定完整配置指纹。

### MC physics 权重与 theory

- 默认归一化常用 `lumi × xsection × N_tree / raw_entries`，按 loaded split 的 raw-weight 和重新分配整样本目标；目前 raw_w 常为 PU，不包含 generator sign。
- source 注释称输入已经 upstream-filtered，可能需要外部归一化分母，但 conversion 默认会覆盖 `raw_entries`。并行 merge 还共用同一个 sample.json 临时路径，存在 race。
- genWeight 的物理处理、筛选前分母、test/capped subset 外推和 PU 对 acceptance 的影响必须全链路一致，涉及训练、SR、QCD、plotting、theory/jet-syst、combine。用户已选择先核实再给方案，不能只补一处。
- nominal 已配置 genWeight scalar，而有 theory 的样本另一次 book 同名 branch，存在重复 branch 风险，需要验证 ROOT 行为和最小正确修复。
- PDF members、alpha_s、LHEScaleWeight、PSWeight 布局不能假定所有样本相同；源元信息调查尚未执行。部分 QCD `has_theory_weights=false` 不一定代表源无可评估权重。

### 类别、模型、混样和参考验证

- `ak8_kept2` 当前实际收集所有 pt>175 的 AK8，没有只保留两个；用户已确认 desired AK4 cleaning，见第 3 节。
- 当前主模型是覆盖五个分析树的 **BDT**，process class_groups 是 VVV、VH、Top、VV、VJets、QCD 六类，不应自动改 NN。主 config 有 Purdue input_root、unmixed pattern，而 mix stage/config/documentation 有 mixed 路径，具体用哪种训练流要澄清并与 variations 的 split/关联保持一致。
- 当前 class_groups 未包含所有注册 MC；某些 inclusive/exclusive VV 可能重叠，低 HT QCD、某些 ZZ channel 等使用范围也需核实。不能擅自合并或删类别。
- 每样本训练/test event cap、固定顺序 split、mix ordering 和保存的 references 决定下游必须精确匹配；实际按当前代码检查，旧文档关于 full test tail 的描述可能与 config/test cap 不一致。
- 一些 SR/plotting reference validation 调用被禁用；QCD weight validation 被注释但还输出通过信息，概率比较容忍度也被放宽。这些曾删除/禁用的逻辑不能未经用户确认恢复，不能对未执行的检查报成功。
- legitimate zero-entry sample/category 可能使训练等阶段直接失败；combine 要求各 signal sample 存在，部分零 channel 的处理需一致。先真实检查，不为了通过而伪造、clamp 或 regularize 缺输入。
- BDT CUDA 尝试后捕获宽泛 XGBoostError 自动 CPU fallback；中央 XGBoost 1.7.5 可能不支持当前 `device` 用法。须确认实际版本/GPU 支持，不能把运行成功当 GPU 真使用。
- CPU threads 常按 `os.cpu_count()` capped32，可能忽略 Condor allocation；custom decorrelation objective 依赖 NumPy，仍有 CPU 成本。需要 worker resource 适配。

### 校准与 systematics

- 现有 JMS/JMR results 是旧 lumi 11.56 的产物，本次必须重做。fit 会 skip missing samples；文档称无 GenPart 而 CR config 要求 GenPart，需要检查实际 source schema。
- 名义/八种 jet variation、corrected selections、model features 和 SR migration 要保持一致，新 cleaning 同时应用。
- JER 使用 scouting/offline proxy、rho recovery，JMS/JMR 来自 CR；输入校准和实现适用性要从真实数据及官方证据核实。有异常必须问，不随意更改定义。
- README 还列有 residual JEC、tagging/lepton efficiency、unclustered MET、normalization 等未实现或缺标定的方面。用户要求所有可评估 syst 都进入 combine；需要判断真实可评估范围，缺资源/物理定义及时问，不能悄悄称已经全覆盖。
- 部分 theory nuisance 目前跨 process/channel 相关性定义可能需要物理审查，不要静默更改。

### SR / QCD / combine

- 主 SR optimizer 与下游 `*_hist` 默认路径不同；histogram optimizer 还未注册为主 runner mode。需要公平比较并让下游指向实际选定的本次结果。
- branch-and-bound 证书范围与 candidate generation/refinement 的顺序要准确解释，最终含 syst 的复评选择按最新用户决定控制计算成本。
- 明确格式 bug：Uproot >=5.7 字典 assignment 默认写 RNTuple，而 combine 对 metadata 使用 `dynamic_cast<TTree*>`。`qcd_est.py` 的 `metadata/signal_regions` / `abcd_closure` 应明确写 TTree（例如合适的 `mktree`），让 producer/consumer 格式一致。该最小兼容修复属于普通 bug 范围，但尚未实施。
- 官方依据：`https://uproot.readthedocs.io/en/stable/basic.html#writing-ttrees-to-a-file`。
- combine 输出缺 quantiles/失败 fit 的一些路径写 0/inf 并继续；最终汇报要明确这种情况是缺失/失败，不当真实测量或成功。
- combine 配置 `keep_work=false` 会清理目录，与用户禁止删除冲突；本次保留 work 文件。

### 独立研究

- b-veto study 使用旧 hardcoded UParT v2 source 和不同选择。
- trigger `efficiency.C` 默认使用 2024F ZeroBias，超出仅 G/v6 data 范围，且有 fixed-array 风险。
- `abcd_2d_study.py` 有旧 Purdue/fat2 路径；`plot_asimov_mu_scan.py` 的近似抛物线图不能当实际 likelihood scan。
- duplicate-entry diagnostic 大样本内存重，且 skip file 可能掩盖覆盖；不能作为唯一 completeness 证明。
- 这些都没有运行；需要与用户明确 ancillary 范围/数据/物理定义后安排。

## 7. 接手建议（不是替用户批准未确认的物理方案）

1. 完整恢复记录，复述当前任务和决定。把最新含 syst 的 SR 优先目标、计算成本可采用近似、跨 agent 提问要求同步到 plan/log/progress/AGENTS；确认当前 filesystem/git 状态，没有用户新编辑冲突。
2. 完成剩余源码/配置阅读，记录准确覆盖，不宣称已读完未读内容。小阶段结束完整复读所有记录和要求。
3. 检查备份证书期限、eossubmit、中央 CMSSW/Python/Combine/GPU 条件。优先复用现有软件；需要安装下载时提出具体请求。
4. 准备可审查的元信息调查方案，利用 batch worker 核实 MC 源 schema、Runs、genWeight、sum weights、过滤分母、theory weight layouts 和 DY cross section；必要时及时问数据生产者信息。记录证据，不猜占位值。
5. 给出 coherent 全链路权重修正方案并等待用户确认；落实已确认的两个 AK8 cleaning 定义和普通安全/格式/cluster fixes，记录文件函数及下游影响，保持 README/AGENTS 同步。
6. 建立 fresh campaign paths/configs/manifests，在依赖完整后依次运行 PU、CR、JMS/JMR、分析转换、混样/训练、SR、ABCD、syst、plots、combine。根据 resources 并行化，避免共享元信息竞争和旧产物混入。
7. 对 SR 采用合理成本的最终 syst combine 复评/近似策略，透明报告 best-found 解及证书范围，不能伪称连续空间全局最优。
8. 持续处理 jobs/输入/物理异常，按可见提问原则询问；所有问题都及时问，不能为了完成而掩盖。完成后给实际输出路径、完整性/物理一致性验证、job 状态、剩余风险，并核实新增文件 ignored/tracked 状态。

**始终记住：用户要的是从 MC/data 重做并完整跑完全链路；有任何问题、不确定或要求无法满足都及时问。已授权的正常工作持续推进，未经确认的物理定义不猜。**

## 8. 本次交接文档的检查点

写作后已完整复读本文，并完整复读当前 campaign plan、log、progress 和全部 426 行 `AGENTS.md`（连续无截断范围），核对历史与最新回答。由于用户最后限定只写 prompt，本检查点记在本文，未继续修改其他记录或分析代码。`git status --short` 仍只有此前的 `.gitignore` 和 `AGENTS.md` tracked 修改；`git check-ignore -v` 确认本文和三份记录由 `/tmp/` 忽略、证书由 `build/` 忽略；`git diff --check` 通过。这些都是文档/只读检查，未编译、测试、运行分析或提交作业。
