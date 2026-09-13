# ELF：GPU 0、1 补齐质量实验

**此质量队列已完成32/32格。最新结果归档与用户授权的正式timing见[文档16](16_elf_results_and_timing.md)。以下保留此前质量执行步骤，不要重复启动生成。**

本页替代文档11–14中手动补跑和旧GPU2队列的安排。使用服务器 `~/diffusion_baseline` 中已经下载的 ELF 权重、数据和 `dlb-elf` 环境；本地只同步代码。此前要求暂缓的 timing 仍不启动。

## 本次范围与已知进度

| 实验 | 本轮完整步数 | 每个步数的数据量 | 用户日志已确认 | 本轮动作 |
|---|---|---|---|---|
| OWT unconditional | 1/2/4/8/16/32/1024 | 1024条 | 8/16/32；64为辅助结果 | 核验并复用8/16/32，补缺失格子 |
| OWT 前缀续写 conditional | 1/2/4/8/16/32/1024 | 1024 prompts；前256各5个completion，共2048条 | 尚未收到结果 | 准备与旧baseline相同的prompts，再补齐 |
| WMT14 validation | 1/2/4/8/16/32/64 | 3000条 | EOS修复后的8/16/32/64 | 核验复用，补1/2/4 |
| XSum validation | 1/2/4/8/16/32/64 | 11332条 | EOS修复后的8/16/32/64 | 核验复用，补1/2/4 |
| WMT14 test | 32/64 | 3003条 | 未看到四个test配置的完整清单 | 只补未完成项 |
| XSum test | 32/64 | 11334条 | 看到一份全集结果，步数需查manifest | 只补未完成项 |

共32个配置，包含已有结果，不是重新跑32次。调度器按步数由小到大派发，两张卡各有一个独立worker；哪张卡先空闲就领下一格。1024步的两项OWT任务在队列末尾，运行最久。生成/评测batch均为8、seed=42、compile关闭；每个子进程只看见一张GPU，内部使用逻辑 `cuda:0`。用UUID绑定物理GPU0/1，避免逻辑编号混淆。

OWT输出 Gen.PPL、entropy和长度诊断。前缀续写输出排除prompt loss的conditional PPL、reference conditional PPL、entropy和grouped Self-BLEU；MAUVE是现有可选评分，本队列不默认计算。WMT14看BLEU；XSum看ROUGE-1/2/L。两项任务并非都是翻译：XSum是摘要。

**与旧OWT表的边界：** 本轮对齐步数、1024样本、seed和GPT-2 Large评分模型/函数；继续使用公开ELF的1024 T5-token生成canvas及当前EOS处理。它不是旧方法的1024 GPT-2-token生成设置，entropy/EOS口径也未完全统一。前缀任务复用旧64 GPT-2-token的文本prompt，T5重编码后生成，评分取最多64 GPT-2-token续写；标记为 `c64_text_t5_v1`。这些结果用于公开ELF基线，不能声称与旧表所有setting完全一致。换成GPU0/1或重跑同样脚本不会消除差异。本轮不重新训练或更换词表，也不根据test分数调整采样参数。

## 1. 同步代码，选择持久终端

先确认此前手动启动的ELF作业已经结束，避免重复跑同一配置。在服务器执行：

```bash
(
  set -e
  cd ~/diffusion_baseline
  test "$(git branch --show-current)" = main
  git fetch --no-tags origin refs/heads/main:refs/remotes/origin/main
  git merge --ff-only refs/remotes/origin/main
  git log -1 --oneline
  test -f scripts/run_elf_remaining.py
)
```

这里使用明确的fetch+fast-forward merge，避开此前 `git pull` 多分支歧义问题。若同步失败，先处理错误，勿继续运行旧文件。

建议在tmux终端运行长队列。有tmux时执行 `tmux new -s elf-gpu01`；若会话已存在，执行 `tmux attach -t elf-gpu01`，确认旧队列状态，不要再启动一份。以下命令都在该终端里执行：

```bash
cd ~/diffusion_baseline
conda activate dlb-elf
export DLB_ROOT="$PWD"
export ELF_PYTHON="$CONDA_PREFIX/bin/python"
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi -i 0,1
```

调度器显式覆盖子进程的 `CUDA_VISIBLE_DEVICES`，不依赖旧GPU2变量。它自己的两张卡互不共享进程；锁只防止本脚本重复使用同一GPU，不能阻止其他人的作业。不要停止其他人的进程。质量实验可以共卡但可能变慢或OOM，正式timing要另选独占时段。

## 2. 只准备OWT前缀输入

WMT14/XSum数据和模型已经就绪，无需重新下载或执行 `prepare_elf.py data`。前缀输入需要服务器保留旧baseline的OWT预处理数据及manifest。下面命令只在原prompt文件不存在时构建；已有的必须先通过核验。

```bash
(
  set -e
  cd ~/diffusion_baseline
  conda activate dlb-bootstrap
  export DLB_ROOT="$PWD"
  if [ ! -f data/conditional/owt-c64/prompts.jsonl ]; then
    python scripts/build_conditional_prompts.py --root "$DLB_ROOT" --dataset owt
  fi
  python scripts/verify_conditional_prompts.py --root "$DLB_ROOT" --dataset owt

  conda activate dlb-elf
  python scripts/prepare_elf.py prefix
)
```

如果缺少原OWT预处理数据或核验失败，不创建临时prompt或换数据集绕过。可先执行第3–4步的 `--tasks owt wmt14 xsum` 子集，前缀输入恢复后再跑 `--tasks owt-prefix`；调度器也会把缺输入项标记为 `blocked_input`，允许其他任务继续。

## 3. 先看计划，不使用GPU

```bash
conda activate dlb-elf
cd ~/diffusion_baseline
python -u scripts/run_elf_remaining.py --gpus 0 1 --plan
```

自动扫描所有 `results/elf*/quality`，包括旧 `elf-h200-gpu2`、`elf-h200-gpu2-eos-v2` 和新目录：

- `skip`：样本与指标哈希/计数一致，且当前权重、输入、采样、batch、seed和协议匹配。也会跳过已明确记录的退化无效质量，避免反复刷随机样本；无效结果仍不可填成有效PPL。
- `evaluate`：已有完整、匹配的生成文件，只补评分。
- `generate+evaluate`：缺少可复用的完整结果。生成到新目录，不覆盖旧样本。
- `BLOCKED`：输入缺失或manifest不匹配。计划命令返回2；需处理对应输入，但其他任务可以照常排队。

1000条sanity不替代1024条正式实验；旧无EOS条件结果不复用。部分完成/损坏目录保留，必要时写入 `elf-gpu01-retry1` 等新根目录。单格生成中断不能逐条续写，因为没有保存中途随机数状态；该格重生成，其他完整格子不重跑。

## 4. 一条命令启动两张卡

```bash
python -u scripts/run_elf_remaining.py --gpus 0 1
```

**只启动这一份队列，不再在另一终端手动启动相同实验循环。** 不需要分别设置两份 `ELF_RESULTS`，也不启动torchrun。脚本会打印 `START GPU 0/1 ...`、`DONE ...` 和每30秒进度。

新质量结果写在 `results/elf-gpu01/quality/...`，已有结果继续保留在原目录。每次调度的终端详细日志及实时清单写在：

```text
results/elf-gpu01/logs/<UTC时间-进程号>/
  queue.json
  <task>__<split>__steps_<N>__seed_42__compile_0__generate.log
  <task>__<split>__steps_<N>__seed_42__compile_0__evaluate.log
```

遇到低步数空输出或不足两个GPT-2 tokens，评分器可能记录 `quality_invalid`。队列保留无效原因并继续其他配置，不删除样本、不换seed。其他异常记录 `failed`，后续格子继续；结束返回非零。不能只看到进程结束就认为所有实验有效。

tmux中按 `Ctrl-b`，再按 `d` 可退出界面，后台继续；恢复用 `tmux attach -t elf-gpu01`。直接 `Ctrl-C` 会终止本队列及它启动的子进程，保留日志/已完成结果，不停止别人的进程。之后重复第3–4步即可补剩余格子。

## 5. 查看日志和最终指标

在另一个终端执行（这是只读操作）：

```bash
cd ~/diffusion_baseline
conda activate dlb-elf
python - <<'PY'
import json
from pathlib import Path
paths = sorted(Path('results/elf-gpu01/logs').glob('*/queue.json'))
if not paths:
    raise SystemExit('尚无队列报告；检查启动终端。')
path = paths[-1]
r = json.loads(path.read_text())
print('报告:', path)
for run in r['runs']:
    c = run['cell']
    print(c['task'], c['split'], c['steps'], run['status'], run['run'])
    if run.get('reason'):
        print('  原因:', run['reason'])
    if run.get('status') == 'failed':
        print('  日志:', run.get('generate_log'), run.get('evaluate_log'))
for item in r['blocked']:
    print('BLOCKED:', item)
PY
```

想看某个配置的实时终端输出，从启动终端给出的日志目录选对应文件运行 `tail -f <具体日志文件>`；`Ctrl-C`仅退出tail，不停止队列。

全部结束后，再执行：

```bash
python scripts/inspect_elf_progress.py --root "$PWD" \
  --output "$PWD/results/elf-progress.json"
python -u scripts/run_elf_remaining.py --gpus 0 1 --plan
```

完成标准：计划为32格、0 queued、0 blocked；另看哪些是 `quality_valid`、哪些是已运行但 `quality_invalid`。若仍有排队项/失败项，先看其日志，修复后重复同一队列命令即可。只汇总 `elf-gpu01` 会漏掉复用的旧结果；以 `queue.json` 的run路径和跨根目录清单归档。OWT64步、1000条sanity和历史无EOS结果仍在总清单中，不能混入本轮主表。

实际H200推理尚需在服务器验证。本地33项CPU检查覆盖输入矩阵、结果复用、协议不匹配、损坏文件、双worker GPU绑定、无效质量继续运行和只读预览；没有本地下载权重或运行GPU实验。
