# ELF 公开权重：OWT 无条件与 C64 条件续写

本页替代文档17的重训方案。用户已明确：只做 OWT，保留公开 ELF-OWT 权重及 T5 tokenizer，不训练。WMT14/XSum 既有结果保留，本次不再运行。

## 固定实验设置

| 设置 | OWT 无条件 | OWT 条件续写 |
|---|---|---|
| 权重 / 生成 tokenizer | 公开 ELF-OWT / T5-small | 同一份公开权重 / T5-small |
| 步数 | 1、2、4、8、16、32、1024 | 相同 |
| seed / compile | 42 / 关闭 | 相同 |
| 内部生成长度 | 1024 个 T5 位置 | 1024 个 T5 位置 |
| prompt | 无 | 旧 baseline 的同一组 1024 个 prompt；每条原始 64 个 GPT-2 token 解码后给 T5 |
| 自由生成位置 | 1024 | 1024 减实际 T5 prompt 长度；保留完整 prompt 文本，不截成64个T5 token |
| 保存 completion 数 | 1024 | 2048 = 1024×1 + 前256个prompt各额外4条 |
| PPL / entropy 统计条数 | 全部1024条 | **全部2048条，与实际旧评测代码一致** |
| 评分范围 | GPT-2重分词后，最多1024 tokens | response前64个GPT-2 tokens；PPL附加原始64-token prompt但排除prompt loss |
| PPL 模型 / 汇总 | 固定版本 GPT-2 Large；exp(总NLL/总有效token数) | 同左 |
| entropy | 每条评分片段的unigram entropy，使用自然对数，再取样本均值 | 同左 |
| Self-BLEU | 不计算 | 前256个prompt的5条completion，调用旧baseline实现 |

T5生成tokenizer是用户接受的差异。表中不能写“ELF原生GPT-2词表”或“固定960个T5自由位置”。保留官方ELF的EOS解码政策，entropy是GPT-2重分词后的文本entropy；旧表中保留特殊token的原生ID entropy仍需注明这一差异，不能声称逐项完全相同。

**重要修正：** 旧 `evaluation/conditional_evaluate.py` 实际把2048条都送入PPL/entropy。之前ELF只用completion 0的1024条做主PPL、分别重编码prefix/suffix再直接拼接token IDs，因此旧ELF条件指标不能直接充当这次的对齐结果。新入口使用旧 `compute_conditional_gen_ppl`、原始prompt IDs及其“拼接IDs后解码、重新分词、屏蔽prompt loss”的实现；2048条全部参与。前256个prompt在这个旧协议中有5条，其他prompt有1条，明确保留这种权重安排。

1024/2048是**样本条数**，不是单条response长度。条件PPL先取64-token response，不会把960个自由位置或2048条completion全部拼给GPT-2。评分函数保持1024的单条上下文上限。

## 复用与补跑

- 已生成的两项任务×7步，共14格样本可以复用，前提是来源、哈希、输入、采样设置和数量通过现有队列验证。
- 无条件PPL算法未改；重评分是为了统一生成一份可审计的新汇总，不需要重新运行1024步ELF生成。
- 条件7格需要重评分，不能复制原来的1024条主CPPL。
- 新指标独立保存，不覆盖原来的 `metrics.json`、samples或timing。
- 如果条件输出不足64个GPT-2 tokens，新入口记录所有短样本ID、实际长度、`valid=false`，不给出冒充完整C64的主指标。不会用padding补齐、静默删除或重采样。无条件至少需要2个tokens以产生next-token目标；较短但合法的无条件文本按实际长度评分，长度诊断另报。
- 计时计算范围未改，仅修正评分，因此已完成的同协议OWT timing可以保留。需要新测时见最后一步。

## 1. 服务器同步及核验

以下步骤只在服务器运行。先进入持久终端（例如你已有的tmux会话）。

```bash
cd ~/diffusion_baseline
git pull --ff-only
conda activate dlb-elf
export DLB_ROOT="$PWD"
export ELF_PYTHON="$CONDA_PREFIX/bin/python"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p results/elf-owt-baseline-logs

python scripts/run_elf_remaining.py \
  --tasks owt owt-prefix --gpus 0 1 --plan
```

预期是14格、0 queued、0 blocked。若有生成或评分缺失，可运行下面队列；已核验完成的格子会跳过。**它先保证旧样本产物完整，不是本次新评分入口。**

```bash
python scripts/run_elf_remaining.py \
  --tasks owt owt-prefix --gpus 0 1
```

如果报告prefix输入缺失且旧 `data/conditional/owt-c64/prompts.jsonl` 已在，执行 `python scripts/prepare_elf.py prefix`，再核验。不要重建或更换旧baseline的prompt集合来填补缺失。

## 2. GPU0：重评无条件7格

单独终端执行；脚本只加载GPT-2 Large评分，不加载ELF生成模型。

```bash
cd ~/diffusion_baseline
conda activate dlb-elf
export PYTHONDONTWRITEBYTECODE=1
mkdir -p results/elf-owt-baseline-logs
set -o pipefail

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_elf_owt_baseline.py \
  --task owt --plan

CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate_elf_owt_baseline.py \
  --task owt --batch-size 8 \
  2>&1 | tee results/elf-owt-baseline-logs/owt-evaluate.log
```

## 3. GPU1：重评条件7格

可与步骤2同时执行；不要同时在这两张卡测timing。

```bash
cd ~/diffusion_baseline
conda activate dlb-elf
export PYTHONDONTWRITEBYTECODE=1
mkdir -p results/elf-owt-baseline-logs
set -o pipefail

CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_elf_owt_baseline.py \
  --task owt-prefix --plan

CUDA_VISIBLE_DEVICES=1 python -u scripts/evaluate_elf_owt_baseline.py \
  --task owt-prefix --batch-size 8 \
  2>&1 | tee results/elf-owt-baseline-logs/owt-prefix-evaluate.log
```

新入口自动找各步数匹配的已生成结果，允许结果分散在多个旧目录；无需设置 `ELF_RESULTS`。逐格输出JSON进度。中断后重复同一条命令，按输入、评分代码、资产等哈希复用已评分格子。短输出会记录后继续后面的步数，全部处理后退出码2表示存在无效格子，需查看原因，不表示必须重新生成。

## 4. 查看新表格

```bash
python - <<'PY'
import csv
from pathlib import Path
for task in ('owt', 'owt-prefix'):
    path = Path('results') / f'elf-owt-baseline-eval-{task}' / 'summary.csv'
    print(f'\n{task}: {path}')
    if not path.exists():
        print('尚未汇总；先查看对应日志及 steps_*/metrics.json')
        continue
    print('steps | N | valid | PPL/CPPL | entropy | short')
    for r in csv.DictReader(path.open()):
        print(' | '.join(r[k] for k in ('steps', 'sample_count', 'valid', 'ppl', 'entropy', 'short_count')))
PY
```

预期无条件每格N=1024，条件每格N=2048。指标文件包含原样本路径、SHA256、评分器代码哈希和token长度诊断。新文件名：

- `results/elf-owt-baseline-eval-owt/summary.csv`
- `results/elf-owt-baseline-eval-owt-prefix/summary.csv`
- 各目录的 `steps_N/metrics.json`

MAUVE不在此次PPL/entropy必跑范围内；本次条件额外输出旧协议的grouped Self-BLEU。

## 5. timing：质量评测结束、GPU空闲后

只改评分不影响已测的生成耗时，本次无需因此强制重测。如果需要一份新测量，使用同一张空闲H200串行测14格：

```bash
cd ~/diffusion_baseline
conda activate dlb-elf
export DLB_ROOT="$PWD"
export ELF_PYTHON="$CONDA_PREFIX/bin/python"
bash scripts/run_elf_timing_group.sh 0 owt --plan
bash scripts/run_elf_timing_group.sh 0 owt
```

batch=1、5次warmup、32次CUDA同步计时、compile关闭。无条件主列 `sampler_seconds`；条件主列 `encoder_plus_sampler_seconds`，缓存条件的 `sampler_cached_condition_seconds`只作附加列。这些列就是对应测量范围的 `generation_seconds_per_sample`，不是生成阶段含IO的总时长除以样本数。条件耗时包含完整1024位置的ELF采样及必要T5条件编码，不是只生成64位置的时间。

新的timing目录会在终端打印为 `results/elf-timing-gpu0-时间戳-PID/summary.csv`。脚本在每格前检查占用，但不能替其他用户预约GPU。质量比较同时附PPL和entropy，不将低熵退化输出当成同质量加速。
