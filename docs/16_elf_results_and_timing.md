# ELF：32格质量结果归档与正式timing步骤

本页依据用户提供的服务器终端输出整理。日志中主队列为 **32 cells、0 queued、0 blocked input groups**，所选32格均为quality_valid。这里核对的是用户提供的清单，不是本地重新执行GPU实验。用户现已授权启动timing，本页替代此前“timing暂缓”的安排。

终端附件SHA256：`8b4ac0adeeded1801ab234d58c9b05b64c4ff078deccf34d0c61b2af4f744e3a`。

## 完成状态

| 实验 | 完成配置数 | 步数 | 每格规模 |
|---|---:|---|---|
| OWT无条件 | 7 | 1/2/4/8/16/32/1024 | 1024条 |
| OWT前缀续写 | 7 | 1/2/4/8/16/32/1024 | 1024 prompts、2048 completions |
| WMT14 validation | 7 | 1/2/4/8/16/32/64 | 3000条 |
| XSum validation | 7 | 1/2/4/8/16/32/64 | 11332条 |
| WMT14 test | 2 | 32/64 | 3003条 |
| XSum test | 2 | 32/64 | 11334条 |

清单共41个历史目录，除主表32格外，还有OWT64步1格、3格sanity、4格旧无EOS WMT14，以及1格旧无EOS XSum未完成生成。后两类不纳入正式结果。**不需要重跑任何主表质量配置。**

旧WMT14的BLEU 12.3720/13.4014/14.0021/14.1667对应错误的无EOS输入路径，已由修复版覆盖实验需求；旧XSum `elf-h200-gpu2/quality/xsum/validation/steps_8` 的incomplete_generation也不需要恢复。它们保留用于诊断。`quality_valid`只说明当前文件/指标通过核验，不保证协议正确或生成文本优质。OWT历史记录的condition_policy=null不等同于条件任务遗漏EOS：OWT本身是无条件生成。

## OWT质量结果

无条件指标为Gen.PPL；前缀指标为prompt loss排除后的conditional PPL，两列不能直接比数值大小。Entropy单位为nats，采用当前ELF的GPT-2文本重编码评分政策。

| 步数 | 无条件PPL | 无条件entropy | 前缀conditional PPL | 前缀entropy |
|---:|---:|---:|---:|---:|
| 1 | 2.9657 | 1.9938 | 15.6772 | 1.5581 |
| 2 | 29.3739 | 4.5456 | 109.1312 | 3.3800 |
| 4 | 60.1296 | 5.0841 | 279.8319 | 3.7205 |
| 8 | 70.9471 | 5.2528 | 195.1630 | 3.7321 |
| 16 | 32.1761 | 5.1609 | 77.8110 | 3.7182 |
| 32 | 24.3517 | 5.1651 | 61.0454 | 3.7205 |
| 1024 | 7.7324 | 4.7420 | 54.4605 | 3.6359 |

辅助OWT64步：PPL=19.1247、entropy=5.0774；不加入七步主表。

单步无条件PPL=2.9657但entropy=1.9938，前缀单步PPL=15.6772但entropy=1.5581，存在低多样性/重复输出的嫌疑。1024步无条件PPL=7.7324、entropy=4.7420，较8–32步的entropy也下降。单凭低PPL不能判定全面优于其他方法；需看样本、长度和Self-BLEU。当前终端没有显示grouped_self_bleu、reference_conditional_ppl及长度诊断数值，不能编造这些结果，后文命令可从已保存metrics读取。无需为读取这些字段重生成。

公开ELF生成canvas为1024 T5 tokens，旧OWT方法为1024 GPT-2 tokens；EOS和entropy处理仍有差异。前缀ELF协议为c64_text_t5_v1，属于公开无条件权重的零样本条件投影，并非任务训练的WMT14/XSum模型。表中各项不得省略这些协议差异后宣称完全公平的绝对排名。

### 为什么与旧OWT baseline表差距很大

重新核对[论文附录D.4、Table 6](https://arxiv.org/html/2605.10938v1#A4.SS4)，可以直接对照本次核心步数，而不只比较其他方法：

| 步数 | 论文ELF PPL（均值±SE） | 本次PPL | 论文ELF entropy（均值±SE） | 本次entropy |
|---:|---:|---:|---:|---:|
| 8 | 67.32 ± 2.25 | 70.9471 | 5.14 ± 0.085 | 5.2528 |
| 16 | 33.66 ± 1.09 | 32.1761 | 5.16 ± 0.026 | 5.1609 |
| 32 | 24.08 ± 0.16 | 24.3517 | 5.15 ± 0.002 | 5.1651 |

论文是seeds 0–5的6次评测统计，本次是seed42的一次1024样本评测；接近只能支持“原生配置复现量级合理”，不是统计等价检验，也不证明所有旧baseline与ELF处理完全相同。1/2/4/1024是本项目扩展网格，没有Table 6对应参考点；尤其不能用32步复现接近，替1024步低PPL作质量保证。

已确认的三个区别是生成词表/实际长度、首EOS及特殊token处理，以及采样配置。当前ELF的entropy也是GPT-2重编码后的逐样本自然对数熵，**不是直接拿T5词表熵与GPT-2熵比较**；真正未统一的是进入统计的token序列。PPL则与旧路径调用同一compute_gen_ppl函数，使用相同锁定GPT-2 Large。因此目前没有依据把差距归因为PPL公式或评分模型用错，但输入处理仍会影响评分，影响方向与大小不能凭代码猜定。

ELF使用SDE、logit-normal schedule、SC-CFG=3，并随步数改变gamma；旧方法各自使用不同采样器。论文[引导消融](https://arxiv.org/html/2605.10938v1#S4.SS1)展示了PPL与entropy的权衡。这能解释为何引导是需要核查的因素，但尚未通过本地消融测出它对旧表差距的贡献。旧表8步FLM约449.15/5.21，而ELF为70.95/5.25，熵相近仍有PPL差距，不能将全部差距归因于低熵。反过来，旧表FLM2步7.74/0.89也说明低PPL与低熵可以同时出现，单独按PPL排名不可靠。

下一步诊断顺序：先用第5节读取原样本及长度/重复度；再固定共同文本处理政策，对旧方法和ELF另存一套重评分；必要时做同一步数SC-CFG对照。所有短/空输出保留，不为得到更好的分数删除样本。原始ELF样本已清除首EOS之后的tokens，无法恢复该部分；若要求评估完整未截断canvas，需要先实现保存完整输出并另行生成，不能靠重评分恢复。现有timing仍可记录为原生配置耗时，不能在质量与长度政策未对齐时当作严格matched-quality加速证据。

## WMT14和XSum validation

| 步数 | WMT14 BLEU | XSum ROUGE-1 | ROUGE-2 | ROUGE-L |
|---:|---:|---:|---:|---:|
| 1 | 11.5059 | 23.7112 | 5.7420 | 19.4264 |
| 2 | 18.2014 | 29.4491 | 8.2950 | 23.3782 |
| 4 | 22.1798 | 32.6299 | 10.0430 | 25.4926 |
| 8 | 24.1705 | 34.3705 | 11.1567 | 26.6864 |
| 16 | 25.3318 | 35.3485 | 11.7578 | 27.2921 |
| 32 | 26.1424 | 35.8402 | 12.1714 | 27.6562 |
| 64 | 26.6741 | 36.0270 | 12.3522 | 27.8165 |

## WMT14和XSum test

| 步数 | WMT14 BLEU | XSum ROUGE-1 | ROUGE-2 | ROUGE-L |
|---:|---:|---:|---:|---:|
| 32 | 25.8662 | 35.6955 | 12.0942 | 27.5345 |
| 64 | 26.3402 | 35.9502 | 12.2767 | 27.7115 |

两项任务随步数增加整体改善，test与validation接近。32/64两个候选保持此前冻结的设置，下一步测它们的耗时，不根据test分数改采样参数。

## 1. 同步代码

以下操作全部在服务器执行，现有模型、输入和质量结果不变：

```bash
(
  set -e
  cd ~/diffusion_baseline
  test "$(git branch --show-current)" = main
  git fetch --no-tags origin refs/heads/main:refs/remotes/origin/main
  git merge --ff-only refs/remotes/origin/main
  test -f scripts/run_elf_timing_group.sh
  git log -1 --oneline
)
```

同步失败时不要继续执行旧脚本。停止自己此前的质量队列后再开始timing，勿终止他人的作业。建议使用两个tmux会话，以免SSH断开中断计时。

## 2. GPU 0：OWT无条件与前缀timing

第一个终端新建 `tmux new -s elf-time0`，在会话中执行：

```bash
cd ~/diffusion_baseline
conda activate dlb-elf
export DLB_ROOT="$PWD"
export ELF_PYTHON="$CONDA_PREFIX/bin/python"
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi -i 0

bash scripts/run_elf_timing_group.sh 0 owt --plan
bash scripts/run_elf_timing_group.sh 0 owt
```

共14格：两项任务各1/2/4/8/16/32/1024步。前缀固定第0条输入。无条件生成完整1024 T5-token canvas，前缀也按ELF完整canvas采样；后者评分只取最多64 GPT-2 tokens，并不意味着推理只生成64 tokens。timing按实际计算量报告，不能用64直接除得“完整生成速度”。

## 3. GPU 1：WMT14/XSum timing

第二个终端新建 `tmux new -s elf-time1`，在会话中执行：

```bash
cd ~/diffusion_baseline
conda activate dlb-elf
export DLB_ROOT="$PWD"
export ELF_PYTHON="$CONDA_PREFIX/bin/python"
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi -i 1

bash scripts/run_elf_timing_group.sh 1 seq2seq --plan
bash scripts/run_elf_timing_group.sh 1 seq2seq
```

共18格：两项任务各validation的1/2/4/8/16/32/64步，以及test的32/64步。对每个split固定第0条输入，不是对全体3000/11332等样本重新推理。两个终端可同时运行，每张卡只有自己的一个采样进程。如果tmux会话已经存在，用 `tmux attach -t elf-time0` 或 `elf-time1` 恢复，不要重复启动同一组。

每次调用自动创建独立目录，终端首先打印 `TIMING OUTPUT: ...`，例如：

```text
results/elf-timing-gpu0-<UTC时间>-<PID>/
results/elf-timing-gpu1-<UTC时间>-<PID>/
```

每个目录含 `timing/`、`logs/`、`environment/`；整组成功后写 `summary.csv` 并打印 `FINISHED`。每格开始前检查该卡是否有compute进程，有则停止；该检查不是GPU预约，运行期间仍需保证同伴不会加入。前后GPU快照保存在environment中。GPU0/1用UUID绑定，避免继承旧的GPU2环境变量；当前脚本也强制覆盖旧seed/compile/prompt/steps设置。

脚本遇到错误会停止本组，日志的错误码不会被tee吞掉；另一个终端不受影响。每次重新调用整组入口会新测该组全部timing，不是自动续跑。若只失败一格，不要重启整组，按第6节只补该格。旧的质量队列支持复用，与本timing入口的行为不同。

## 4. timing口径与汇总

固定seed=42、batch=1、compile关闭、官方精度/采样参数、5次warmup、32次同步重复。沿用现有 `dlb.timing.benchmark`，每次计时前后CUDA synchronize，记录均值、median、population std及全部32次raw durations。

- OWT无条件主列：`sampler_seconds`。
- 条件任务主列：`encoder_plus_sampler_seconds`，包含T5条件编码和采样。
- 条件任务补充列：`sampler_cached_condition_seconds`，复用预先编码的条件，便于区分编码开销。
- 均不含权重加载、CPU tokenizer、字符串解码、结果IO和指标计算；包含噪声/时间表、去噪及最后的神经解码。不能将其称为完整请求端到端耗时。

条件任务测的是固定输入重复32次，不是全集平均延迟。两种条件计时模式分别测量，二者均值之差不是单独精确测得的编码耗时。后续若要报告输入分布上的平均延迟，应预先固定多条输入并对所有方法使用相同列表，另行记录；不要把本轮固定prompt计时当成全集均值。

本轮可为ELF建立速度曲线。与自己的方法/旧baseline比较speedup时，必须使用同型号GPU、相同batch和计时边界、相同prompt及可比的生成长度；旧表若来自不同硬件，不能直接相除。步数也不等于NFE：当前OWT约N+1次ELF forward，WMT14/XSum为2N+1次，再计条件编码。报告应保留该字段。

两个终端结束后运行以下只读命令，检查各GPU最新一组结果。它明确打印所选根目录；如果此前重启过，应按日志确认该目录是想归档的一组，不混合不同轮次选取最快数字。

```bash
cd ~/diffusion_baseline
conda activate dlb-elf
python - <<'PY'
import json
from pathlib import Path

for gpu, expected in ((0, 14), (1, 18)):
    roots = sorted(Path('results').glob(f'elf-timing-gpu{gpu}-*'),
                   key=lambda p: p.stat().st_mtime)
    if not roots:
        print(f'GPU {gpu}: 尚无timing目录')
        continue
    root = roots[-1]
    paths = list(root.glob('timing/**/timing.json'))
    print(f'\nGPU {gpu}: {root}; timing文件={len(paths)}/{expected}')
    records = [json.loads(p.read_text()) for p in paths]
    records.sort(key=lambda r: (r['task'], (r.get('input') or {}).get('split', ''), r['steps']))
    for r in records:
        split = (r.get('input') or {}).get('split', 'c64-text-t5' if r['task'] == 'owt-prefix' else 'unconditional')
        for name, value in r['results'].items():
            assert value['batch_size'] == 1 and value['warmups'] == 5 and value['repeats'] == 32
            assert len(value['raw_durations_seconds']) == 32
            mean, sd = value['seconds_per_sample'], value['standard_deviation_seconds']
            print(f"{r['task']:10s} {split:12s} s={r['steps']:4d} {name:26s} "
                  f"{mean:.6f} ± {sd:.6f} s/sample")
    print('CSV:', root / 'summary.csv', 'exists=', (root / 'summary.csv').exists())
PY
```

文件计数不是独占或低抖动的证明。出现较大std/mean或GPU快照有其他作业时，先排查共享负载/时钟波动，固定相同条件复测；不要只选择更快的一次。CSV仅为各组timing清单，质量仍引用本页32格原目录。

## 5. 读取现有OWT诊断，不重新生成

下面打印两项任务的1/1024步已有指标与前三条主completion片段。确认短文本/重复问题需结合这些输出，不能仅凭PPL断言原因。

```bash
cd ~/diffusion_baseline
python - <<'PY'
import json
from pathlib import Path

for task in ('owt', 'owt-prefix'):
    for steps in (1, 1024):
        base = 'elf-h200-gpu2' if task == 'owt' and steps == 1 else 'elf-gpu01'
        split = 'unconditional' if task == 'owt' else 'c64-text-t5'
        path = Path('results') / base / 'quality' / task / split / f'steps_{steps}' / 'seed_42/compile_0'
        print('\nRUN:', path)
        m = json.loads((path / 'metrics.json').read_text())['metrics']
        print(json.dumps(m, indent=2, ensure_ascii=False))
        count = 0
        with (path / 'samples.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                if row['completion_id'] != 0:
                    continue
                print('prompt_id=', row['prompt_id'], 'text=', repr(row['generated'][:800]))
                count += 1
                if count == 3:
                    break
PY
```

## 6. 只补一格失败timing

先确认失败原因已消除。例如只补GPU1的XSum test64步，使用新目录：

```bash
(
  set -euo pipefail
  cd ~/diffusion_baseline
  conda activate dlb-elf
  export DLB_ROOT="$PWD" ELF_PYTHON="$CONDA_PREFIX/bin/python"
  export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
  nvidia-smi -i 1
  export ELF_RESULTS="$PWD/results/elf-timing-retry-$(date -u +%Y%m%dT%H%M%S)-$$"
  export ELF_STEPS=64 ELF_SPLIT=test ELF_SEED=42 ELF_COMPILE=0 ELF_TIMING_PROMPT=0
  bash scripts/run_elf_suite.sh timing xsum
  python scripts/summarize_elf.py --results "$ELF_RESULTS" --output "$ELF_RESULTS/summary.csv"
)
```

该例不会自动判断设备独占，执行前确认GPU1已空闲；其他失败格按原来的task/split/steps修改，不改变采样参数。保留第一次失败日志，并在归档中注明补测路径。不要通过重新运行整组入口覆盖或挑选已有timing。
