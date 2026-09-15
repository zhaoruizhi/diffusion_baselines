# ELF OWT条件4步、8步：短输出补评分

用户提供的结果：4步已生成2048条，其中2条不足64个GPT-2 tokens；8步已生成2048条，其中1条不足64个GPT-2 tokens。两格都有timing。这不是生成任务缺失或GPU报错，而是严格C64评分拒绝了短输出。仅凭CSV还不能判断是提前EOS还是T5/GPT-2重分词长度差异。

保持相同seed和生成配置，重复生成不保证解决短输出；改变seed、只替换短样本或反复抽到足长样本会引入筛选。因此这次复用原始2048条，**只重新运行评分，不训练、不重新生成、不重测timing**。

## 两种评分结果分开保存

- `strict`（原默认）：每条response必须至少64 tokens。旧4步、8步仍为 `valid=false`，本次不覆盖。
- `observed`（新增可选）：所有2048条都参与，长输出取前64 tokens，短输出取实际非空长度，不补padding、不丢样本。PPL仍使用GPT-2 Large并排除prompt loss，按总NLL/总实际有效response token数汇总；entropy是每条实际评分片段熵的均值。若有空response，仍不能产生完整主指标。

后一种结果必须标注为“response最多64个GPT-2 tokens，短输出按实际长度评分”。它并不是把原来严格64的失败变成成功。其他5个步数没有短输出，在两种长度政策下评分片段相同；如果合并展示，整张表统一标注最多64政策，并保留每格短输出计数及原结果来源。可用 `--short-response-policy observed` 不指定 `--steps` 为全部7格生成同一协议的新表。

## 服务器：只补4步和8步

一张空闲GPU足够，以下使用GPU0。与ELF1024步生成无关，只加载评分器。

```bash
cd ~/diffusion_baseline
git pull --ff-only
conda activate dlb-elf
export PYTHONDONTWRITEBYTECODE=1
mkdir -p results/elf-owt-baseline-logs
set -o pipefail

python scripts/evaluate_elf_owt_baseline.py \
  --task owt-prefix --steps 4 8 \
  --short-response-policy observed --plan

CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate_elf_owt_baseline.py \
  --task owt-prefix --steps 4 8 \
  --short-response-policy observed --batch-size 8 \
  2>&1 | tee results/elf-owt-baseline-logs/owt-prefix-s4-s8-observed.log
```

`--plan`应只列出两格，不进行GPU推理。新结果默认写入独立的 `results/elf-owt-baseline-eval-owt-prefix-observed`，不需要设置ELF_RESULTS。中断后重复同一条评分命令即可，已核验的新指标会跳过。

```bash
cat results/elf-owt-baseline-eval-owt-prefix-observed/summary-steps-4-8.csv
```

预期：两格sample_count均2048；如所有短输出均非空且有有效评分目标，valid=True；short_count仍为2和1。short_count不会被“修成0”。若依然失败，保留日志，不更换seed或删样本。

## 可选：查看原短输出的真实长度与文本

下面只用CPU tokenizer，不加载GPU模型。读取旧指标定位样本，并校验保存的样本哈希。

```bash
python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from elf_common import asset, sha256
from transformers import AutoTokenizer
root = Path.cwd()
tok = AutoTokenizer.from_pretrained(str(asset(root, 'gpt2')), local_files_only=True)
for steps in (4, 8):
    mp = root / f'results/elf-owt-baseline-eval-owt-prefix/steps_{steps}/metrics.json'
    meta = json.loads(mp.read_text())
    samples = Path(meta['source_run']) / 'samples.jsonl'
    assert sha256(samples) == meta['samples_sha256']
    selected = set(meta['short_sample_ids'])
    for line in samples.open():
        row = json.loads(line)
        if row['id'] in selected:
            print(json.dumps({
                'steps': steps, 'id': row['id'],
                'prompt_id': row['prompt_id'], 'completion_id': row['completion_id'],
                'gpt2_tokens': len(tok(row['generated'], add_special_tokens=False)['input_ids']),
                'text': row['generated'],
            }, ensure_ascii=False))
PY
```

由于旧生成文件已经裁剪EOS之后的内容，本次不能从现有文件恢复完整原生输出尾部；诊断只能说明已保存文本的情况。无需为补这两个评分数字再生成整批样本。
