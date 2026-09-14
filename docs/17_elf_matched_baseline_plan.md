# ELF与LM1B/OWT旧baseline严格对齐方案

本页是2026-09-14按用户最新要求制定的设计方案，不是已实现的训练/运行入口。此前ELF公开权重的质量与timing结果保留为原生协议结果，不再称为已经补齐旧baseline的LM1B/OWT四类任务。WMT14/XSum保持独立任务，不代替LM1B/OWT前缀续写。

## 已确认事实与需核对的历史设置

1. 官方[模型列表](https://huggingface.co/embedded-language-flows/models)当前列出OWT、WMT14 De-En、XSum等权重，未列出LM1B权重。该结论限于已核查的官方发布，不声称全网不存在任何第三方LM1B模型。
2. 本项目的ELF锁只绑定OWT/WMT14/XSum公开权重；OWT使用T5词表。不能把OWT模型生成128个tokens后称为LM1B模型，也不能只更换tokenizer加载原输出层权重。
3. 原有数据锁规定LM1B用bert-base-uncased、长度128；OWT用GPT-2、长度1024。这里的token是各数据集的生成端native token。
4. 当前本地 `configs/conditional.yaml`、`src/dlb/conditional_prompts.py` 和 `evaluation/conditional_evaluate.py` 都把response评分长度固定为64。用户已确认服务器与本地代码相同，并要求按本地代码执行，因此以实际旧代码为准，不建立64+512新协议。
5. “1024 prompts、2048 completions”是数量：1024个输入各生成1条，前256个输入各增加4条，共1024+256×4=2048条。它与每条prompt/response的token长度独立；旧C64设计本身也是这个数量安排。

用户最初提到OWT response=512；在核查并获得“按本地代码，服务器相同”的回复后，该理解已纠正。下面完整保留旧代码的64-token评分范围和1024内部canvas。

## 目标矩阵

| 任务 | 权重训练数据 | 生成端tokenizer | 每条prompt长度 | 自由生成位置数 | 评分文本/response位置数 | 实际推理canvas |
|---|---|---|---:|---:|---:|---:|
| LM1B unconditional | LM1B train | bert-base-uncased | 0 | 128 | 128（沿用padding政策） | 128 |
| OWT unconditional | OWT train | GPT-2 | 0 | 1024 | 1024 | 1024 |
| LM1B prefix | 同一LM1B权重 | bert-base-uncased | 64 | 64 | 64 | 128 |
| OWT prefix | 同一OWT权重 | GPT-2 | 64 | 960 | 64 | 1024 |

四类任务统一steps=1/2/4/8/16/32/1024、sampling seed=42、质量主样本1024条。若保留旧五路多样性评测，条件任务每格保存2048条，其中completion0的1024条用于主CPPL/entropy，前256个prompt的5条用于grouped Self-BLEU。不是把2048条拼成一条文本。

两份数据集权重分别用于无条件与零样本前缀任务，不额外做条件微调。若加入条件微调，应作为新的训练设置单列，不能与旧zero-shot prefix混用。

**OWT条件实验是“64-token prefix、960个自由生成位置、只评分response前64 tokens、内部计算1024位置”。** 这是旧代码的真实协议。无需改成576长度，也不能把1024-canvas采样耗时称为只计算64-token response的耗时。位置数与EOS后解码文本的实际长度仍有区别，需保持原始ID记录。

## GPT-2 Large如何评分

- 数据集样本数N与单条上下文长度L独立。2048条可以分batch评测，GPT-2的位置上限约束每条序列，不约束文件共有多少条。
- 旧OWT条件评分取64-token prefix与response前64个GPT-2 tokens，输入约128个位置，低于1024上限。attention mask区分padding，loss mask排除prompt，只统计response next-token损失；内部生成1024位置不等于给评分器输入2048个位置。即使假设另测64+512，其576位置也不超过1024，但那不是当前协议。
- 当前ELF条件评分实际把每条response截为最多64个GPT-2 tokens再与各自prefix拼接，并在长度超过1024时报错；没有把2048条接成长串。因此现有结果并非由“2048超位置上限”造成。
- LM1B原生tokenizer为BERT，64+64 BERT tokens不能直接断言等于128 GPT-2 tokens。使用共同的解码与GPT-2重分词策略，逐条检查重编码上下文长度并保存诊断；不静默丢掉长/短样本。
- PPL统一调用锁定GPT-2 Large，采用exp(总有效response NLL/总有效response token数)。无条件计算生成文本的有效next-token损失。不要平均每条PPL，也不要把prompt自身计入条件PPL。
- 严格旧表entropy使用其既有native-ID政策：LM1B只排除明确padding，OWT保留旧约定的特殊token；同一数据集各方法使用同一函数和token范围。若另做GPT-2重编码entropy，作为独立新列为所有方法重算，不覆盖旧native entropy。
- EOS/短输出处理沿用旧代码对应数据集的既有政策，保存未裁剪native IDs和单独的decoded text/EOS位置，不像当前ELF那样在保存前丢失EOS之后tokens。不同评分政策写入不同目录与协议标识，不覆盖c64_zs_v1已有结果。

## ELF权重与训练方案

要同时满足LM1B训练来源、native词表和长度，不能沿用当前公开T5-OWT权重当作严格对齐模型。主方案是训练两份明确标注的 `ELF-matched` 研究版本，同时保留原版公开权重结果作为参考。

实现方向：保留ELF的连续嵌入flow matching、自条件与共享最终解码方法，提供与native tokenizer对应的encoder接口和输出词表。可采用LM1B的BERT encoder与OWT的GPT-2 encoder作为待验证方案，冻结或训练encoder的政策在实验前统一记录。GPT-2在这里作为训练表示encoder，与独立GPT-2 Large质量评分器是不同角色；这会改变原版T5表示，必须标注，不能冒称作者原版checkpoint。

替换encoder不是只改名字：latent维度、归一化统计、输出分类头、attention mask、自条件拼接维度及prefix-only编码都要适配；当前脚本中的T5特定处理不能复用后声称等价。另一种研究设计是使用native词表的可学习表示，但也需独立训练与验证。最终encoder设计和预训练来源要在训练前冻结，不按test分数选择。

服务器训练流程：

1. 复用旧数据锁与已预处理数据，固定LM1B/OWT训练及held-out划分；不把测试数据用于encoder统计、训练或超参选择。
2. 先做小规模训练和采样验收：loss有限、词表合法、固定长度、prefix逐步clamp不变、输出不读取reference、checkpoint恢复后的随机数一致。
3. 用各训练集校准latent归一化统计；训练配置以ELF方法为起点，微batch先在H200上profile，再通过梯度累积固定有效batch。训练token预算与EMA选择写进manifest。现有公开旧权重的训练预算不同，未获得证据时不宣称所有模型同等训练算力。
4. GPU0训练LM1B版本，GPU1训练OWT版本；这只是排卡方案，正式训练命令待上述实现验收后提供。训练阶段不同时测timing。
5. 使用validation做预先限定的采样/引导选择，冻结后生成四类任务各7步，共28个主质量配置；保留额外多样性completion和所有退化输出。
6. 全部质量协议核验通过后，独占H200测对应28个timing配置。自己的方法与其他baseline按相同推理canvas、输入列表和计时边界重测需要重测的格子。

此路线能够对齐任务/生成与评分协议，但改变了ELF的表示配置；它不能保证复现原版ELF的24.1 PPL，也不保证与旧模型具有完全相同的预训练或训练预算。若必须坚持作者发布权重不变，则LM1B只能标记未提供，T5与GPT-2长度差异也必须保留，无法同时完成上述严格四行目标。

## 哪些结果复用，哪些重做

| 已有结果 | 处理 |
|---|---|
| 公开ELF OWT无条件PPL/entropy | 保留为native参考；严格native词表版本需要新权重并重新生成/评分 |
| 公开ELF OWT前缀64评分结果 | 数量和64评分长度与旧任务相近，但T5重编码prefix、生成native词表和EOS政策不一致；保留为c64_text_t5_v1参考，对齐版本需新权重及新生成 |
| 公开ELF两项OWT timing | 保留为T5原生系统耗时，不放入严格native长度主表 |
| ELF LM1B无条件及前缀 | 尚无对应权重/结果，需要上述训练路线，不能由OWT权重替代 |
| 旧baseline LM1B/OWT无条件 | 协议和硬件符合时可保留；不是因为加入ELF就全部重跑 |
| 旧baseline LM1B 64+64 | manifest确认相同则保留 |
| 旧baseline OWT条件 | 保留c64_zs_v1现有64评分结果，内部canvas保持1024；不因为此前对512的误解而重生成或改评分。计时仅在硬件/范围不匹配时补测 |
| WMT14/XSum质量与timing | 独立任务，全部保留；与上述四行无替代关系 |

## timing验收条件

继续使用batch=1、5 warmups、32同步重复、compile关闭、明确精度和固定seed；固定相同prompt IDs并记录真实长度。对条件任务以包含必要条件编码的耗时为主，另列缓存条件采样耗时。固定输入重复测量不称为全集平均延迟。

记录 `native_tokenizer`、训练数据锁、实际推理canvas、prefix长度、生成响应位置数、评分响应长度、EOS策略、输出实际长度、NFE、GPU型号/UUID和编码范围。不能只给一个generation_seconds_per_sample却省略这些定义。

如果模型按固定canvas算完再处理EOS，即使文本较短也必须计入全部计算时间。速度比较同时附PPL、entropy、短/空输出率及重复度，不将低熵退化配置当作同质量加速。CUDA同步和相同GPU不足以弥补不同计算长度。

## 当前可以运行的只读核验

先核对服务器实际协议，下面命令不改文件、不使用GPU：

```bash
cd ~/diffusion_baseline
conda activate dlb-bootstrap
python - <<'PY'
import json
from pathlib import Path

print('CURRENT CONFIG:')
print(Path('configs/conditional.yaml').read_text())
for path in sorted(Path('data/manifests').glob('conditional-*.json')):
    print('\nPROMPT MANIFEST:', path)
    record = json.loads(path.read_text())
    for key in ('protocol','dataset','tokenizer_id','prompt_count','prefix_length',
                'evaluation_continuation_length','model_length','prompt_file_sha256'):
        print(key, '=', record.get(key))
PY
```

用户已确认本地/服务器协议一致，以上命令用于归档证明，不是等待另一版512结果。不要把conditional.yaml中的64改成512。实现工作仅新增ELF-matched训练与baseline适配路径，复用旧prompt选择、切片和评分合约；现有公开ELF入口继续保留原生身份。训练命令及checkpoint尚未实现或生成，不能把本设计文档当成一条已经可运行的训练指令。
