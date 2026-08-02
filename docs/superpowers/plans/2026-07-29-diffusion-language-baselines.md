# Diffusion Language Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个可搬到 Linux GPU 服务器直接运行的 LM1B/OWT 扩散语言模型 baseline 工程，包含锁定源码、数据与 checkpoint、独立 Conda 环境、统一生成/计时/评测流程和中文操作手册。

**Architecture:** 8 个上游仓库作为只读依赖放入 `upstreams/`，11 个逻辑模型通过本工程的注册表和薄适配器调用。所有模型输出统一 JSONL；模型无关的评测包负责 GPT-2 Large Gen. PPL、unigram entropy、Self-BLEU 和 CUDA 同步计时。资源 manifest、运行 metadata 与结果 schema 形成端到端可追溯链。

**Tech Stack:** Bash、Python 3.10+、PyTorch、Transformers、Hugging Face Datasets/Hub、Hydra、PyYAML、pydantic、pytest、Conda/Mamba、Slurm。

## Global Constraints

- 数据集固定为 LM1B（`bert-base-uncased`，长度 128）与 OWT（`gpt2`，长度 1024）。
- OWT 原始顺序的最后 100,000 篇为 validation，其余为 train。
- many-step 网格严格为 `1,2,4,8,16,32,1024`。
- few-step 网格严格为 `1,2,4,8,16,32`。
- 每个受支持配置必须生成恰好 1,024 个有效样本。
- 离散基线默认 ancestral sampling、temperature 1.0；FLM 使用 Euler；FMLM 使用论文 gamma-sampling；RDLM 使用官方 SDE sampler。
- 主计时指标使用 batch size 1，5 个 warm-up 样本和 32 个计时样本；checkpoint 加载、decode 与落盘不计时。
- 上游源码只读；任何必要修复保存为可审计 patch。
- 公开资源缺失的组合必须标记 `unsupported`，不能用不等价模型替代。
- 当前 Mac 只下载/保存资产并做 CPU/静态验证，不安装 GPU Conda 环境、不执行训练。

---

## File Map

| Path | Responsibility |
|---|---|
| `pyproject.toml` | 本工程 Python 包、CLI 与测试依赖 |
| `src/dlb/schema.py` | 样本、运行信息、指标与失败记录的数据类型 |
| `src/dlb/registry.py` | 加载并验证实验注册表 |
| `src/dlb/command.py` | 将统一参数渲染为上游命令 |
| `src/dlb/io.py` | JSONL 原子写入、样本校验、SHA256 |
| `src/dlb/runner.py` | 环境选择、子进程运行、恢复与失败记录 |
| `src/dlb/aggregate.py` | 结果完整性审计和 CSV/Markdown 汇总 |
| `configs/experiments.yaml` | 11 个逻辑模型、数据集支持矩阵、步数、环境与 adapter |
| `configs/schema.json` | 注册表与结果约束的机器可读 schema |
| `artifacts/sources.yaml` | 8 个仓库 URL、branch、锁定 SHA |
| `artifacts/data.yaml` | 数据集、tokenizer、revision、split、长度与下载路径 |
| `artifacts/checkpoints.yaml` | checkpoint 来源、模型/数据集映射、文件和校验和 |
| `scripts/fetch_sources.sh` | 幂等克隆、checkout 和源码校验 |
| `scripts/fetch_data.py` | 下载 LM1B、OWT 与 tokenizer snapshot |
| `scripts/preprocess_data.py` | 创建固定 packing cache 和 manifest |
| `scripts/fetch_checkpoints.py` | 从 HF、Google Drive、Zenodo 下载并校验权重 |
| `scripts/run_one.sh` | 运行单个配置 |
| `scripts/run_all.sh` | 遍历完整矩阵 |
| `scripts/smoke_all.sh` | 每个受支持模型运行 1 样本 smoke test |
| `scripts/benchmark_one.sh` | 统一 latency protocol |
| `adapters/*.py` | 各上游的命令构建和统一输出转换 |
| `evaluation/*.py` | 四项统一指标 |
| `envs/*.yml` | 9 个 Conda 环境定义 |
| `envs/create_all.sh` | 逐环境创建和错误汇总 |
| `envs/verify_all.sh` | GPU/import/version smoke checks |
| `envs/pack_all.sh` | 用 conda-pack 生成可搬运环境 |
| `slurm/*.sbatch` | 下载、生成、评测和计时 job arrays |
| `tests/` | 注册表、下载器、schema、命令、指标和汇总测试 |
| `docs/01_*.md` … `docs/08_*.md` | 中文服务器操作手册 |

---

### Task 1: Core package, schemas, and registry

**Files:**
- Create: `pyproject.toml`
- Create: `src/dlb/__init__.py`
- Create: `src/dlb/schema.py`
- Create: `src/dlb/registry.py`
- Create: `src/dlb/io.py`
- Create: `configs/experiments.yaml`
- Create: `configs/schema.json`
- Create: `tests/test_registry.py`
- Create: `tests/test_schema.py`

**Interfaces:**
- Produces: `load_registry(path: Path) -> ExperimentRegistry`
- Produces: `SampleRecord`, `RunMetadata`, `MetricRecord`, `FailureRecord`
- Produces: `atomic_json_write(path: Path, value: object) -> None` and `sha256_file(path: Path) -> str`
- Consumes: no project-internal interfaces

- [ ] **Step 1: Write failing registry and schema tests**

```python
def test_registry_contains_full_scope(registry):
    assert set(registry.models) == {
        "flm", "langflow", "duo", "mdlm", "candi", "rdlm",
        "fmlm", "duo_dcd", "duo_di4c", "mdlm_sdtt", "mdlm_di4c",
    }
    assert registry.step_grids["many"] == [1, 2, 4, 8, 16, 32, 1024]
    assert registry.step_grids["few"] == [1, 2, 4, 8, 16, 32]

def test_unsupported_cells_have_reason(registry):
    for model in registry.models.values():
        for support in model.datasets.values():
            if support.status == "unsupported":
                assert len(support.reason) >= 20

def test_sample_record_rejects_empty_text():
    with pytest.raises(ValueError):
        SampleRecord(sample_id=0, text="", token_ids=[1], seed=42,
                     generation_seconds=0.1)
```

- [ ] **Step 2: Run tests and confirm import failures**

Run: `python -m pytest tests/test_registry.py tests/test_schema.py -v`
Expected: FAIL because `dlb.registry` and `dlb.schema` do not exist.

- [ ] **Step 3: Implement typed models and validated registry loading**

Use pydantic models with `extra="forbid"`; validate model category, environment, adapter, dataset support, source provenance and exact step grids. Populate all 11 models and explicit `unsupported` cells for LangFlow/LM1B and RDLM/OWT. Add dependency-free atomic JSON writing and streaming SHA256 helpers to `src/dlb/io.py` so all later asset tasks share one implementation.

```python
MANY_STEPS = [1, 2, 4, 8, 16, 32, 1024]
FEW_STEPS = [1, 2, 4, 8, 16, 32]

class SampleRecord(BaseModel):
    sample_id: int = Field(ge=0)
    text: str = Field(min_length=1)
    token_ids: list[int] = Field(min_length=1)
    seed: int
    generation_seconds: float = Field(ge=0)
```

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_registry.py tests/test_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/dlb configs tests/test_registry.py tests/test_schema.py
git commit -m "feat: define baseline experiment registry"
```

### Task 2: Source manifest and reproducible repository fetcher

**Files:**
- Create: `artifacts/sources.yaml`
- Create: `scripts/fetch_sources.sh`
- Create: `scripts/verify_sources.py`
- Create: `tests/test_source_manifest.py`
- Create: `.gitignore`

**Interfaces:**
- Consumes: `load_registry()` for required source IDs
- Produces: `upstreams/<source_id>/` and `artifacts/source_lock.json`

- [ ] **Step 1: Write manifest coverage test**

```python
EXPECTED = {
    "flm": "https://github.com/david3684/flm.git",
    "langflow": "https://github.com/nealchen2003/LangFlow.git",
    "duo": "https://github.com/s-sahoo/duo.git",
    "mdlm": "https://github.com/kuleshov-group/mdlm.git",
    "candi": "https://github.com/patrickpynadath1/candi-diffusion.git",
    "rdlm": "https://github.com/harryjo97/RDLM.git",
    "sdtt": "https://github.com/jdeschena/sdtt.git",
    "di4c": "https://github.com/sony/di4c.git",
}

def test_source_manifest_has_exact_repositories(source_manifest):
    assert {k: v["url"] for k, v in source_manifest.items()} == EXPECTED
    assert all(len(v["commit"]) == 40 for v in source_manifest.values())
```

- [ ] **Step 2: Run test and confirm missing manifest failure**

Run: `python -m pytest tests/test_source_manifest.py -v`
Expected: FAIL because `artifacts/sources.yaml` does not exist.

- [ ] **Step 3: Fetch each official repository and pin the observed commit**

The script must clone when absent, verify origin when present, fetch only the pinned SHA, checkout detached, and reject dirty upstreams. After the first successful clone, write the actual 40-character `git rev-parse HEAD` values into `artifacts/sources.yaml` using `apply_patch` so future runs never follow moving branches.

```bash
git clone --filter=blob:none "$url" "$DLB_ROOT/upstreams/$name"
git -C "$DLB_ROOT/upstreams/$name" checkout --detach "$commit"
test "$(git -C "$DLB_ROOT/upstreams/$name" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$DLB_ROOT/upstreams/$name" status --porcelain)"
```

- [ ] **Step 4: Ignore large/untracked runtime assets without ignoring manifests**

Ignore `upstreams/`, `data/raw/`, `data/processed/`, `checkpoints/`, `results/`, `.venv/`, caches and environment archives. Do not ignore `artifacts/*.yaml`, source lock JSON, tests, docs or scripts.

- [ ] **Step 5: Run source verifier**

Run: `python scripts/verify_sources.py --root .`
Expected: eight `OK <name> <sha>` lines and exit 0.

- [ ] **Step 6: Commit**

```bash
git add .gitignore artifacts/sources.yaml scripts/fetch_sources.sh scripts/verify_sources.py tests/test_source_manifest.py
git commit -m "build: lock official baseline sources"
```

### Task 3: Data download, split, and preprocessing

**Files:**
- Create: `artifacts/data.yaml`
- Create: `scripts/fetch_data.py`
- Create: `scripts/preprocess_data.py`
- Create: `scripts/verify_data.py`
- Create: `tests/test_data_manifest.py`
- Create: `tests/test_packing.py`

**Interfaces:**
- Produces: `data/raw/huggingface/`, `data/processed/lm1b-bert-128/`, `data/processed/owt-gpt2-1024/`
- Produces: `data/manifests/<dataset>.json`
- Consumes: `atomic_json_write()` and `sha256_file()` from `dlb.io`

- [ ] **Step 1: Write split and packing tests**

```python
def test_owt_split_is_last_100k_documents():
    split = build_owt_split(total_documents=8_013_769)
    assert split.train == "train[:-100000]"
    assert split.validation == "train[-100000:]"

def test_pack_tokens_reserves_boundaries():
    blocks = list(pack_tokens([[10, 11], [12, 13, 14]], length=6,
                              bos_id=101, eos_id=102))
    assert blocks[0] == [101, 10, 11, 102, 12, 102]
    assert all(len(block) == 6 for block in blocks)
```

- [ ] **Step 2: Run focused tests and confirm failures**

Run: `python -m pytest tests/test_data_manifest.py tests/test_packing.py -v`
Expected: FAIL on missing data functions.

- [ ] **Step 3: Implement resumable Hugging Face downloads**

Use `HF_HOME=$DLB_ROOT/data/raw/huggingface`, fixed dataset/model revisions, `datasets.load_dataset(..., download_mode="reuse_dataset_if_exists")`, and `snapshot_download()` for `bert-base-uncased`, `gpt2`, and `gpt2-large`. Before OWT, require at least 55 GiB free; `--allow-low-disk` may bypass only with an explicit warning logged in metadata.

- [ ] **Step 4: Implement deterministic preprocessing**

Persist DatasetDict objects with exact sequence lengths and manifests containing source revision, tokenizer revision, split expression, number of documents, number of packed sequences, vocabulary bounds and creation timestamp.

- [ ] **Step 5: Complete local CPU/static validation and dry-run**

Run: `python -m pytest tests/test_data_manifest.py tests/test_packing.py -v`
Expected: PASS.
Run: `python scripts/fetch_data.py --dry-run --root .`
Expected: lists LM1B, OWT, BERT, GPT-2 and GPT-2 Large targets without network writes.

- [ ] **Step 6: SERVER-ONLY acceptance — download, preprocess, and verify full data**

Do not run these three commands on the local Mac. After cloning the committed
project on the Linux GPU/server, run:

Run: `python scripts/fetch_data.py --root .`
Run: `python scripts/preprocess_data.py --root . --dataset all`
Run: `python scripts/verify_data.py --root . --dataset all`
Expected: both data manifests report `verified: true`.
Local completion requires only Step 5; the ignored `data/` runtime tree remains
absent until this server-only acceptance step.

- [ ] **Step 7: Commit**

```bash
git add artifacts/data.yaml scripts/fetch_data.py scripts/preprocess_data.py scripts/verify_data.py tests/test_data_manifest.py tests/test_packing.py
git commit -m "feat: add reproducible LM1B and OWT data pipeline"
```

### Task 4: Checkpoint manifest and downloaders

**Files:**
- Create: `artifacts/checkpoints.yaml`
- Create: `scripts/fetch_checkpoints.py`
- Create: `scripts/verify_checkpoints.py`
- Create: `tests/test_checkpoint_manifest.py`
- Create: `tests/test_download_backends.py`

**Interfaces:**
- Produces: `checkpoints/{official,reference_reproduction}/...`
- Produces: `artifacts/checkpoint_lock.json`
- Consumes: Hugging Face, direct HTTP, Google Drive and Zenodo backend descriptors

- [ ] **Step 1: Write coverage and provenance tests**

```python
def test_every_supported_cell_has_checkpoint_or_training_recipe(registry, checkpoints):
    for model in registry.models.values():
        for dataset, support in model.datasets.items():
            if support.status == "supported":
                assert (model.name, dataset) in checkpoints or support.train_recipe

def test_checkpoint_sources_are_typed(checkpoints):
    assert {c.backend for c in checkpoints.values()} <= {
        "huggingface", "gdrive", "zenodo", "direct"
    }
```

- [ ] **Step 2: Run tests and confirm missing manifest failure**

Run: `python -m pytest tests/test_checkpoint_manifest.py tests/test_download_backends.py -v`.

- [ ] **Step 3: Encode known primary sources**

Include FLM/FMLM repos `david3684/{FLM-B-LM1B,FMLM-B-LM1B,FLM-B-OWT,FMLM-B-OWT}`, LangFlow `Continuous-Rivals-Discrete/langflow-owt`, Duo HF collection/checkpoints, `kuleshov-group/mdlm-owt`, `jdeschena/sdtt`, FLM baseline Drive folder `1TJO3aFWqI7ukbmjciZ6krAUFlAak1itl`, RDLM author Drive folder discovered from its README, and Zenodo record `15124163` for Di4C. Record license/terms and distinguish official from FLM reproduction.

- [ ] **Step 4: Implement resumable backends and quarantine**

Download into `<file>.partial`, bind resumable HTTP bytes to ETag/Last-Modified with `If-Range`, compute SHA256, then atomically rename. Existing mismatches move to `checkpoints/quarantine/<timestamp>/`. Google Drive resources download every manifest-pinned file ID with `gdown --id` into ID-keyed staging; folder IDs are provenance context only. Zenodo files come from its records API; HF uses immutable `snapshot_download` allow patterns plus a required-file inventory that must include the primary weight.

- [ ] **Step 5: Run tests and dry-run**

Run: `python -m pytest tests/test_checkpoint_manifest.py tests/test_download_backends.py -v`
Expected: PASS.
Run: `python scripts/fetch_checkpoints.py --root . --dry-run`
Expected: each downloadable artifact lists backend, destination and provenance.

- [ ] **Step 6: SERVER-ONLY acceptance — download public checkpoints and verify**

Do not run these commands on the local Mac. After cloning the committed project
on the Linux GPU/server, run:

Run: `python scripts/fetch_checkpoints.py --root . --all-public`
Run: `python scripts/verify_checkpoints.py --root .`
Expected: all downloaded files have size and SHA256 in `checkpoint_lock.json`; inaccessible resources have structured status and are not treated as success.

- [ ] **Step 7: Commit**

```bash
git add artifacts/checkpoints.yaml scripts/fetch_checkpoints.py scripts/verify_checkpoints.py tests/test_checkpoint_manifest.py tests/test_download_backends.py
git commit -m "feat: track baseline checkpoints and provenance"
```

### Task 5: Conda environment definitions

**Files:**
- Create: `envs/flm.yml`
- Create: `envs/langflow.yml`
- Create: `envs/duo.yml`
- Create: `envs/mdlm.yml`
- Create: `envs/candi.yml`
- Create: `envs/rdlm.yml`
- Create: `envs/sdtt.yml`
- Create: `envs/di4c.yml`
- Create: `envs/eval.yml`
- Create: `envs/create_all.sh`
- Create: `envs/verify_all.sh`
- Create: `envs/pack_all.sh`
- Create: `tests/test_envs.py`

**Interfaces:**
- Produces: environments `dlb-flm`, `dlb-langflow`, `dlb-duo`, `dlb-mdlm`, `dlb-candi`, `dlb-rdlm`, `dlb-sdtt`, `dlb-di4c`, `dlb-eval`
- Consumes: pinned upstream requirement files

- [ ] **Step 1: Write environment coverage and pinning tests**

```python
def test_all_declared_environments_exist(registry):
    names = {yaml.safe_load(p.read_text())["name"] for p in Path("envs").glob("*.yml")}
    assert {m.environment for m in registry.models.values()} <= names
    assert "dlb-eval" in names

def test_gpu_envs_pin_python_and_torch():
    for path in Path("envs").glob("*.yml"):
        text = path.read_text()
        assert re.search(r"python[=<>]", text)
        assert "torch" in text
```

- [ ] **Step 2: Run test and confirm missing YAML failure**

Run: `python -m pytest tests/test_envs.py -v`.

- [ ] **Step 3: Inspect each pinned source and create explicit YAMLs**

Preserve upstream versions where valid. Fix incompatibilities explicitly: FLM uses FlashAttention 2.8.3; Duo uses 2.7.4.post1 and CUDA 12.4; RDLM uses Python 3.9/PyTorch 2.3.1; SDTT replaces its obsolete nightly CPU torchdata command with a CUDA-compatible stable package set. Put FlashAttention installation after PyTorch in `create_all.sh` with `--no-build-isolation` where required.

- [ ] **Step 4: Implement non-destructive create/verify/pack scripts**

`create_all.sh` uses `conda env create` only when absent and `conda env update --file <env.yml>` when present. `verify_all.sh` runs imports and prints JSON version records. `pack_all.sh` writes `artifacts/conda-packs/<name>.tar.gz` via `conda-pack` without deleting environments.

- [ ] **Step 5: Validate without creating environments**

Run: `python -m pytest tests/test_envs.py -v`
Run: `bash -n envs/create_all.sh envs/verify_all.sh envs/pack_all.sh`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add envs tests/test_envs.py
git commit -m "build: define isolated baseline conda environments"
```

### Task 6: Atomic sample I/O and runner

**Files:**
- Modify: `src/dlb/io.py`
- Create: `src/dlb/runner.py`
- Create: `scripts/run_one.sh`
- Create: `tests/test_io.py`
- Create: `tests/test_runner.py`

**Interfaces:**
- Produces: `write_samples_atomic(path, records)` and `validate_samples(path, expected=1024)`
- Produces: `run_experiment(request: RunRequest) -> RunResult`
- Consumes: registry and adapter `build_command()` / `convert_outputs()`

- [ ] **Step 1: Write atomicity and failure tests**

```python
def test_validate_requires_exact_count(tmp_path):
    path = tmp_path / "samples.jsonl"
    write_samples_atomic(path, make_records(3))
    with pytest.raises(SampleCountError):
        validate_samples(path, expected=1024)

def test_runner_records_command_failure(tmp_path, fake_adapter):
    result = run_experiment(make_request(command=["sh", "-c", "exit 7"]), tmp_path)
    assert result.status == "failed"
    assert json.loads((result.run_dir / "failure.json").read_text())["exit_code"] == 7
```

- [ ] **Step 2: Run tests and confirm failures**

Run: `python -m pytest tests/test_io.py tests/test_runner.py -v`.

- [ ] **Step 3: Implement atomic outputs and resumable runner**

Write JSONL to `.partial`, fsync, validate unique IDs/nonempty text/token IDs/count, then `os.replace`. A run is skippable only when output schema, count, command hash, checkpoint SHA and source SHA match metadata.

- [ ] **Step 4: Implement `run_one.sh`**

Resolve `DLB_ROOT` from the script location, reject unknown model/dataset/steps, call the correct Conda environment with `conda run -n <env> python -m dlb.runner`, and preserve the child exit code.

- [ ] **Step 5: Run tests and shell syntax check**

Run: `python -m pytest tests/test_io.py tests/test_runner.py -v`
Run: `bash -n scripts/run_one.sh`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dlb/io.py src/dlb/runner.py scripts/run_one.sh tests/test_io.py tests/test_runner.py
git commit -m "feat: add resumable baseline run harness"
```

### Task 7: FLM-lineage and discrete teacher adapters

**Files:**
- Create: `adapters/__init__.py`
- Create: `adapters/base.py`
- Create: `adapters/flm.py`
- Create: `adapters/duo.py`
- Create: `adapters/mdlm.py`
- Create: `adapters/candi.py`
- Create: `src/dlb/command.py`
- Create: `patches/flm/README.md`
- Create: `patches/duo/README.md`
- Create: `patches/mdlm/README.md`
- Create: `patches/candi/README.md`
- Create: `tests/test_teacher_adapters.py`

**Interfaces:**
- Produces: `Adapter.build_command(request) -> list[str]`
- Produces: `Adapter.convert_outputs(upstream_path, run_dir) -> Path`
- Produces: `python -m dlb.command` dry-run and command-rendering CLI
- Covers: FLM, FMLM, Duo, Duo+DCD, MDLM, CANDI

- [ ] **Step 1: Write command rendering tests**

```python
def test_flm_owt_command_maps_steps_and_count(request_factory):
    cmd = FLMAdapter().build_command(request_factory("flm", "owt", 32))
    joined = " ".join(cmd)
    assert "main.py" in joined and "sampling.steps=32" in joined
    assert "sampling.num_sample_batches" in joined
    assert "eval.checkpoint_path=" in joined

def test_discrete_adapters_force_paper_sampler(request_factory):
    for adapter in [DuoAdapter(), MDLMAdapter()]:
        joined = " ".join(adapter.build_command(request_factory(adapter.name, "owt", 8)))
        assert "sampling.noise_removal=ancestral" in joined
        assert "sampling.temperature=1.0" in joined
```

- [ ] **Step 2: Run tests and confirm adapter import failure**

Run: `python -m pytest tests/test_teacher_adapters.py -v`.

- [ ] **Step 3: Map actual pinned upstream options**

Inspect the checked-out Hydra configs and official scripts. Render argv arrays rather than shell strings. Derive `sampling.num_sample_batches` from 1,024 and the configured eval batch size, then trim only excess samples deterministically. Use model-specific solver/algorithm configs exactly as specified globally.

- [ ] **Step 4: Convert every upstream output to canonical JSONL**

Prefer upstream token IDs before decode. When only text exists, re-tokenize with the dataset tokenizer and record `token_ids_source: retokenized`. Reject empty output, unexpected sample count and missing checkpoint metadata.

- [ ] **Step 5: Run adapter tests and dry-render every covered cell**

Run: `python -m pytest tests/test_teacher_adapters.py -v`
Run: `python -m dlb.command --models flm,fmlm,duo,duo_dcd,mdlm,candi --datasets lm1b,owt --dry-run`
Expected: supported commands contain no braces, `None`, empty path or unresolved environment variables.

- [ ] **Step 6: Commit**

```bash
git add adapters src/dlb/command.py patches tests/test_teacher_adapters.py
git commit -m "feat: adapt FLM and discrete teacher baselines"
```

### Task 8: LangFlow and RDLM adapters

**Files:**
- Create: `adapters/langflow.py`
- Create: `adapters/rdlm.py`
- Create: `patches/langflow/README.md`
- Create: `patches/rdlm/README.md`
- Create: `tests/test_continuous_adapters.py`

**Interfaces:**
- Produces the same Adapter contract as Task 7
- Covers: LangFlow/OWT and RDLM/LM1B only

- [ ] **Step 1: Write supported/unsupported and command tests**

```python
def test_langflow_is_owt_only(registry):
    assert registry.models["langflow"].datasets["owt"].status == "supported"
    assert registry.models["langflow"].datasets["lm1b"].status == "unsupported"

def test_langflow_inference_cli(request_factory):
    cmd = LangFlowAdapter().build_command(request_factory("langflow", "owt", 1024))
    assert cmd[1].endswith("inference.py")
    assert cmd[cmd.index("--num_steps") + 1] == "1024"
    assert cmd[cmd.index("--seq_length") + 1] == "1024"

def test_rdlm_uses_official_sde_sampler(request_factory):
    joined = " ".join(RDLMAdapter().build_command(request_factory("rdlm", "lm1b", 32)))
    assert "run_mode=sample" in joined and "exp=sample_lm1b" in joined
```

- [ ] **Step 2: Run tests and confirm failures**

Run: `python -m pytest tests/test_continuous_adapters.py -v`.

- [ ] **Step 3: Implement adapters against pinned CLIs**

LangFlow calls `inference.py --checkpoint ... --num_samples 1024 --num_steps N --seq_length 1024 --seed 42 --output ...`. RDLM calls its Hydra `main.py` with the saved `config.yaml`, `sde.pkl`, checkpoint path, requested step override confirmed against `sampling.py`, and `seed=42`.

- [ ] **Step 4: Run tests and dry-run**

Run: `python -m pytest tests/test_continuous_adapters.py -v`
Run: `python -m dlb.command --models langflow,rdlm --datasets lm1b,owt --dry-run`
Expected: two supported commands plus two explicit unsupported records.

- [ ] **Step 5: Commit**

```bash
git add adapters/langflow.py adapters/rdlm.py patches/langflow patches/rdlm tests/test_continuous_adapters.py
git commit -m "feat: adapt LangFlow and RDLM baselines"
```

### Task 9: SDTT and Di4C distilled adapters

**Files:**
- Create: `adapters/sdtt.py`
- Create: `adapters/di4c.py`
- Create: `adapters/sample_sdtt.py`
- Create: `adapters/sample_di4c.py`
- Create: `patches/sdtt/README.md`
- Create: `patches/di4c/README.md`
- Create: `tests/test_distilled_adapters.py`

**Interfaces:**
- Produces the same Adapter contract as Task 7
- Covers: MDLM+SDTT, MDLM+Di4C, Duo+Di4C

- [ ] **Step 1: Write checkpoint round and sampler tests**

```python
def test_sdtt_uses_kld_round_seven(request_factory):
    cmd = SDTTAdapter().build_command(request_factory("mdlm_sdtt", "owt", 4))
    joined = " ".join(cmd)
    assert "--loss" in cmd and "kld" in cmd
    assert "--round" in cmd and "7" in cmd
    assert "--num-steps" in cmd and "4" in cmd

def test_di4c_requires_teacher_family_match(request_factory):
    mdlm = Di4CAdapter("mdlm").resolve_checkpoint(request_factory("mdlm_di4c", "owt", 8))
    duo = Di4CAdapter("duo").resolve_checkpoint(request_factory("duo_di4c", "owt", 8))
    assert mdlm.teacher_family == "masked"
    assert duo.teacher_family == "uniform"
```

- [ ] **Step 2: Run tests and confirm failures**

Run: `python -m pytest tests/test_distilled_adapters.py -v`.

- [ ] **Step 3: Implement SDTT sampling wrapper**

Load the released KLD round-7 student for the primary result, call `model.sample(n_samples=batch, num_steps=N, seq_len=L)`, and stream canonical records without accumulating all decoded text on GPU.

- [ ] **Step 4: Implement Di4C wrappers with explicit teacher identity**

Use `sdtt6-di4c2.ckpt`/`sdtt7-di4c2.ckpt` only for the masked teacher family. Use FLM reference reproduction artifacts or a separately trained uniform-teacher Di4C checkpoint for Duo+Di4C; never relabel the masked checkpoint. The manifest determines which intermediate checkpoint corresponds to 20k LM1B or 50k OWT training steps.

- [ ] **Step 5: Run tests and render matrix**

Run: `python -m pytest tests/test_distilled_adapters.py -v`
Run: `python -m dlb.command --models mdlm_sdtt,mdlm_di4c,duo_di4c --datasets lm1b,owt --dry-run`
Expected: commands carry checkpoint provenance and teacher family.

- [ ] **Step 6: Commit**

```bash
git add adapters/sdtt.py adapters/di4c.py adapters/sample_sdtt.py adapters/sample_di4c.py patches/sdtt patches/di4c tests/test_distilled_adapters.py
git commit -m "feat: adapt SDTT and Di4C distilled baselines"
```

### Task 10: Unified quality and diversity metrics

**Files:**
- Create: `evaluation/__init__.py`
- Create: `evaluation/generative_perplexity.py`
- Create: `evaluation/unigram_entropy.py`
- Create: `evaluation/self_bleu.py`
- Create: `evaluation/evaluate.py`
- Create: `tests/test_perplexity.py`
- Create: `tests/test_entropy.py`
- Create: `tests/test_self_bleu.py`
- Create: `tests/fixtures/sample_texts.jsonl`

**Interfaces:**
- Produces: `compute_gen_ppl(texts, model, tokenizer) -> PPLResult`
- Produces: `mean_unigram_entropy(records, special_ids) -> EntropyResult`
- Produces: `compute_self_bleu(texts, config) -> SelfBleuResult`
- Consumes: canonical sample JSONL

- [ ] **Step 1: Write metric invariant tests**

```python
def test_ppl_uses_token_weighted_nll():
    result = aggregate_nll([(math.log(2), 1), (3 * math.log(4), 3)])
    assert result.perplexity == pytest.approx(math.exp((math.log(2) + 3 * math.log(4)) / 4))

def test_entropy_in_nats():
    assert unigram_entropy([1, 1, 2, 2]) == pytest.approx(math.log(2))
    assert unigram_entropy([7, 7, 7]) == 0.0

def test_self_bleu_detects_mode_collapse():
    collapsed = ["the same four words here"] * 8
    diverse = [f"unique sentence number {i} token{i}" for i in range(8)]
    assert compute_self_bleu(collapsed).score > 0.99
    assert compute_self_bleu(diverse).score < compute_self_bleu(collapsed).score
```

- [ ] **Step 2: Run tests and confirm failures**

Run: `python -m pytest tests/test_perplexity.py tests/test_entropy.py tests/test_self_bleu.py -v`.

- [ ] **Step 3: Implement GPT-2 Large PPL**

Retokenize text, mask padding, shift logits/labels, accumulate `reduction="sum"` loss and valid token count across batches, then exponentiate once. Save model/tokenizer revision and total evaluated token count.

- [ ] **Step 4: Implement paper-compatible entropy and Self-BLEU**

Port the exact FLM repository conventions after inspecting its `metrics.py`, including n-gram weights, smoothing and score normalization. Keep pure functions for regression tests; record special-token handling.

- [ ] **Step 5: Implement evaluation CLI and tests**

Run: `python -m pytest tests/test_perplexity.py tests/test_entropy.py tests/test_self_bleu.py -v`
Run: `python -m evaluation.evaluate --samples tests/fixtures/sample_texts.jsonl --metrics entropy,self_bleu --output /tmp/dlb-metrics.json`
Expected: PASS and valid JSON with sample count.

- [ ] **Step 6: Commit**

```bash
git add evaluation tests/test_perplexity.py tests/test_entropy.py tests/test_self_bleu.py tests/fixtures/sample_texts.jsonl
git commit -m "feat: add unified generation quality metrics"
```

### Task 11: Standardized generation timing

**Files:**
- Create: `evaluation/timing.py`
- Create: `scripts/benchmark_one.sh`
- Create: `tests/test_timing.py`

**Interfaces:**
- Produces: `benchmark(generate_one, synchronize, warmups=5, repeats=32) -> TimingResult`
- Consumes: adapter-provided in-memory generation callback

- [ ] **Step 1: Write exclusion and synchronization tests**

```python
def test_benchmark_excludes_warmups(fake_clock):
    calls = []
    result = benchmark(lambda: calls.append("generate"), lambda: calls.append("sync"),
                       warmups=5, repeats=32, clock=fake_clock)
    assert calls.count("generate") == 37
    assert result.num_timed_samples == 32
    assert result.seconds_per_sample > 0

def test_benchmark_synchronizes_around_each_timed_call(sync_spy):
    benchmark(lambda: None, sync_spy, warmups=0, repeats=2)
    assert sync_spy.call_count == 4
```

- [ ] **Step 2: Run test and confirm failure**

Run: `python -m pytest tests/test_timing.py -v`.

- [ ] **Step 3: Implement timing result and CLI**

Report arithmetic mean seconds/sample, median, standard deviation, raw durations, batch size, GPU metadata and exclusions. Require batch size 1 for `primary_latency`; allow throughput runs under a distinct label.

- [ ] **Step 4: Run tests and shell check**

Run: `python -m pytest tests/test_timing.py -v`
Run: `bash -n scripts/benchmark_one.sh`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/timing.py scripts/benchmark_one.sh tests/test_timing.py
git commit -m "feat: standardize generation latency measurement"
```

### Task 12: Training and distillation recipes

**Files:**
- Create: `src/dlb/recipes.py`
- Create: `scripts/train/flm.sh`
- Create: `scripts/train/duo.sh`
- Create: `scripts/train/mdlm.sh`
- Create: `scripts/train/candi.sh`
- Create: `scripts/train/rdlm.sh`
- Create: `scripts/distill/fmlm.sh`
- Create: `scripts/distill/duo_dcd.sh`
- Create: `scripts/distill/mdlm_sdtt.sh`
- Create: `scripts/distill/di4c.sh`
- Create: `tests/test_training_recipes.py`

**Interfaces:**
- Produces self-trained checkpoints in `checkpoints/self_trained/<dataset>/<model>/`
- Consumes processed datasets and teacher checkpoints
- Produces: `load_recipe(model: str, dataset: str) -> TrainingRecipe`
- The SDTT/Di4C wrappers must adapt teacher embeddings/output heads before loading: uniform Duo teachers need an absorbing mask state appended, while masked BERT/MDLM teachers need their existing mask state mapped into the absorbing model layout. Direct `strict=False` loading is forbidden because it still rejects tensor-shape mismatches.
- `scripts/train/candi.sh` must implement `--source upstreams/candi --dataset <dataset> --output <path>` without delegating to site-specific Slurm scripts or hidden scratch paths.
- `scripts/distill/duo_dcd.sh` must implement `--source upstreams/duo --dataset <dataset> --teacher <path> --output <path> --rounds 8 --steps-per-round 10000 --global-batch-size 128 --learning-rate 6e-5`; these manifest commands are pinned prerequisites and the wrappers are created only in Task 12.

- [ ] **Step 1: Write paper-hyperparameter tests**

```python
def test_flm_recipe_matches_reference():
    recipe = load_recipe("flm", "lm1b")
    assert recipe.max_steps == 1_000_000
    assert recipe.global_batch_size == 512
    assert recipe.learning_rate == pytest.approx(3e-4)
    assert recipe.warmup_steps == 2500

def test_sdtt_and_dcd_round_schedule():
    for name in ["mdlm_sdtt", "duo_dcd"]:
        recipe = load_recipe(name, "lm1b")
        assert recipe.rounds == 8
        assert recipe.steps_per_round == 10_000
        assert recipe.global_batch_size == 128
        assert recipe.learning_rate == pytest.approx(6e-5)
```

- [ ] **Step 2: Run tests and confirm missing recipes**

Run: `python -m pytest tests/test_training_recipes.py -v`.

- [ ] **Step 3: Wrap official training scripts with exact overrides**

Use the reference global batch size via gradient accumulation, never silently reduce it. FMLM distills 100k steps with the FLM teacher. SDTT/DCD run 8 × 10k rounds. Di4C saves and registers intermediate 20k LM1B and 50k OWT checkpoints used by the reference comparison.

- [ ] **Step 4: Validate all rendered commands**

Run: `python -m pytest tests/test_training_recipes.py -v`
Run: `for f in scripts/train/*.sh scripts/distill/*.sh; do bash -n "$f"; done`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dlb/recipes.py scripts/train scripts/distill tests/test_training_recipes.py
git commit -m "feat: document reproducible training and distillation recipes"
```

### Task 13: Matrix orchestration, Slurm, and aggregation

**Files:**
- Create: `src/dlb/matrix.py`
- Create: `src/dlb/aggregate.py`
- Create: `scripts/run_all.sh`
- Create: `scripts/smoke_all.sh`
- Create: `scripts/evaluate_all.sh`
- Create: `scripts/aggregate_results.py`
- Create: `slurm/generate_array.sbatch`
- Create: `slurm/evaluate.sbatch`
- Create: `slurm/benchmark_array.sbatch`
- Create: `slurm/train.sbatch`
- Create: `tests/test_matrix.py`
- Create: `tests/test_aggregate.py`

**Interfaces:**
- Produces: matrix task TSV, `results/summary/results_long.csv`, `results_wide.csv`, `README.md`, `failures.csv`
- Consumes: registry, canonical samples, metric and timing JSON

- [ ] **Step 1: Write matrix cardinality and aggregation tests**

```python
def test_matrix_has_only_declared_steps(registry):
    tasks = build_matrix(registry)
    for task in tasks:
        expected = registry.step_grids[registry.models[task.model].category]
        assert task.steps in expected

def test_aggregate_rejects_missing_metric(tmp_path, complete_fake_run):
    complete_fake_run.metric_path("self_bleu").unlink()
    with pytest.raises(IncompleteMatrixError):
        aggregate(complete_fake_run.root, strict=True)
```

- [ ] **Step 2: Run tests and confirm failures**

Run: `python -m pytest tests/test_matrix.py tests/test_aggregate.py -v`.

- [ ] **Step 3: Implement serial/local orchestration**

`run_all.sh` emits a deterministic TSV, skips explicit unsupported cells, preserves per-task exit status and continues after failures. `smoke_all.sh` overrides samples to 1 without marking full tasks complete.

- [ ] **Step 4: Implement Slurm arrays**

Each array reads one TSV row by `SLURM_ARRAY_TASK_ID`, activates the registered environment through `conda run`, writes isolated logs, and passes GPU/partition/time/memory through environment variables with documented defaults.

- [ ] **Step 5: Implement strict and partial aggregation**

Strict mode exits nonzero for any missing supported cell; partial mode emits results plus failures/unsupported reasons. Wide tables use dataset/model rows and step/metric columns.

- [ ] **Step 6: Run tests and syntax checks**

Run: `python -m pytest tests/test_matrix.py tests/test_aggregate.py -v`
Run: `bash -n scripts/run_all.sh scripts/smoke_all.sh scripts/evaluate_all.sh slurm/*.sbatch`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/dlb/matrix.py src/dlb/aggregate.py scripts/run_all.sh scripts/smoke_all.sh scripts/evaluate_all.sh scripts/aggregate_results.py slurm tests/test_matrix.py tests/test_aggregate.py
git commit -m "feat: orchestrate and aggregate baseline matrix"
```

### Task 14: Chinese server runbook

**Files:**
- Create: `README.md`
- Create: `docs/01_sources_and_availability.md`
- Create: `docs/02_datasets.md`
- Create: `docs/03_environments.md`
- Create: `docs/04_checkpoints.md`
- Create: `docs/05_training.md`
- Create: `docs/06_generation_and_evaluation.md`
- Create: `docs/07_slurm.md`
- Create: `docs/08_troubleshooting.md`
- Create: `docs/EXPERIMENT_MATRIX.md`
- Create: `scripts/verify_docs.py`
- Create: `tests/test_docs_commands.py`

**Interfaces:**
- Consumes every public script and registry option
- Produces copy-paste commands from server bootstrap through result collection

- [ ] **Step 1: Write docs command validation test**

```python
def test_documented_local_scripts_exist():
    for md in Path("docs").glob("*.md"):
        for command in extract_shell_commands(md.read_text()):
            for path in extract_project_script_paths(command):
                assert Path(path).exists(), f"{md}: missing {path}"
```

- [ ] **Step 2: Run test and confirm missing docs failure**

Run: `python -m pytest tests/test_docs_commands.py -v`.

- [ ] **Step 3: Write root quick start**

The exact top-level sequence is:

```bash
export DLB_ROOT="$(pwd -P)"
bash scripts/fetch_sources.sh
python scripts/fetch_data.py --root "$DLB_ROOT"
python scripts/fetch_checkpoints.py --root "$DLB_ROOT" --all-public
bash envs/create_all.sh
python scripts/preprocess_data.py --root "$DLB_ROOT" --dataset all
bash scripts/smoke_all.sh
bash scripts/run_all.sh
bash scripts/evaluate_all.sh
python scripts/aggregate_results.py --root "$DLB_ROOT" --strict
```

- [ ] **Step 4: Document every method and resource gap**

For all 11 logical models, give environment creation, checkpoint/training choice, one-config command, full-grid command, expected output path, GPU guidance, recovery and provenance interpretation. Explicitly explain LangFlow/LM1B and RDLM/OWT unsupported cells.

- [ ] **Step 5: Document Slurm and troubleshooting**

Cover job arrays, gradient accumulation, FlashAttention build issues, HF offline mode, Google Drive quota, Zenodo retries, OOM policy, corrupted partial files and how to resume without overwriting valid results.

- [ ] **Step 6: Run docs test and link scan**

Run: `python -m pytest tests/test_docs_commands.py -v`
Run: `python scripts/verify_docs.py --root .`
Expected: all local links/scripts resolve and all 11 models are mentioned.

- [ ] **Step 7: Commit**

```bash
git add README.md docs tests/test_docs_commands.py scripts/verify_docs.py
git commit -m "docs: add end-to-end baseline server runbook"
```

### Task 15: Final verification and completion audit

**Files:**
- Create: `scripts/verify_project.py`
- Create: `tests/test_project_audit.py`
- Modify: `README.md`

**Interfaces:**
- Consumes all manifests, registries, scripts, environments, docs and local assets
- Produces `artifacts/project_audit.json`

- [ ] **Step 1: Write completion-audit test**

```python
def test_audit_covers_every_requirement(project_audit):
    required = {
        "sources", "datasets", "checkpoints", "environments", "models",
        "step_grids", "sample_count", "metrics", "timing", "commands", "docs",
    }
    assert set(project_audit["requirements"]) == required
    assert all(item["status"] in {"verified", "unsupported_with_evidence"}
               for item in project_audit["requirements"].values())
```

- [ ] **Step 2: Run full unit test suite**

Run: `python -m pytest -q`
Expected: all tests PASS.

- [ ] **Step 3: Run static script and configuration verification**

Run: `find scripts envs slurm -type f \( -name '*.sh' -o -name '*.sbatch' \) -print0 | xargs -0 -n1 bash -n`
Run: `python scripts/verify_project.py --root . --write-report artifacts/project_audit.json`
Expected: exit 0; report contains eight verified source clones, two data downloads/preprocessed caches or explicit download failure evidence, every public checkpoint status, nine parseable environments, 11 models and four metric implementations.

- [ ] **Step 4: Audit exact user requirements against authoritative state**

Inspect source directories and SHAs, `du -sh data checkpoints`, environment YAMLs, rendered commands for every supported matrix cell, metric tests, timing tests and every documented path. Do not infer success from a narrow test; record missing/inaccessible external assets as incomplete unless the design explicitly permits `unsupported_with_evidence` for that model/dataset cell.

- [ ] **Step 5: Update README verification snapshot**

Add the audit timestamp, source SHA table, local dataset/checkpoint sizes, test command/output summary and the two known unsupported combinations. Do not claim server GPU smoke tests have run locally.

- [ ] **Step 6: Commit**

```bash
git add scripts/verify_project.py tests/test_project_audit.py artifacts/project_audit.json README.md
git commit -m "test: audit diffusion baseline project completeness"
```

---

## Execution Checkpoints

1. After Tasks 1–5: verify source/data/checkpoint manifests and all environment YAMLs before writing adapters.
2. After Tasks 6–9: dry-render every supported model/dataset/step command and review upstream compatibility.
3. After Tasks 10–13: run all CPU tests and a mocked full matrix before writing final docs.
4. After Task 15: perform the requirement-by-requirement completion audit before marking the goal complete.
