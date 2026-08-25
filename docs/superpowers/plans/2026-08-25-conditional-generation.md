# Zero-Shot Conditional Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a zero-training `c64_zs_v1` benchmark that evaluates every existing unconditional checkpoint with an immutable 64-token real prefix and a generated continuation.

**Architecture:** Extend the shared request and adapter boundaries with an explicit conditional mode while keeping unconditional identities and artifacts backward-compatible. Conditional prompts, samples, matrix, metrics, timing, aggregation, and scripts use separate versioned contracts and `results/conditional/` paths; native sampler projection APIs are used when available, with scoped adapter-level runtime clamps for other families.

**Tech Stack:** Python 3.11, Pydantic 2, PyYAML, Hugging Face Datasets/Transformers, PyTorch, MAUVE 0.3.0, pytest 8, Bash, existing pinned model-specific Conda environments.

**Spec:** `docs/superpowers/specs/2026-08-25-conditional-generation-design.md`

## Global Constraints

- Protocol identifier is exactly `c64_zs_v1`; configuration schema version is `1`.
- Reuse existing checkpoints only: no training, fine-tuning, checkpoint rewriting, or upstream repository edits.
- Use selection seed `42`, sampling seed `42`, exactly `1,024` prompts, prefix length `64`, and aligned evaluation-continuation length `64`.
- Produce completion `0` for prompt IDs `0..1023` and completions `1..4` for prompt IDs `0..255`, exactly `2,048` records per supported step task.
- LM1B uses the existing BERT 128-token processed validation split; OWT uses the existing GPT-2 1,024-token processed validation split.
- OWT generation retains the full 960-token suffix; aligned quality metrics use only its first 64 suffix tokens.
- The first 64 output tokens must equal the prompt prefix after every relevant sampler transition and in every published record; any mismatch invalidates the artifact.
- The conditional matrix contains the same 132 supported step tasks as the unconditional matrix, with RDLM/OWT retained in a separate unsupported inventory.
- Conditional outputs live below `results/conditional/`; unconditional result paths and successful-publication identity semantics remain unchanged.
- All model/tokenizer/checkpoint/prompt/sample artifacts are local-only and SHA-256 bound; no runtime network fallback is allowed.
- Artifact writes are atomic and fail closed; batch-size incompatibility, partial schedules, duplicate IDs, digest drift, and token-bound violations are errors.
- Preserve unrelated working-tree changes, including user-owned untracked diagnostics files.

## File and responsibility map

**Protocol and prompt artifacts**

- Create `configs/conditional.yaml`: canonical experiment constants.
- Create `src/dlb/conditional_prompts.py`: typed protocol/prompt contracts, deterministic selection, prompt build/verify logic.
- Create `scripts/build_conditional_prompts.py` and `scripts/verify_conditional_prompts.py`: thin CLIs.

**Sample and execution contracts**

- Modify `src/dlb/schema.py`: add `ConditionalSampleRecord` without changing `SampleRecord`.
- Modify `src/dlb/io.py`: conditional JSONL reader/validator/atomic writer.
- Modify `src/dlb/runner.py`: conditional request identity, validation, publication branch, and CLI flags.
- Create `src/dlb/conditional_matrix.py`: separate versioned 132-task matrix and unsupported inventory.

**Conditioning runtime**

- Create `src/dlb/adapters/conditional_runtime.py`: prompt batching, token/one-hot/embedding clamps, scoped monkey-patch restoration, native projection builders.
- Modify `src/dlb/adapters/base.py`, `src/dlb/adapters/capture.py`, and all files in `src/dlb/adapters/{flm,mdlm,duo,candi,langflow,rdlm,sdtt,di4c}.py`: serialize and consume conditional arguments.
- Modify `adapters/_distilled_runtime.py`, `adapters/sample_sdtt.py`, `adapters/sample_di4c.py`, and `adapters/sample_langflow.py`: server-backed prompt loading, projection, capture, and timing.

**Evaluation and reporting**

- Create `evaluation/conditional_perplexity.py`: suffix-only causal loss masking.
- Create `evaluation/mauve_score.py`: pinned local-only MAUVE wrapper.
- Create `evaluation/conditional_evaluate.py`: exact schedule validation and all conditional metrics.
- Modify `evaluation/self_bleu.py` and `evaluation/unigram_entropy.py`: token-row/group APIs reusable by both protocols.
- Create `src/dlb/conditional_benchmarking.py`: conditional latency command and provenance publication.
- Create `src/dlb/conditional_aggregate.py`: strict conditional result validation and summary publication.

**Launch and verification**

- Create conditional one/all/four-GPU generation, evaluation, timing, smoke, and aggregate scripts under `scripts/`.
- Modify `src/dlb/gpu_matrix.py`: accept an explicit conditional protocol and route to conditional scripts without altering default routing.
- Modify `README.md` and the relevant experiment documentation with server commands and zero-shot/OOD interpretation.

---

### Task 1: Canonical protocol and deterministic prompt artifacts

**Files:**

- Create: `configs/conditional.yaml`
- Create: `src/dlb/conditional_prompts.py`
- Create: `scripts/build_conditional_prompts.py`
- Create: `scripts/verify_conditional_prompts.py`
- Test: `tests/test_conditional_prompts.py`

**Interfaces:**

- Consumes: `artifacts/data.yaml`, `data/manifests/{lm1b,owt}.json`, and processed validation rows containing `input_ids`.
- Produces: `ConditionalProtocol`, `PromptRecord`, `PromptManifest`, `load_protocol(path)`, `select_source_indices(row_count, count, seed)`, `build_prompts(root, dataset_id, protocol)`, and `verify_prompts(root, dataset_id, protocol) -> PromptManifest`.

- [ ] **Step 1: Write failing protocol and selection tests**

```python
def test_protocol_has_exact_c64_production_contract(tmp_path):
    path = tmp_path / "conditional.yaml"
    path.write_text(PRODUCTION_CONFIG, encoding="utf-8")
    protocol = load_protocol(path)
    assert protocol.protocol == "c64_zs_v1"
    assert (protocol.prompt_count, protocol.prefix_length) == (1024, 64)
    assert protocol.datasets["owt"].model_length == 1024


def test_selection_is_unique_stable_and_seed_bound():
    first = select_source_indices(4096, 1024, 42)
    assert first == select_source_indices(4096, 1024, 42)
    assert first != select_source_indices(4096, 1024, 43)
    assert len(first) == len(set(first)) == 1024
```

- [ ] **Step 2: Run tests and confirm missing-module failure**

Run: `pytest tests/test_conditional_prompts.py -q`

Expected: collection fails because `dlb.conditional_prompts` does not exist.

- [ ] **Step 3: Implement the typed protocol, SHA-256 ordering, and prompt records**

```python
class ConditionalProtocol(StrictModel):
    schema_version: Literal[1]
    protocol: Literal["c64_zs_v1"]
    selection_seed: StrictInt
    sampling_seed: StrictInt
    prompt_count: Literal[1024]
    prefix_length: Literal[64]
    evaluation_continuation_length: Literal[64]
    diversity_prompt_count: Literal[256]
    completions_per_diversity_prompt: Literal[5]
    datasets: dict[str, ConditionalDataset]


def select_source_indices(row_count: int, count: int, seed: int) -> list[int]:
    if row_count < count or count <= 0:
        raise ValueError("processed validation split has too few rows")
    return sorted(
        range(row_count),
        key=lambda index: hashlib.sha256(f"{seed}:{index}".encode()).digest(),
    )[:count]
```

`build_prompts` must load `data/processed/<dataset-contract>/validation` with `datasets.load_from_disk`, preserve selected order as `prompt_id`, slice exactly `[0:64]` and `[64:128]`, validate against the dataset vocabulary size, write deterministic compact JSONL through the safe atomic writer, and publish a manifest that binds source manifest SHA, tokenizer ID/revision, selection algorithm, seed, counts, lengths, and prompt-file SHA.

- [ ] **Step 4: Add tamper, token-bound, and CLI tests, then make them pass**

```python
def test_verify_rejects_prompt_file_tampering(built_prompt_tree):
    prompts = built_prompt_tree / "data/conditional/lm1b-c64/prompts.jsonl"
    prompts.write_bytes(prompts.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="prompt file SHA-256"):
        verify_prompts(built_prompt_tree, "lm1b", production_protocol())


def test_prompt_cli_uses_processed_validation_without_repacking(monkeypatch, tmp_path):
    observed = install_fake_load_from_disk(monkeypatch, rows=2048)
    assert build_main(["--root", str(tmp_path), "--dataset", "lm1b"]) == 0
    assert observed == [tmp_path / "data/processed/lm1b-bert-128/validation"]
```

Run: `pytest tests/test_conditional_prompts.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit protocol and prompt artifacts**

```bash
git add configs/conditional.yaml src/dlb/conditional_prompts.py \
  scripts/build_conditional_prompts.py scripts/verify_conditional_prompts.py \
  tests/test_conditional_prompts.py
git commit -m "feat: add deterministic conditional prompt manifests"
```

### Task 2: Conditional sample schema and atomic I/O

**Files:**

- Modify: `src/dlb/schema.py`
- Modify: `src/dlb/io.py`
- Create: `tests/test_conditional_schema.py`
- Create: `tests/test_conditional_io.py`

**Interfaces:**

- Consumes: `PromptRecord` and protocol lengths from Task 1.
- Produces: `ConditionalSampleRecord`, `expected_conditional_schedule(prompt_count=1024, diversity_prompt_count=256, completions=5) -> Sequence[tuple[int, int]]`, `read_conditional_samples(path)`, `validate_conditional_samples(path, expected=2048, schedule=production_schedule, sequence_length=model_length, vocab_size=dataset_vocab_size)`, and `write_conditional_samples_atomic(path, records, expected=2048, schedule=production_schedule, sequence_length=model_length, vocab_size=dataset_vocab_size)`.

- [ ] **Step 1: Write failing schema-invariant tests**

```python
def test_conditional_record_binds_prefix_full_and_suffix_slices():
    record = ConditionalSampleRecord.model_validate(valid_conditional_record())
    assert record.full_token_ids[:64] == record.prefix_token_ids
    assert record.full_token_ids[64:] == record.continuation_token_ids
    assert record.prefix_exact_match is True


@pytest.mark.parametrize("mutation,message", [
    (lambda value: value.update(prefix_exact_match=False), "prefix_exact_match"),
    (lambda value: value["full_token_ids"].__setitem__(0, 999), "prefix"),
    (lambda value: value.update(reference_token_ids=[1] * 63), "64"),
])
def test_conditional_record_rejects_invalid_publication(mutation, message):
    value = valid_conditional_record()
    mutation(value)
    with pytest.raises(ValidationError, match=message):
        ConditionalSampleRecord.model_validate(value)
```

- [ ] **Step 2: Run the schema tests and verify they fail**

Run: `pytest tests/test_conditional_schema.py -q`

Expected: import fails because `ConditionalSampleRecord` is absent.

- [ ] **Step 3: Implement the separate record and canonical schedule**

```python
NonNegativeToken = Annotated[StrictInt, Field(ge=0)]


class ConditionalSampleRecord(StrictModel):
    sample_id: StrictInt = Field(ge=0)
    prompt_id: StrictInt = Field(ge=0, le=1023)
    completion_id: StrictInt = Field(ge=0, le=4)
    source_index: StrictInt = Field(ge=0)
    prefix_token_ids: list[NonNegativeToken]
    continuation_token_ids: list[NonNegativeToken] = Field(min_length=1)
    reference_token_ids: list[NonNegativeToken]
    full_token_ids: list[NonNegativeToken] = Field(min_length=65)
    prefix_text: str
    continuation_text: str
    reference_text: str
    full_text: str
    seed: StrictInt
    generation_seconds: Annotated[FiniteFloat, Field(ge=0)]
    prefix_exact_match: Literal[True]

    @model_validator(mode="after")
    def validate_slices(self):
        if len(self.prefix_token_ids) != 64 or len(self.reference_token_ids) != 64:
            raise ValueError("prefix and reference must each contain 64 tokens")
        if self.full_token_ids[:64] != self.prefix_token_ids:
            raise ValueError("full token prefix differs from fixed prefix")
        if self.full_token_ids[64:] != self.continuation_token_ids:
            raise ValueError("continuation differs from full token suffix")
        return self


def expected_conditional_schedule(
    prompt_count: int = 1024,
    diversity_prompt_count: int = 256,
    completions: int = 5,
) -> list[tuple[int, int]]:
    return [(prompt, 0) for prompt in range(prompt_count)] + [
        (prompt, completion)
        for completion in range(1, completions)
        for prompt in range(diversity_prompt_count)
    ]
```

- [ ] **Step 4: Implement streaming conditional I/O and verify exact schedule**

The conditional validator must parse JSON with duplicate-key rejection, require `sample_id == line index`, require `(prompt_id, completion_id)` to equal the same index in `expected_conditional_schedule()`, require the dataset model length (128 for LM1B or 1,024 for OWT), validate token bounds against the supplied `vocab_size`, and never call or weaken `validate_samples`.

```python
def test_writer_rejects_duplicate_schedule_entry_without_replacing_final(tmp_path):
    final = tmp_path / "samples.jsonl"
    final.write_text("old\n", encoding="utf-8")
    records = production_records()
    records[1024]["prompt_id"] = 1
    with pytest.raises(SampleValidationError, match="expected prompt/completion"):
        write_conditional_samples_atomic(final, records, expected=2048)
    assert final.read_text(encoding="utf-8") == "old\n"
```

Run: `pytest tests/test_conditional_schema.py tests/test_conditional_io.py tests/test_schema.py tests/test_io.py -q`

Expected: all conditional and unconditional schema/I/O tests pass.

- [ ] **Step 5: Commit the artifact contract**

```bash
git add src/dlb/schema.py src/dlb/io.py \
  tests/test_conditional_schema.py tests/test_conditional_io.py
git commit -m "feat: add conditional sample artifact contract"
```

### Task 3: Conditional matrix and shared runner mode

**Files:**

- Create: `src/dlb/conditional_matrix.py`
- Modify: `src/dlb/runner.py`
- Create: `tests/test_conditional_matrix.py`
- Create: `tests/test_conditional_runner.py`

**Interfaces:**

- Consumes: `ConditionalProtocol`, verified prompt manifests, conditional I/O, existing registry/checkpoint provenance, and existing `SampleAdapter`.
- Produces: `ConditionalMatrixTask`, `build_conditional_matrix(registry, root=None, protocol=None) -> list[ConditionalMatrixTask]`, schema `dlb-conditional-generation-matrix-v1`, and conditional `RunRequest` fields passed unchanged to adapters.

- [ ] **Step 1: Write failing matrix parity and path-isolation tests**

```python
def test_conditional_matrix_matches_supported_unconditional_tasks(registry, tmp_path):
    ordinary = build_matrix(registry, root=tmp_path)
    conditional = build_conditional_matrix(registry, root=tmp_path)
    assert len(conditional) == len(ordinary) == 132
    assert [(t.model, t.dataset, t.steps) for t in conditional] == [
        (t.model, t.dataset, t.steps) for t in ordinary
    ]
    assert all("/results/conditional/" in task.sample_dir for task in conditional)
    assert all(task.sample_count == 2048 for task in conditional)


def test_conditional_unsupported_inventory_is_rdlm_owt(registry):
    assert conditional_unsupported_inventory(registry) == [{
        "status": "unsupported", "model": "rdlm", "dataset": "owt",
        "category": "few", "reason": registry.models["rdlm"].datasets["owt"].reason,
    }]
```

- [ ] **Step 2: Write failing request identity and publication tests**

```python
def test_unconditional_identity_has_no_conditional_keys(base_request):
    identity = _identity(base_request, ["python", "sample.py"])
    assert "generation_mode" not in identity
    assert "conditioning_manifest_sha256" not in identity


def test_conditional_identity_binds_manifest_and_schedule(conditional_request):
    identity = _identity(conditional_request, ["python", "sample.py"])
    assert identity["generation_mode"] == "conditional_prefix"
    assert identity["conditioning_manifest_sha256"] == "a" * 64
    assert identity["prefix_length"] == 64
    assert identity["completion_schedule"] == "c0:p0-1023;c1-4:p0-255"
```

- [ ] **Step 3: Implement the separate matrix**

```python
CONDITIONAL_MATRIX_SCHEMA = "dlb-conditional-generation-matrix-v1"

@dataclass(frozen=True)
class ConditionalMatrixTask:
    task_id: str
    category: str
    model: str
    dataset: str
    steps: int
    sample_count: Literal[2048]
    seed: int
    environment: str
    adapter: str
    source: str
    provenance: str
    protocol: Literal["c64_zs_v1"]
    conditioning_manifest: str
    conditioning_manifest_sha256: str
    sample_dir: str
    metrics_path: str
    timing_path: str
```

Derive model/dataset/step ordering from `build_matrix`, replace only protocol-specific columns and paths, validate the two prompt manifests first, and reuse the existing unsupported inventory writer with a conditional schema header.

- [ ] **Step 4: Extend `RunRequest` and branch publication without changing unconditional behavior**

```python
@dataclass(frozen=True)
class RunRequest:
    # existing fields remain in their current order
    generation_mode: Literal["unconditional", "conditional_prefix"] = "unconditional"
    conditioning_manifest: str | None = None
    conditioning_manifest_sha256: str | None = None
    conditioning_config_sha256: str | None = None
    prefix_length: int | None = None
    evaluation_continuation_length: int | None = None
    prompt_count: int | None = None
    diversity_prompt_count: int | None = None
    completions_per_diversity_prompt: int | None = None
    completion_schedule: str | None = None
```

`_resolve_request` must call `verify_prompts` for conditional requests and reject conditional fields on unconditional requests. `_identity` must add conditional keys only when `generation_mode == "conditional_prefix"`. `run_experiment` selects `validate_samples`/`write_samples_atomic` for unconditional mode and `validate_conditional_samples`/`write_conditional_samples_atomic` for conditional mode. Add matching CLI flags including `--generation-mode`, `--conditioning-manifest`, and `--conditioning-config`.

Run: `pytest tests/test_conditional_matrix.py tests/test_conditional_runner.py tests/test_matrix.py tests/test_runner.py -q`

Expected: all tests pass and the old runner fixtures produce byte-for-byte equivalent unconditional identity dictionaries.

- [ ] **Step 5: Commit runner and matrix support**

```bash
git add src/dlb/conditional_matrix.py src/dlb/runner.py \
  tests/test_conditional_matrix.py tests/test_conditional_runner.py
git commit -m "feat: add conditional matrix and runner mode"
```

### Task 4: Shared hard-prefix runtime primitives

**Files:**

- Create: `src/dlb/adapters/conditional_runtime.py`
- Create: `tests/test_conditional_runtime.py`

**Interfaces:**

- Consumes: verified `PromptRecord` rows and tensors produced by model-specific samplers.
- Produces: `ConditioningBatch`, `load_conditioning_batch(manifest_path, expected_manifest_sha256, completion_id, prompt_start, batch_size, device, vocab_size) -> ConditioningBatch`, `clamp_token_prefix(state, prefix_ids)`, `clamp_vocab_prefix(state, prefix_ids)`, `clamp_embedding_prefix(state, clean_embeddings)`, `token_project_fn(prefix_ids)`, `vocab_project_fn(prefix_ids)`, `embedding_project_fn(clean_embeddings)`, and `patched_attribute(owner, name, replacement)`.

- [ ] **Step 1: Write failing clamp and restoration tests**

```python
def test_token_projector_clamps_every_call():
    prefix = torch.tensor([[4, 5], [6, 7]])
    project = token_project_fn(prefix)
    state = torch.tensor([[99, 99, 8], [99, 99, 9]])
    assert project(state).tolist() == [[4, 5, 8], [6, 7, 9]]
    state[:, :2] = 88
    assert project(state).tolist() == [[4, 5, 8], [6, 7, 9]]


def test_scoped_patch_restores_even_after_sampler_error():
    owner = FakeOwner()
    original = owner.update
    with pytest.raises(RuntimeError):
        with patched_attribute(owner, "update", lambda *_: (_ for _ in ()).throw(RuntimeError())):
            owner.update(None)
    assert owner.update.__func__ is original.__func__
```

- [ ] **Step 2: Run tests and verify the runtime module is missing**

Run: `pytest tests/test_conditional_runtime.py -q`

Expected: import failure for `dlb.adapters.conditional_runtime`.

- [ ] **Step 3: Implement shape-safe token, vocabulary, and embedding projection**

```python
@dataclass(frozen=True)
class ConditioningBatch:
    prompt_ids: Sequence[int]
    source_indices: Sequence[int]
    prefix_token_ids: object
    reference_token_ids: object


def clamp_token_prefix(state, prefix_ids):
    if state.ndim != 2 or prefix_ids.ndim != 2:
        raise ValueError("token state and prefix must be rank two")
    if state.shape[0] != prefix_ids.shape[0] or state.shape[1] < prefix_ids.shape[1]:
        raise ValueError("token state is incompatible with prefix batch")
    result = state.clone()
    result[:, : prefix_ids.shape[1]] = prefix_ids.to(device=state.device, dtype=state.dtype)
    return result


def clamp_vocab_prefix(state, prefix_ids):
    clean = torch.nn.functional.one_hot(prefix_ids, num_classes=state.shape[-1]).to(state)
    result = state.clone()
    result[:, : prefix_ids.shape[1], :] = clean
    return result


def clamp_embedding_prefix(state, clean_embeddings):
    result = state.clone()
    result[:, : clean_embeddings.shape[1], :] = clean_embeddings.to(state)
    return result
```

All functions must reject device/batch/sequence/vocabulary mismatches, preserve suffix bytes, and avoid in-place mutation of caller-owned tensors. Projector closures expose `conditioning_implementation` metadata and are safe for repeated calls.

- [ ] **Step 4: Add exact prompt-batch schedule tests and pass them**

`load_conditioning_batch` accepts a prompt JSONL path, `completion_id`, `start_prompt`, `batch_size`, device, and tokenizer vocabulary. It rejects completion 0 beyond prompt 1023, completions 1–4 beyond prompt 255, and any batch that crosses its schedule boundary.

Run: `pytest tests/test_conditional_runtime.py -q`

Expected: token, vocabulary, embedding, schedule, and restoration cases pass.

- [ ] **Step 5: Commit runtime primitives**

```bash
git add src/dlb/adapters/conditional_runtime.py tests/test_conditional_runtime.py
git commit -m "feat: add hard-prefix conditioning runtime"
```

### Task 5: Teacher-family adapters for MDLM, Duo, FLM/FMLM, and CANDI

**Files:**

- Modify: `src/dlb/adapters/base.py`
- Modify: `src/dlb/adapters/capture.py`
- Modify: `src/dlb/adapters/mdlm.py`
- Modify: `src/dlb/adapters/duo.py`
- Modify: `src/dlb/adapters/flm.py`
- Modify: `src/dlb/adapters/candi.py`
- Create: `tests/test_conditional_teacher_adapters.py`
- Modify: `tests/test_capture.py`

**Interfaces:**

- Consumes: conditional request fields and runtime projectors from Tasks 3–4.
- Produces: uniform capture flags, `install_teacher_conditioning(owner, family, batch, tokenizer)`, and conditional capture records containing prompt/source/completion identity plus full generated token IDs.

- [ ] **Step 1: Write failing command serialization tests for every teacher family**

```python
@pytest.mark.parametrize("model", ["flm", "fmlm", "duo", "duo-dcd", "mdlm", "candi"])
def test_teacher_command_serializes_complete_conditioning_contract(model, conditional_request, adapter_for):
    command = adapter_for(model).build_command(conditional_request(model), RUN_DIR)
    assert option(command, "--generation-mode") == "conditional_prefix"
    assert option(command, "--conditioning-manifest") == conditional_request(model).conditioning_manifest
    assert option(command, "--prefix-length") == "64"
    assert option(command, "--completion-schedule") == "c0:p0-1023;c1-4:p0-255"
```

- [ ] **Step 2: Write fake-sampler tests that observe the prefix at every update**

```python
def test_mdlm_prior_and_every_ancestral_state_keep_real_prefix(fake_mdlm, batch):
    with install_teacher_conditioning(fake_mdlm, "mdlm", batch, fake_tokenizer):
        output = fake_mdlm.sample()
    assert fake_mdlm.observed_prefixes == [batch.prefix.tolist()] * fake_mdlm.step_count
    assert output[:, :64].tolist() == batch.prefix.tolist()


def test_duo_clamps_before_and_after_every_ancestral_update(fake_duo, batch):
    with install_teacher_conditioning(fake_duo, "duo", batch, fake_tokenizer):
        output = fake_duo.sample()
    assert all(prefix == batch.prefix.tolist() for prefix in fake_duo.before_update)
    assert all(prefix == batch.prefix.tolist() for prefix in fake_duo.after_update)


def test_flm_forward_always_receives_clean_prefix_one_hot(fake_flm, batch):
    with install_teacher_conditioning(fake_flm, "flm", batch, fake_tokenizer):
        fake_flm.sample()
    assert all(torch.equal(value, batch.clean_one_hot) for value in fake_flm.forward_prefixes)
```

- [ ] **Step 3: Extend the capture invocation and adapter output conversion**

Add these exact capture-only arguments: `--generation-mode`, `--conditioning-manifest`, `--conditioning-manifest-sha256`, `--prefix-length`, `--prompt-count`, `--diversity-prompt-count`, `--completions-per-diversity-prompt`, and `--completion-schedule`. In conditional mode, `_capture_teacher` iterates the canonical schedule without ceiling batches and emits one raw capture item per scheduled prompt/completion pair with `prompt_id`, `completion_id`, `source_index`, `prefix_token_ids`, `reference_token_ids`, `full_token_ids`, and batch elapsed time divided by the exact batch size.

`BaseTeacherAdapter.convert_outputs` must branch to a `_convert_conditional_capture_outputs` method that decodes prefix/reference/full/continuation separately with the locked canonical tokenizer and constructs `ConditionalSampleRecord`; its unconditional branch remains unchanged.

- [ ] **Step 4: Implement family-specific hard conditioning**

```python
if family == "mdlm":
    patch_prior_to_insert_clean_prefix(owner, batch.prefix_token_ids)
    patch_ancestral_update_with_token_pre_post_clamp(owner, batch.prefix_token_ids)
elif family == "duo":
    patch_ancestral_update_with_token_pre_post_clamp(owner, batch.prefix_token_ids)
elif family in {"flm", "fmlm"}:
    patch_forward_with_vocab_prefix(owner, batch.prefix_token_ids)
elif family == "candi":
    kwargs["prompt_tokens"] = batch.prefix_token_ids
    kwargs["prompt_mask"] = torch.ones_like(batch.prefix_token_ids, dtype=torch.bool)
```

All paths perform a final token clamp plus exact assertion. Metadata is `native_projection` for CANDI and `zero_shot_runtime_projection` for MDLM/Duo/FLM/FMLM. FMLM metadata also contains `paper_conditional_training_reproduced: false`.

Run: `pytest tests/test_conditional_teacher_adapters.py tests/test_capture.py tests/test_teacher_adapters.py -q`

Expected: conditional tests pass and all existing teacher/capture tests remain green.

- [ ] **Step 5: Commit teacher-family integration**

```bash
git add src/dlb/adapters/base.py src/dlb/adapters/capture.py \
  src/dlb/adapters/mdlm.py src/dlb/adapters/duo.py \
  src/dlb/adapters/flm.py src/dlb/adapters/candi.py \
  tests/test_conditional_teacher_adapters.py tests/test_capture.py
git commit -m "feat: condition teacher-family samplers on fixed prefixes"
```

### Task 6: Native projected SDTT and Di4C samplers

**Files:**

- Modify: `src/dlb/adapters/sdtt.py`
- Modify: `src/dlb/adapters/di4c.py`
- Modify: `adapters/_distilled_runtime.py`
- Modify: `adapters/sample_sdtt.py`
- Modify: `adapters/sample_di4c.py`
- Create: `tests/test_conditional_distilled_adapters.py`

**Interfaces:**

- Consumes: conditional request/capture contract and `token_project_fn`/`vocab_project_fn`.
- Produces: native `project_fn` wiring for SDTT and Di4C, exact conditional capture JSON, and benchmark callbacks that include projection time.

- [ ] **Step 1: Write failing SDTT/Di4C CLI and native-projection tests**

```python
@pytest.mark.parametrize("model", ["mdlm-sdtt", "duo-di4c", "mdlm-di4c"])
def test_distilled_command_passes_manifest_sha_and_schedule(model, request, adapter_for):
    command = adapter_for(model).build_command(request(model), RUN_DIR)
    assert option(command, "--conditioning-manifest-sha256") == "a" * 64
    assert option(command, "--completion-schedule") == "c0:p0-1023;c1-4:p0-255"


def test_sdtt_native_project_fn_is_called_at_every_step(fake_sdtt_runtime, prefix):
    result = fake_sdtt_runtime.sample_with_prefix(prefix, steps=7)
    assert fake_sdtt_runtime.project_calls == 7
    assert torch.equal(result[:, :64], prefix)
```

- [ ] **Step 2: Run focused tests and confirm missing flags/projection failures**

Run: `pytest tests/test_conditional_distilled_adapters.py -q`

Expected: assertions fail because current wrappers have no conditional arguments.

- [ ] **Step 3: Add authenticated prompt loading to the server runtime**

`adapters/_distilled_runtime.py` must verify prompt-file and manifest SHA-256 before model materialization, validate tokenizer ID/revision and vocabulary bounds, materialize only the exact prompt batch, and expose:

```python
def load_conditional_batch(
    manifest_path: Path,
    expected_manifest_sha256: str,
    *,
    completion_id: int,
    prompt_start: int,
    prompt_count: int,
    device: str,
    vocab_size: int,
) -> ConditioningBatch:
    return load_conditioning_batch(
        manifest_path,
        expected_manifest_sha256,
        completion_id=completion_id,
        prompt_start=prompt_start,
        batch_size=prompt_count,
        device=device,
        vocab_size=vocab_size,
    )
```

- [ ] **Step 4: Pass native `project_fn` throughout both sampler entrypoints**

For masked states, the native projection writes prefix token IDs into the discrete state. For categorical/one-hot states, it writes the clean prefix one-hot endpoint. Pass the closure to the pinned upstream sampler's existing `project_fn` parameter for every reverse step, assert the result prefix, and write conditional capture fields in canonical schedule order. `benchmark_model` must time the same closure inside the measured generation callback.

Run: `pytest tests/test_conditional_distilled_adapters.py tests/test_distilled_adapters.py tests/test_timing.py -q`

Expected: new and existing distilled adapter tests pass.

- [ ] **Step 5: Commit native distilled projections**

```bash
git add src/dlb/adapters/sdtt.py src/dlb/adapters/di4c.py \
  adapters/_distilled_runtime.py adapters/sample_sdtt.py adapters/sample_di4c.py \
  tests/test_conditional_distilled_adapters.py
git commit -m "feat: add native prefix projection to distilled samplers"
```

### Task 7: LangFlow embedding projection and RDLM manifold projection

**Files:**

- Modify: `src/dlb/adapters/langflow.py`
- Modify: `src/dlb/adapters/rdlm.py`
- Modify: `src/dlb/adapters/capture.py`
- Modify: `adapters/sample_langflow.py`
- Create: `tests/test_conditional_continuous_adapters.py`

**Interfaces:**

- Consumes: `embedding_project_fn`, `vocab_project_fn`, conditional request fields, and canonical prompt batches.
- Produces: `install_langflow_conditioning(model, batch, tokenizer)` and `run_rdlm_conditional(model, batch, tokenizer, sampling_config)`, both with final token exact-match assertions.

- [ ] **Step 1: Write failing continuous-state observation tests**

```python
def test_langflow_clamps_noisy_and_self_conditioning_before_every_forward(fake_langflow, batch):
    with install_langflow_conditioning(fake_langflow, batch):
        output = fake_langflow.generate()
    assert len(fake_langflow.noisy_prefixes) == fake_langflow.step_count
    assert all(torch.equal(x, batch.clean_embeddings) for x in fake_langflow.noisy_prefixes)
    assert all(torch.equal(x, batch.clean_embeddings) for x in fake_langflow.self_cond_prefixes)
    assert torch.equal(output[:, :64], batch.prefix_token_ids)


def test_rdlm_passes_clean_manifold_projection_to_sampling_factory(fake_rdlm, batch):
    output = run_rdlm_conditional(fake_rdlm, batch)
    assert fake_rdlm.proj_fn is not None
    assert fake_rdlm.proj_calls == fake_rdlm.step_count
    assert torch.equal(output[:, :64], batch.prefix_token_ids)
```

- [ ] **Step 2: Run focused tests and verify current adapters fail**

Run: `pytest tests/test_conditional_continuous_adapters.py -q`

Expected: failures show no LangFlow conditioning hook and no RDLM projection.

- [ ] **Step 3: Implement LangFlow clean-embedding projection**

Before each LangFlow forward evaluation, derive clean prefix embeddings from the loaded model embedding table and replace positions `:64` in both `noisy_embeds` and non-null `x_self_cond`. The suffix tensor must be byte-identical before/after the hook. Restore the original forward method after sampling and record `conditioning_implementation: zero_shot_runtime_projection` and `paper_conditional_training_reproduced: false`.

- [ ] **Step 4: Implement RDLM native manifold projection**

Construct the clean one-hot prefix endpoint in RDLM's vocabulary/manifold coordinates and pass `proj_fn=project` to the pinned `get_sampling_fn` call. The closure clamps every call, preserves suffix coordinates, and the decoded output receives a final token-level assertion. Record `conditioning_implementation: native_projection`.

Run: `pytest tests/test_conditional_continuous_adapters.py tests/test_continuous_adapters.py tests/test_capture.py -q`

Expected: all continuous, capture, LangFlow, and RDLM tests pass.

- [ ] **Step 5: Commit continuous/manifold integration**

```bash
git add src/dlb/adapters/langflow.py src/dlb/adapters/rdlm.py \
  src/dlb/adapters/capture.py adapters/sample_langflow.py \
  tests/test_conditional_continuous_adapters.py
git commit -m "feat: condition flow and manifold samplers on prefixes"
```

### Task 8: Conditional quality and diversity evaluation

**Files:**

- Create: `evaluation/conditional_perplexity.py`
- Create: `evaluation/mauve_score.py`
- Create: `evaluation/conditional_evaluate.py`
- Modify: `evaluation/self_bleu.py`
- Modify: `evaluation/unigram_entropy.py`
- Create: `tests/test_conditional_perplexity.py`
- Create: `tests/test_mauve_score.py`
- Create: `tests/test_conditional_evaluate.py`

**Interfaces:**

- Consumes: 2,048 validated `ConditionalSampleRecord` objects and pinned local GPT-2-large assets.
- Produces: `compute_conditional_ppl(prefixes, continuations, model, tokenizer, batch_size=8)`, `compute_mauve_score(generated, references, assets, device_id)`, `compute_grouped_self_bleu(token_rows_by_prompt)`, `conditional_entropy(records, slice_length=64)`, and schema `dlb-conditional-metrics-v1`.

- [ ] **Step 1: Write failing suffix-only causal-loss tests**

```python
def test_conditional_ppl_masks_prefix_targets_and_counts_suffix_only(fake_scorer, tokenizer):
    result = compute_conditional_ppl(
        prefixes=["alpha beta"], continuations=["gamma delta"],
        model=fake_scorer, tokenizer=tokenizer, batch_size=1,
    )
    assert fake_scorer.target_masks == [[0, 0, 1, 1]]
    assert result.valid_token_count == 2


def test_conditional_ppl_tokenizes_prefix_and_suffix_separately(spy_tokenizer, fake_scorer):
    compute_conditional_ppl(["prefix"], [" suffix"], fake_scorer, spy_tokenizer)
    assert spy_tokenizer.calls == [(["prefix"], False), ([" suffix"], False)]
```

- [ ] **Step 2: Implement target-mask scoring**

```python
def compute_conditional_ppl(prefixes, continuations, model, tokenizer, *, batch_size=8):
    prefix_rows = tokenize_without_special_tokens(tokenizer, prefixes)
    suffix_rows = tokenize_without_special_tokens(tokenizer, continuations)
    input_rows = [prefix + suffix for prefix, suffix in zip(prefix_rows, suffix_rows, strict=True)]
    target_masks = [[0] * len(prefix) + [1] * len(suffix) for prefix, suffix in zip(prefix_rows, suffix_rows, strict=True)]
    return score_causal_targets(model, input_rows, target_masks, batch_size=batch_size)
```

The scorer shifts logits and target masks together, rejects empty/truncated suffix targets, and limits the combined sequence to 1,024 evaluator tokens. Run the same computation for ground-truth references.

- [ ] **Step 3: Write and pass pinned MAUVE tests**

```python
def test_mauve_uses_local_snapshot_and_aligned_completion_zero(monkeypatch, assets):
    call = install_fake_mauve(monkeypatch)
    result = compute_mauve_score(GENERATED_1024, REFERENCES_1024, assets, device_id=0)
    assert call["featurize_model_name"] == str(assets.model_path)
    assert call["p_text"] == REFERENCES_1024
    assert call["q_text"] == GENERATED_1024
    assert result.sample_count == 1024
```

The wrapper imports `mauve` lazily, requires `mauve-text==0.3.0`, sets Hugging Face offline variables, uses the pinned local GPT-2-large snapshot, and records seed, bucket count, model revision, and sample count.

- [ ] **Step 4: Implement grouped diversity, entropy, exact-prefix gating, and CLI artifact**

`conditional_evaluate` retokenizes each continuation text independently with the pinned local GPT-2 evaluator tokenizer, records that evaluator revision in the metric artifact, and passes exactly five 64-token-truncated GPT-2 rows for each prompt `0..255` to `compute_grouped_self_bleu`; each candidate uses its other four same-prompt rows as references with BLEU-4 equal weights and method-1 smoothing. `conditional_entropy` uses the first 64 model-native continuation IDs and no prefix tokens. `conditional_evaluate.evaluate` rejects any schedule or prefix mismatch before loading evaluator models and publishes:

```python
{
    "schema": "dlb-conditional-metrics-v1",
    "protocol": "c64_zs_v1",
    "sample_count": 2048,
    "quality_sample_count": 1024,
    "diversity_prompt_count": 256,
    "samples_sha256": sha256_file(samples_path),
    "prompt_manifest_sha256": manifest_sha,
    "metrics": {
        "conditional_generation_perplexity": generated_ppl,
        "reference_conditional_perplexity": reference_ppl,
        "mauve": mauve_result,
        "grouped_self_bleu": self_bleu_result,
        "continuation_entropy": entropy_result,
        "prefix_exact_match_rate": 1.0,
    },
}
```

Run: `pytest tests/test_conditional_perplexity.py tests/test_mauve_score.py tests/test_conditional_evaluate.py tests/test_perplexity.py tests/test_self_bleu.py tests/test_entropy.py -q`

Expected: all conditional and unconditional evaluation tests pass.

- [ ] **Step 5: Commit conditional evaluation**

```bash
git add evaluation/conditional_perplexity.py evaluation/mauve_score.py \
  evaluation/conditional_evaluate.py evaluation/self_bleu.py \
  evaluation/unigram_entropy.py tests/test_conditional_perplexity.py \
  tests/test_mauve_score.py tests/test_conditional_evaluate.py
git commit -m "feat: evaluate conditional quality and diversity"
```

### Task 9: Conditional timing protocol

**Files:**

- Create: `src/dlb/conditional_benchmarking.py`
- Modify: `src/dlb/benchmarking.py`
- Modify: `src/dlb/timing.py`
- Modify: `src/dlb/adapters/base.py`
- Create: `tests/test_conditional_timing.py`

**Interfaces:**

- Consumes: prompt 0, conditional matrix task, adapter `benchmark_hook`, and existing CUDA timing primitives.
- Produces: schema `dlb-conditional-timing-v1` with 5 warmups, 32 repeats, batch size 1, full model canvas, and conditional provenance.

- [ ] **Step 1: Write failing timer-boundary and metadata tests**

```python
def test_conditional_timer_moves_prompt_before_first_clock(monkeypatch, fake_adapter):
    events = install_timing_spies(monkeypatch, fake_adapter)
    benchmark_conditional_cell(fake_task(), fake_adapter)
    assert events.index("prompt.to:cuda") < events.index("clock:start")
    assert events.count("generate_with_projection") == 5 + 32


def test_owt_timing_records_full_canvas_not_metric_slice():
    metadata = conditional_timing_metadata(fake_task(dataset="owt"))
    assert metadata["model_length"] == 1024
    assert metadata["generated_suffix_length"] == 960
    assert metadata["evaluation_continuation_length"] == 64
```

- [ ] **Step 2: Run focused timing tests and verify missing conditional module**

Run: `pytest tests/test_conditional_timing.py -q`

Expected: import failure for `dlb.conditional_benchmarking`.

- [ ] **Step 3: Implement conditional benchmark rendering and publication**

Use a `RunRequest` with `generation_mode="conditional_prefix"`, `sample_count=1`, `prompt_count=1`, `diversity_prompt_count=0`, and `completions_per_diversity_prompt=1`. Load prompt 0 and move it to the target GPU before passing the measured closure to `benchmark`. Reuse the current synchronization and raw-duration logic with `warmups=5`, `repeats=32`; projection/clamp calls stay inside `generate_one`.

```python
metadata.update({
    "schema": "dlb-conditional-timing-v1",
    "protocol": "c64_zs_v1",
    "prompt_id": 0,
    "prefix_length": 64,
    "batch_size": 1,
    "warmups": 5,
    "repeats": 32,
    "conditioning_manifest_sha256": task.conditioning_manifest_sha256,
})
```

- [ ] **Step 4: Prove unconditional timing remains unchanged**

Run: `pytest tests/test_conditional_timing.py tests/test_timing.py -q`

Expected: conditional timing passes; existing timing schema, warmup/repeat behavior, and benchmark command tests remain green.

- [ ] **Step 5: Commit conditional timing**

```bash
git add src/dlb/conditional_benchmarking.py src/dlb/benchmarking.py \
  src/dlb/timing.py src/dlb/adapters/base.py tests/test_conditional_timing.py
git commit -m "feat: benchmark conditional generation latency"
```

### Task 10: Strict conditional aggregation

**Files:**

- Create: `src/dlb/conditional_aggregate.py`
- Create: `scripts/aggregate_conditional_results.py`
- Create: `tests/test_conditional_aggregate.py`

**Interfaces:**

- Consumes: canonical conditional matrix, unsupported inventory, prompts, sample/run metadata, metrics, and timing artifacts.
- Produces: `ConditionalAggregateReport`, `aggregate_conditional(root, strict=True, partial=False, output_dir=None)`, `results/conditional/summary/results.csv`, `failures.csv`, `unsupported.csv`, `provenance.json`, and `README.md`.

- [ ] **Step 1: Write failing completeness and provenance tests**

```python
def test_conditional_aggregate_accepts_complete_bound_cell(complete_conditional_cell):
    report = aggregate_conditional(complete_conditional_cell.root, strict=True)
    assert report.complete is True
    assert report.rows[0]["prefix_exact_match_rate"] == 1.0


@pytest.mark.parametrize("mutation,message", [
    (remove_one_sample, "2048"),
    (duplicate_completion, "schedule"),
    (change_prompt_sha, "prompt manifest"),
    (change_metric_sample_sha, "sample artifact"),
    (change_timing_steps, "step count"),
])
def test_conditional_aggregate_fails_closed(complete_conditional_cell, mutation, message):
    mutation(complete_conditional_cell)
    with pytest.raises(IncompleteConditionalMatrixError, match=message):
        aggregate_conditional(complete_conditional_cell.root, strict=True)
```

- [ ] **Step 2: Run focused tests and confirm missing aggregator**

Run: `pytest tests/test_conditional_aggregate.py -q`

Expected: import failure for `dlb.conditional_aggregate`.

- [ ] **Step 3: Implement cell linkage and strict matrix accounting**

For each of the 132 tasks, validate sample JSONL first, then require run identity, checkpoint/source/tokenizer/prompt SHA linkage, metric `samples_sha256`, timing protocol/task identity, finite metric values, exact record counts, and prefix rate 1.0. Collect failures with `task_id`, stage, path, and message. Strict mode raises when any task fails; partial mode writes a visibly partial report.

- [ ] **Step 4: Publish deterministic summary files atomically**

```python
@dataclass(frozen=True)
class ConditionalAggregateReport:
    complete: bool
    rows: Sequence[dict[str, object]]
    failures: Sequence[dict[str, object]]
    unsupported: Sequence[dict[str, object]]
```

Sort result rows by model, dataset, and integer steps. CSV columns include conditional PPL, reference PPL, MAUVE, grouped Self-BLEU, entropy, prefix exact match, median/mean latency, checkpoint provenance, and conditioning implementation label.

Run: `pytest tests/test_conditional_aggregate.py tests/test_aggregate.py -q`

Expected: new strict/partial tests and existing unconditional aggregation tests pass.

- [ ] **Step 5: Commit aggregation**

```bash
git add src/dlb/conditional_aggregate.py scripts/aggregate_conditional_results.py \
  tests/test_conditional_aggregate.py
git commit -m "feat: aggregate conditional benchmark results"
```

### Task 11: One-cell, serial, four-GPU, and smoke launchers

**Files:**

- Create: `scripts/run_conditional_one.sh`
- Create: `scripts/run_conditional_all.sh`
- Create: `scripts/run_conditional_4gpu.sh`
- Create: `scripts/evaluate_conditional_all.sh`
- Create: `scripts/evaluate_conditional_4gpu.sh`
- Create: `scripts/benchmark_conditional_one.sh`
- Create: `scripts/benchmark_conditional_all.sh`
- Create: `scripts/benchmark_conditional_4gpu.sh`
- Create: `scripts/smoke_conditional_all.sh`
- Create: `scripts/smoke_conditional_4gpu.sh`
- Modify: `src/dlb/gpu_matrix.py`
- Create: `tests/test_conditional_scripts.py`

**Interfaces:**

- Consumes: conditional matrix and CLIs from Tasks 1–10.
- Produces: safe argv-only launchers with serial/resumable and four-GPU modes plus two-prompt/two-completion smoke coverage.

- [ ] **Step 1: Write failing shell contract and dry-run routing tests**

```python
def test_conditional_one_exports_project_src_and_complete_protocol(tmp_path):
    invocation = run_with_fake_conda("scripts/run_conditional_one.sh", tmp_path)
    assert invocation.module == "dlb.runner"
    assert invocation.option("--generation-mode") == "conditional_prefix"
    assert invocation.option("--num-samples") == "2048"
    assert invocation.option("--results-root").endswith("results/conditional")


def test_conditional_gpu_dry_run_routes_each_stage(tmp_path):
    commands = conditional_gpu_dry_run(tmp_path, stages=("generate", "evaluate", "benchmark"))
    assert [Path(command[0]).name for command in commands] == [
        "run_conditional_one.sh", "python", "benchmark_conditional_one.sh"
    ]
    assert "evaluation.conditional_evaluate" in commands[1]
```

- [ ] **Step 2: Run tests and verify missing scripts**

Run: `pytest tests/test_conditional_scripts.py -q`

Expected: failures because the conditional scripts do not exist.

- [ ] **Step 3: Implement safe one-cell and serial scripts**

Every script uses Bash arrays, validates integer/model/dataset arguments, quotes every path, regenerates or verifies the conditional matrix before launch, and records failures below `results/conditional/logs/`. The one-cell generator first calls runner `--validate-only`, then launches the registry-selected Conda environment with:

```bash
runner_args=(
  --root "$DLB_ROOT" --model "$model" --dataset "$dataset" --steps "$steps"
  --num-samples 2048 --seed 42 --generation-mode conditional_prefix
  --conditioning-config "$DLB_ROOT/configs/conditional.yaml"
  --conditioning-manifest "$manifest"
  --results-root "$DLB_ROOT/results/conditional"
)
```

- [ ] **Step 4: Implement four-GPU routing and smoke schedule**

Add `--protocol conditional` to `dlb.gpu_matrix`; this selects `read_conditional_matrix` and conditional scripts while the omitted/default value keeps existing unconditional routing. Smoke launchers use a separately labeled output root, two prompt IDs, completions 0 and 1, and step counts 1 plus each category's maximum; they never publish into production paths.

Run: `pytest tests/test_conditional_scripts.py tests/test_gpu_matrix.py tests/test_run_one_script.py -q`

Expected: all conditional script and existing GPU/script tests pass.

- [ ] **Step 5: Commit launchers**

```bash
git add scripts/run_conditional_one.sh scripts/run_conditional_all.sh \
  scripts/run_conditional_4gpu.sh scripts/evaluate_conditional_all.sh \
  scripts/evaluate_conditional_4gpu.sh scripts/benchmark_conditional_one.sh \
  scripts/benchmark_conditional_all.sh scripts/benchmark_conditional_4gpu.sh \
  scripts/smoke_conditional_all.sh scripts/smoke_conditional_4gpu.sh \
  src/dlb/gpu_matrix.py tests/test_conditional_scripts.py
git commit -m "feat: add conditional benchmark launchers"
```

### Task 12: Documentation, smoke fixtures, and full regression verification

**Files:**

- Modify: `README.md`
- Modify: `docs/06_generation_and_evaluation.md`
- Modify: `docs/09_four_gpu_local.md`
- Create: `tests/fixtures/conditional_samples.jsonl`
- Modify: tests from Tasks 1–11 only when full-suite failures expose a contract mismatch.

**Interfaces:**

- Consumes: every feature and command added in Tasks 1–11.
- Produces: operator instructions, zero-shot/OOD interpretation notes, deterministic fixture coverage, and complete verification evidence.

- [ ] **Step 1: Add a small deterministic fixture and end-to-end offline test**

```python
def test_conditional_offline_pipeline_from_prompts_to_partial_summary(tmp_path):
    build_fixture_prompts(tmp_path, prompt_count=2)
    run_fake_conditional_adapter(tmp_path, completions=2)
    evaluate_fixture_without_mauve(tmp_path, allow_partial=True)
    report = aggregate_conditional(tmp_path, strict=False, partial=True)
    assert len(report.rows) == 1
    assert report.rows[0]["prefix_exact_match_rate"] == 1.0
```

Run: `pytest tests/test_conditional_prompts.py tests/test_conditional_runner.py tests/test_conditional_evaluate.py tests/test_conditional_aggregate.py -q`

Expected: the offline prompt → fake generation → evaluation → partial aggregate path passes.

- [ ] **Step 2: Document exact server workflow and interpretation boundary**

Document these commands in order:

```bash
python scripts/build_conditional_prompts.py --root "$PWD" --dataset lm1b
python scripts/build_conditional_prompts.py --root "$PWD" --dataset owt
python scripts/verify_conditional_prompts.py --root "$PWD" --dataset all
bash scripts/smoke_conditional_4gpu.sh --gpus 0,1,2,3
bash scripts/run_conditional_4gpu.sh --gpus 0,1,2,3
bash scripts/evaluate_conditional_4gpu.sh --gpus 0,1,2,3
bash scripts/benchmark_conditional_4gpu.sh --gpus 0,1,2,3
python scripts/aggregate_conditional_results.py --root "$PWD"
```

State explicitly that FLM/FMLM/LangFlow conditioning is zero-shot/OOD projection, the comparison does not reproduce conditional fine-tuning, OWT generation uses a 960-token suffix canvas, and reported aligned quality metrics use the first 64 suffix tokens.

- [ ] **Step 3: Run formatting/syntax checks**

Run: `git diff --check`

Run: `python -m compileall -q src evaluation scripts`

Run: `for file in scripts/*conditional*.sh; do bash -n "$file"; done`

Expected: every command exits 0 with no output indicating an error.

- [ ] **Step 4: Run the complete regression suite**

Run: `pytest -q`

Expected: all existing unconditional tests and all conditional tests pass; no tests are skipped because of an accidental online dependency.

- [ ] **Step 5: Commit documentation and final verification assets**

```bash
git add README.md docs/06_generation_and_evaluation.md \
  docs/09_four_gpu_local.md tests/fixtures/conditional_samples.jsonl
git commit -m "docs: add conditional benchmark operating guide"
```

## Final server acceptance run

The local implementation is complete after Task 12. On the checkpoint/data server, run these acceptance gates in order and stop on the first failure:

1. `python scripts/verify_data.py --root "$PWD"`
2. `python scripts/verify_checkpoints.py --root "$PWD"`
3. `python scripts/verify_conditional_prompts.py --root "$PWD" --dataset all`
4. `bash scripts/smoke_conditional_4gpu.sh --gpus 0,1,2,3`
5. Inspect smoke artifacts for all sampler families at one and maximum steps.
6. `bash scripts/run_conditional_4gpu.sh --gpus 0,1,2,3`
7. `bash scripts/evaluate_conditional_4gpu.sh --gpus 0,1,2,3`
8. `bash scripts/benchmark_conditional_4gpu.sh --gpus 0,1,2,3`
9. `python scripts/aggregate_conditional_results.py --root "$PWD"`
10. Require `complete=true`, `rows=132`, `failures=0`, one RDLM/OWT unsupported record, and prefix exact-match rate `1.0` for every row.
