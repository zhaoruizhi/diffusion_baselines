"""Paper-locked server recipes for baseline training and distillation.

This module is intentionally importable without PyTorch.  Checkpoint tensor
loading happens lazily only when a real server distillation run is launched.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
from typing import Collection, Mapping, MutableMapping, Sequence

from dlb.io import atomic_json_write, ensure_safe_directory, sha256_file


class RecipeError(ValueError):
    """A requested training run does not satisfy the locked recipe contract."""


class RecipeProcessError(RecipeError):
    """An upstream process failed and its exit status must be preserved."""

    def __init__(self, message: str, exit_status: int) -> None:
        super().__init__(message)
        self.exit_status = exit_status


BACKBONE_TENSOR_KEYS = (
    "vocab_embed.embedding",
    "output_layer.linear.weight",
    "output_layer.linear.bias",
)
OPTIONAL_BACKBONE_STATE_KEYS = frozenset({"rotary_emb.inv_freq"})

SOURCE_COMMITS = {
    "flm": "a1918d5164e5038e37d0b7a4fb2010ce75b863b3",
    "duo": "7c9b498f5b717de064d6fad7e2509c866e6cb620",
    "mdlm": "c112c526d193436838c98d81455ee51f90309470",
    "candi": "cd57ae9eec98d6ac71cd52bdc50eeec8dfd70f91",
    "rdlm": "67443aa6a2d0fa981eb7c6105f9cbc563e59c5c1",
    "sdtt": "1150985e90b8f2d5749e4469d5154eff9ec922c4",
    "di4c": "ac61ff9fe8e85120f9e1d2a8c5a332f8b8353dd3",
}

TOKENIZER_REVISIONS = {
    "lm1b": ("bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594"),
    "owt": ("gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
}
SEQUENCE_LENGTHS = {"lm1b": 128, "owt": 1024}
BASE_VOCAB_SIZES = {"lm1b": 30_522, "owt": 50_257}


@dataclass(frozen=True)
class TrainingRecipe:
    model: str
    dataset: str
    source: str
    source_path: str
    source_commit: str
    entrypoint: str
    max_steps: int
    global_batch_size: int
    learning_rate: float
    warmup_steps: int
    optimizer: str
    precision: str
    sequence_length: int
    checkpoint_every: int
    default_per_device_batch_size: int
    evidence: str
    max_steps_override: str
    learning_rate_override: str
    output_override: str
    algorithm: str | None = None
    rounds: int | None = None
    steps_per_round: int | None = None
    teacher_family: str | None = None
    teacher_adapter: str | None = None
    distillation_loss: str | None = None
    sampling_step: int | None = None
    release_script_max_steps: int | None = None
    release_evidence: str | None = None


@dataclass(frozen=True)
class LaunchSpec:
    recipe: TrainingRecipe
    command: tuple[str, ...]
    cwd: Path
    entrypoint: Path
    data_path: Path
    output: Path
    teacher: Path | None
    adapted_teacher: Path | None
    devices: int
    nodes: int
    per_device_batch_size: int
    gradient_accumulation: int
    effective_global_batch_size: int
    seed: int
    resume: bool


def _base_recipe(
    model: str,
    dataset: str,
    *,
    source: str,
    entrypoint: str = "main.py",
    max_steps: int = 1_000_000,
    global_batch_size: int = 512,
    learning_rate: float = 3e-4,
    warmup_steps: int = 2_500,
    optimizer: str = "adam",
    precision: str = "bf16",
    checkpoint_every: int = 20_000,
    default_per_device_batch_size: int | None = None,
    evidence: str = "paper_v3_main",
    max_steps_override: str = "trainer.max_steps",
    learning_rate_override: str = "optim.lr",
    output_override: str = "checkpointing.save_dir",
    **values: object,
) -> TrainingRecipe:
    if dataset not in SEQUENCE_LENGTHS:
        raise RecipeError(f"unsupported dataset {dataset!r}")
    per_device = default_per_device_batch_size
    if per_device is None:
        per_device = 64 if dataset == "lm1b" else 16
    return TrainingRecipe(
        model=model,
        dataset=dataset,
        source=source,
        source_path=f"upstreams/{source}",
        source_commit=SOURCE_COMMITS[source],
        entrypoint=entrypoint,
        max_steps=max_steps,
        global_batch_size=global_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        optimizer=optimizer,
        precision=precision,
        sequence_length=SEQUENCE_LENGTHS[dataset],
        checkpoint_every=checkpoint_every,
        default_per_device_batch_size=per_device,
        evidence=evidence,
        max_steps_override=max_steps_override,
        learning_rate_override=learning_rate_override,
        output_override=output_override,
        **values,
    )


def load_recipe(model: str, dataset: str) -> TrainingRecipe:
    """Return the immutable paper/reference recipe for one training route."""

    if dataset not in SEQUENCE_LENGTHS:
        raise RecipeError(f"unsupported dataset {dataset!r}")
    if model == "flm":
        return _base_recipe(
            model,
            dataset,
            source="flm",
            algorithm="flm",
            release_script_max_steps=1_500_000,
            release_evidence="official_release_script",
        )
    if model == "duo":
        return _base_recipe(
            model,
            dataset,
            source="duo",
            algorithm="duo",
            evidence="paper_appendix_d" if dataset == "lm1b" else "official_release_script",
        )
    if model == "mdlm":
        return _base_recipe(
            model,
            dataset,
            source="mdlm",
            algorithm="mdlm",
            evidence="paper_appendix_d" if dataset == "lm1b" else "official_release_script",
        )
    if model == "candi":
        return _base_recipe(
            model,
            dataset,
            source="candi",
            algorithm="candi",
            evidence="paper_appendix_d" if dataset == "lm1b" else "project_fallback",
        )
    if model == "rdlm":
        if dataset != "lm1b":
            raise RecipeError("RDLM/OWT is unsupported by the official release")
        return _base_recipe(
            model,
            dataset,
            source="rdlm",
            algorithm="rdlm",
            max_steps_override="training.n_iters",
            learning_rate_override="optim.lr",
            output_override="hydra.run.dir",
            evidence="official_release_script",
        )
    if model == "fmlm":
        return _base_recipe(
            model,
            dataset,
            source="flm",
            max_steps=100_000,
            algorithm="fmlm",
            teacher_family="continuous_flm",
            distillation_loss="cross_entropy",
            release_script_max_steps=1_000_000,
            release_evidence="official_release_script",
        )
    if model == "duo_dcd":
        return _base_recipe(
            model,
            dataset,
            source="duo",
            max_steps=80_000,
            global_batch_size=128,
            learning_rate=6e-5,
            checkpoint_every=10_000,
            default_per_device_batch_size=16,
            algorithm="distillation",
            rounds=8,
            steps_per_round=10_000,
            teacher_family="uniform_duo",
            distillation_loss="kl_backward",
            evidence="paper_appendix_d",
        )
    if model == "mdlm_sdtt":
        return _base_recipe(
            model,
            dataset,
            source="sdtt",
            entrypoint="src/sdtt/main.py",
            max_steps=80_000,
            global_batch_size=128,
            learning_rate=6e-5,
            checkpoint_every=10_000,
            default_per_device_batch_size=16,
            algorithm="multi-round-sdtt",
            rounds=8,
            steps_per_round=10_000,
            teacher_family="masked_mdlm",
            teacher_adapter="masked_to_absorbing",
            distillation_loss="kl_backward",
            sampling_step=70_000,
            evidence="paper_appendix_d",
        )
    if model in {"duo_di4c", "mdlm_di4c"}:
        teacher_family = "uniform_duo" if model == "duo_di4c" else "masked_mdlm"
        adapter = (
            "uniform_to_absorbing"
            if teacher_family == "uniform_duo"
            else "masked_to_absorbing"
        )
        sampling_step = 20_000 if dataset == "lm1b" else 50_000
        return _base_recipe(
            model,
            dataset,
            source="di4c",
            entrypoint="sdtt/src/sdtt/main.py",
            max_steps=sampling_step,
            global_batch_size=2,
            learning_rate=3e-5,
            checkpoint_every=10_000,
            default_per_device_batch_size=1,
            algorithm="di4c",
            teacher_family=teacher_family,
            teacher_adapter=adapter,
            distillation_loss="dimensional_correlations",
            sampling_step=sampling_step,
            evidence="paper_appendix_d",
        )
    if model == "di4c":
        raise RecipeError("Di4C requires --model duo_di4c or mdlm_di4c")
    raise RecipeError(f"unknown training recipe {model!r}")


def validate_effective_batch(
    global_batch_size: int,
    *,
    devices: int,
    nodes: int,
    per_device_batch_size: int,
) -> int:
    """Return exact accumulation or reject a reduced/non-divisible paper batch."""

    values = (global_batch_size, devices, nodes, per_device_batch_size)
    if any(value <= 0 for value in values):
        raise RecipeError("batch size, devices, nodes, and per-device batch must be positive")
    denominator = devices * nodes * per_device_batch_size
    if global_batch_size % denominator:
        raise RecipeError(
            f"global batch {global_batch_size} is not divisible by "
            f"{devices} devices * {nodes} nodes * {per_device_batch_size} per device"
        )
    accumulation = global_batch_size // denominator
    if devices * nodes * per_device_batch_size * accumulation != global_batch_size:
        raise RecipeError("effective global batch differs from the locked recipe")
    return accumulation


def _tokenizer_snapshot(root: Path, dataset: str) -> Path:
    name, revision = TOKENIZER_REVISIONS[dataset]
    hub_name = "models--" + name.replace("/", "--")
    return (root / "data" / "raw" / "huggingface" / "hub" / hub_name / "snapshots" / revision).absolute()


def _data_cache(root: Path, source: str, dataset: str) -> Path:
    if source == "rdlm":
        return (root / "data" / "raw" / "huggingface" / "datasets").absolute()
    return (root / "data" / "training_cache" / source / dataset).absolute()


def _duo_integral_cache(root: Path, dataset: str) -> Path:
    filename = "bert-base-uncased.pkl" if dataset == "lm1b" else "gpt2.pkl"
    return (root / "upstreams" / "duo" / "integral" / filename).absolute()


def _float_text(value: float) -> str:
    return format(value, ".12g")


def _common_lightning_overrides(
    recipe: TrainingRecipe,
    *,
    root: Path,
    output: Path,
    data_cache: Path,
    tokenizer: Path,
    devices: int,
    nodes: int,
    per_device_batch_size: int,
    accumulation: int,
    seed: int,
    resume: bool,
) -> list[str]:
    if recipe.dataset == "lm1b":
        data_group = "lm1b" if recipe.source == "mdlm" else "lm1b-wrap"
    else:
        data_group = "openwebtext-split"
    arguments = [
        "mode=train",
        f"data={data_group}",
        f"data.cache_dir={data_cache}",
        f"data.tokenizer_name_or_path={tokenizer}",
        "model=small",
        f"model.length={recipe.sequence_length}",
        f"loader.global_batch_size={recipe.global_batch_size}",
        f"loader.eval_global_batch_size={recipe.global_batch_size}",
        f"loader.batch_size={per_device_batch_size}",
        f"loader.eval_batch_size={per_device_batch_size}",
        f"trainer.devices={devices}",
        f"trainer.num_nodes={nodes}",
        f"trainer.accumulate_grad_batches={accumulation}",
        f"trainer.max_steps={recipe.max_steps}",
        f"trainer.precision={recipe.precision}",
        f"optim.lr={_float_text(recipe.learning_rate)}",
        "lr_scheduler=constant_warmup",
        f"lr_scheduler.num_warmup_steps={recipe.warmup_steps}",
        (
            "callbacks.checkpoint_every_n_steps.every_n_train_steps="
            f"{recipe.checkpoint_every}"
        ),
        f"checkpointing.save_dir={output}",
        f"checkpointing.resume_from_ckpt={str(resume).lower()}",
        f"checkpointing.resume_ckpt_path={output / 'checkpoints' / 'last.ckpt'}",
        f"hydra.run.dir={output}",
        "hydra.job.chdir=true",
        "wandb=null",
        f"seed={seed}",
    ]
    if recipe.source == "mdlm" and recipe.dataset == "lm1b":
        arguments.extend(
            [
                "data.wrap=true",
                "+data.insert_train_eos=true",
                "+data.insert_valid_eos=true",
            ]
        )
    return arguments


def _standard_command(
    recipe: TrainingRecipe,
    *,
    python: Path,
    entrypoint: Path,
    root: Path,
    output: Path,
    data_cache: Path,
    tokenizer: Path,
    teacher: Path | None,
    devices: int,
    nodes: int,
    per_device_batch_size: int,
    accumulation: int,
    seed: int,
    resume: bool,
) -> list[str]:
    command = [str(python), "-u", str(entrypoint)]
    command.extend(
        _common_lightning_overrides(
            recipe,
            root=root,
            output=output,
            data_cache=data_cache,
            tokenizer=tokenizer,
            devices=devices,
            nodes=nodes,
            per_device_batch_size=per_device_batch_size,
            accumulation=accumulation,
            seed=seed,
            resume=resume,
        )
    )
    if recipe.model == "flm":
        command.extend(["algo=flm", "algo.double_temb=false"])
    elif recipe.model == "fmlm":
        if teacher is None:
            raise RecipeError("FMLM requires an FLM teacher checkpoint")
        command.extend(
            [
                "algo=fmlm",
                "algo.double_temb=true",
                "algo.distillation_method=PSD",
                "algo.use_mse_loss_psd=false",
                "algo.learnable_loss_weighting=false",
                "algo.initialize_student_from_teacher=true",
                f"algo.teacher_path={teacher}",
            ]
        )
    elif recipe.model == "duo":
        command.extend(
            [
                "algo=duo",
                "algo.curriculum.mode=simple",
                "algo.curriculum.gumbel_tau_log10_start=-3.0",
                "algo.curriculum.gumbel_tau_log10_end=-3.0",
                "algo.curriculum.gamma_min=-3.5",
                "algo.curriculum.gamma_max=-1.75",
                "algo.curriculum.start=0",
                "algo.curriculum.end=500000",
            ]
        )
    elif recipe.model == "duo_dcd":
        if teacher is None:
            raise RecipeError("Duo+DCD requires a uniform Duo teacher checkpoint")
        command.extend(
            [
                "algo=distillation",
                f"training.finetune_path={teacher}",
                "training.ema=0.999",
                "algo.T=512",
                f"algo.integral_cache_path={_duo_integral_cache(root, recipe.dataset)}",
                "+algo.curriculum.mode=simple",
                "+algo.curriculum.gumbel_tau_log10_start=-1",
                "+algo.curriculum.gumbel_tau_log10_end=-1",
                "+algo.curriculum.start=-1",
                "+algo.curriculum.end=-1",
                "+algo.curriculum.gamma_min=-4",
                "+algo.curriculum.gamma_max=-1",
                f"+algo.curriculum.integral_cache_path={_duo_integral_cache(root, recipe.dataset)}",
                f"algo.update_teacher_every={recipe.steps_per_round}",
                "algo.teacher_ema=false",
                "algo.linear_growth_dt=false",
            ]
        )
    elif recipe.model == "mdlm":
        command.extend(["parameterization=subs", "diffusion=absorbing_state"])
    elif recipe.model == "candi":
        command.extend(
            [
                f"scratch_dir={root / 'work' / 'candi'}",
                "algo=candi",
                "algo.sampler=cached",
                "algo.mixed_coeff=0.5",
                "algo.step_size=1.0",
                "algo.temp=1.0",
            ]
        )
    else:
        raise RecipeError(f"no standard command renderer for {recipe.model}")
    return command


def _rdlm_command(
    recipe: TrainingRecipe,
    *,
    python: Path,
    entrypoint: Path,
    output: Path,
    data_cache: Path,
    devices: int,
    accumulation: int,
    seed: int,
) -> list[str]:
    return [
        str(python),
        "-u",
        str(entrypoint),
        "run_mode=train",
        "exp=lm1b",
        "optim=adam",
        f"ngpus={devices}",
        f"training.batch_size={recipe.global_batch_size}",
        f"training.accum={accumulation}",
        f"training.n_iters={recipe.max_steps}",
        f"training.snapshot_freq={recipe.checkpoint_every}",
        "training.snapshot_freq_for_preemption=5000",
        "training.snapshot_sampling=false",
        f"data.cache_dir={data_cache}",
        f"optim.lr={_float_text(recipe.learning_rate)}",
        f"optim.warmup={recipe.warmup_steps}",
        f"hydra.run.dir={output}",
        "use_wandb=false",
        f"seed={seed}",
    ]


def _sdtt_command(
    recipe: TrainingRecipe,
    *,
    python: Path,
    entrypoint: Path,
    output: Path,
    data_cache: Path,
    tokenizer: Path,
    adapted_teacher: Path,
    devices: int,
    nodes: int,
    per_device_batch_size: int,
    accumulation: int,
    seed: int,
    resume: bool,
) -> list[str]:
    train_name = "lm1b" if recipe.dataset == "lm1b" else "openwebtext-train"
    valid_name = "lm1b" if recipe.dataset == "lm1b" else "openwebtext-valid"
    command = [
        str(python),
        "-u",
        str(entrypoint),
        "mode=train",
        f"data.train={train_name}",
        f"data.valid={valid_name}",
        f"tokenizer.name={tokenizer}",
        f"data_preprocess.data_cache={data_cache}",
        f"data_preprocess.seq_len={recipe.sequence_length}",
        "data_preprocess.group_text=true",
        "data_preprocess.remove_text=true",
        "data_preprocess.add_bos=false",
        "data_preprocess.add_eos=true",
        "data_preprocess.legacy_start_end_bos=true",
        "model=dit-orig-small",
        f"model.length={recipe.sequence_length}",
        "parameterization=multi-round-sdtt",
        f"parameterization.checkpoint_path={adapted_teacher}",
        "parameterization.start_from_hf=false",
        f"parameterization.grow_dt_every={recipe.steps_per_round or 10000}",
        "parameterization.orig_num_sampling_steps=1024",
        f"loader.global_batch_size={recipe.global_batch_size}",
        f"loader.eval_global_batch_size={recipe.global_batch_size}",
        f"loader.batch_size={per_device_batch_size}",
        f"loader.eval_batch_size={per_device_batch_size}",
        f"trainer.devices={devices}",
        f"trainer.num_nodes={nodes}",
        f"trainer.accumulate_grad_batches={accumulation}",
        f"trainer.max_steps={recipe.max_steps}",
        "trainer.precision=bf16-mixed",
        f"optim.lr={_float_text(recipe.learning_rate)}",
        f"lr_scheduler.num_warmup_steps={recipe.warmup_steps}",
        (
            "callbacks.checkpoint_every_n_steps.every_n_train_steps="
            f"{recipe.checkpoint_every}"
        ),
        f"checkpointing.save_dir={output}",
        f"checkpointing.resume_from_ckpt={str(resume).lower()}",
        f"checkpointing.resume_ckpt_path={output / 'checkpoints' / 'last.ckpt'}",
        f"hydra.run.dir={output}",
        "hydra.job.chdir=true",
        "wandb=null",
        f"+seed={seed}",
    ]
    if recipe.model.endswith("_di4c"):
        command.extend(
            [
                "is_di4c=true",
                "is_teacher_di4c=false",
                "T=1024",
                "latent_bsize=16",
                "round=7",
            ]
        )
    return command


def build_launch(
    recipe: TrainingRecipe,
    *,
    root: Path,
    source: Path,
    output: Path,
    teacher: Path | None,
    devices: int,
    nodes: int,
    per_device_batch_size: int,
    seed: int,
    resume: bool,
) -> LaunchSpec:
    """Render one real upstream command without touching data, checkpoints, or GPUs."""

    root = root.absolute()
    source = source.absolute()
    output = output.absolute()
    teacher = teacher.absolute() if teacher is not None else None
    accumulation = validate_effective_batch(
        recipe.global_batch_size,
        devices=devices,
        nodes=nodes,
        per_device_batch_size=per_device_batch_size,
    )
    python = Path(sys.executable).absolute()
    entrypoint = (source / recipe.entrypoint).absolute()
    data_cache = _data_cache(root, recipe.source, recipe.dataset)
    tokenizer = _tokenizer_snapshot(root, recipe.dataset)
    adapted_teacher = (
        output / "inputs" / "teacher_adapted.ckpt"
        if recipe.teacher_adapter is not None
        else None
    )
    if recipe.teacher_family is not None and teacher is None:
        raise RecipeError(f"{recipe.model} requires --teacher or --teacher-checkpoint")
    if recipe.source == "rdlm":
        command = _rdlm_command(
            recipe,
            python=python,
            entrypoint=entrypoint,
            output=output,
            data_cache=data_cache,
            devices=devices,
            accumulation=accumulation,
            seed=seed,
        )
    elif recipe.source in {"sdtt", "di4c"}:
        if adapted_teacher is None:
            raise RecipeError("SDTT/Di4C recipes require a teacher adapter")
        command = _sdtt_command(
            recipe,
            python=python,
            entrypoint=entrypoint,
            output=output,
            data_cache=data_cache,
            tokenizer=tokenizer,
            adapted_teacher=adapted_teacher,
            devices=devices,
            nodes=nodes,
            per_device_batch_size=per_device_batch_size,
            accumulation=accumulation,
            seed=seed,
            resume=resume,
        )
    else:
        command = _standard_command(
            recipe,
            python=python,
            entrypoint=entrypoint,
            root=root,
            output=output,
            data_cache=data_cache,
            tokenizer=tokenizer,
            teacher=teacher,
            devices=devices,
            nodes=nodes,
            per_device_batch_size=per_device_batch_size,
            accumulation=accumulation,
            seed=seed,
            resume=resume,
        )
    return LaunchSpec(
        recipe=recipe,
        command=tuple(command),
        cwd=source,
        entrypoint=entrypoint,
        data_path=data_cache,
        output=output,
        teacher=teacher,
        adapted_teacher=adapted_teacher,
        devices=devices,
        nodes=nodes,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation=accumulation,
        effective_global_batch_size=(
            devices * nodes * per_device_batch_size * accumulation
        ),
        seed=seed,
        resume=resume,
    )


def _shape(value: object, name: str) -> tuple[int, ...]:
    raw = getattr(value, "shape", None)
    if raw is not None:
        try:
            return tuple(int(item) for item in raw)
        except (TypeError, ValueError) as error:
            raise RecipeError(f"{name} has an invalid shape") from error
    if isinstance(value, list):
        if not value:
            return (0,)
        children = [_shape(item, name) for item in value]
        if any(child != children[0] for child in children):
            raise RecipeError(f"{name} is not rectangular")
        return (len(value), *children[0])
    return ()


def _clone(value: object) -> object:
    clone = getattr(value, "detach", None)
    if callable(clone):
        return clone().clone()
    return deepcopy(value)


def _zero_row_like(value: object, row: int) -> object:
    if isinstance(value, list):
        selected = value[row]
        if isinstance(selected, list):
            return [0 for _ in selected]
        return 0
    selected = value[row]
    zeros_like = getattr(selected, "new_zeros", None)
    if callable(zeros_like):
        return zeros_like(selected.shape)
    raise RecipeError("tensor object does not provide new_zeros")


def _append_or_move_row(
    value: object,
    *,
    source_row: int | None,
    target_size: int,
    zero_source: bool,
    name: str,
) -> object:
    shape = _shape(value, name)
    if not shape or shape[0] <= 0:
        raise RecipeError(f"{name} must have a non-empty vocabulary dimension")
    source_size = shape[0]
    if target_size not in {source_size, source_size + 1}:
        raise RecipeError(
            f"{name} vocabulary {source_size} cannot map to target {target_size}"
        )
    if source_row is not None and not 0 <= source_row < source_size:
        raise RecipeError(f"{name} source mask row is out of range")
    if isinstance(value, list):
        result = deepcopy(value)
        selected = (
            deepcopy(value[source_row])
            if source_row is not None
            else _zero_row_like(value, 0)
        )
        if target_size == source_size + 1:
            result.append(selected)
        elif source_row is not None and source_row != target_size - 1:
            result[target_size - 1] = selected
        if zero_source and source_row is not None and source_row != target_size - 1:
            result[source_row] = _zero_row_like(value, source_row)
        return result

    result = value.new_zeros((target_size, *shape[1:]))
    result[:source_size].copy_(value)
    if source_row is not None:
        result[target_size - 1].copy_(value[source_row])
        if zero_source and source_row != target_size - 1:
            result[source_row].zero_()
    return result


def _validate_state(
    state: Mapping[str, object], expected_keys: Collection[str] | None
) -> tuple[int, int]:
    if expected_keys is not None:
        expected = set(expected_keys)
        observed = set(state)
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected - OPTIONAL_BACKBONE_STATE_KEYS)
        if missing or unexpected:
            raise RecipeError(
                f"teacher state keys differ: missing={missing}, unexpected={unexpected}"
            )
    missing_tensors = sorted(set(BACKBONE_TENSOR_KEYS) - set(state))
    if missing_tensors:
        raise RecipeError(f"teacher state keys are missing {missing_tensors}")
    embedding_shape = _shape(state[BACKBONE_TENSOR_KEYS[0]], BACKBONE_TENSOR_KEYS[0])
    weight_shape = _shape(state[BACKBONE_TENSOR_KEYS[1]], BACKBONE_TENSOR_KEYS[1])
    bias_shape = _shape(state[BACKBONE_TENSOR_KEYS[2]], BACKBONE_TENSOR_KEYS[2])
    if len(embedding_shape) != 2 or len(weight_shape) != 2 or len(bias_shape) != 1:
        raise RecipeError("teacher embedding/head ranks must be 2, 2, and 1")
    vocab, hidden = embedding_shape
    if weight_shape[0] != vocab or bias_shape[0] != vocab:
        raise RecipeError("teacher embedding/head vocabulary dimensions differ")
    if weight_shape[1] != hidden:
        raise RecipeError("teacher embedding/head hidden dimensions differ")
    if vocab <= 0 or hidden <= 0:
        raise RecipeError("teacher vocabulary and hidden dimensions must be positive")
    return vocab, hidden


def uniform_to_absorbing(
    state: Mapping[str, object],
    *,
    target_vocab_size: int | None = None,
    expected_keys: Collection[str] | None = None,
) -> dict[str, object]:
    """Append an all-zero absorbing state to a uniform-Duo backbone."""

    vocab, _ = _validate_state(state, expected_keys)
    target_size = target_vocab_size or vocab + 1
    if target_size not in {vocab, vocab + 1}:
        raise RecipeError(
            f"uniform teacher vocabulary {vocab} cannot map to target {target_size}"
        )
    result = {key: _clone(value) for key, value in state.items()}
    if target_size == vocab:
        for key in BACKBONE_TENSOR_KEYS:
            result[key] = _zero_vocab_row(result[key], target_size - 1)
        _validate_state(result, set(state))
        return result
    for key in BACKBONE_TENSOR_KEYS:
        result[key] = _append_or_move_row(
            state[key],
            source_row=None,
            target_size=target_size,
            zero_source=False,
            name=key,
        )
    _validate_state(result, set(state))
    return result


def _zero_vocab_row(value: object, row: int) -> object:
    if isinstance(value, list):
        result = deepcopy(value)
        result[row] = _zero_row_like(value, row)
        return result
    result = value.clone()
    result[row].zero_()
    return result


def masked_to_absorbing(
    state: Mapping[str, object],
    *,
    source_mask_index: int,
    target_vocab_size: int,
    expected_keys: Collection[str] | None = None,
) -> dict[str, object]:
    """Move an existing masked-model state to the final absorbing-mask row."""

    vocab, _ = _validate_state(state, expected_keys)
    if target_vocab_size not in {vocab, vocab + 1}:
        raise RecipeError(
            f"masked teacher vocabulary {vocab} cannot map to {target_vocab_size}"
        )
    result = {key: _clone(value) for key, value in state.items()}
    if source_mask_index == target_vocab_size - 1 and target_vocab_size == vocab:
        return result
    for key in BACKBONE_TENSOR_KEYS:
        result[key] = _append_or_move_row(
            state[key],
            source_row=source_mask_index,
            target_size=target_vocab_size,
            zero_source=True,
            name=key,
        )
    _validate_state(result, set(state))
    return result


def expected_backbone_keys(n_blocks: int = 12) -> set[str]:
    if n_blocks <= 0:
        raise RecipeError("backbone must contain at least one transformer block")
    keys = {
        "vocab_embed.embedding",
        "sigma_map.mlp.0.weight",
        "sigma_map.mlp.0.bias",
        "sigma_map.mlp.2.weight",
        "sigma_map.mlp.2.bias",
        "output_layer.norm_final.weight",
        "output_layer.linear.weight",
        "output_layer.linear.bias",
        "output_layer.adaLN_modulation.weight",
        "output_layer.adaLN_modulation.bias",
    }
    block_suffixes = {
        "norm1.weight",
        "attn_qkv.weight",
        "attn_out.weight",
        "norm2.weight",
        "mlp.0.weight",
        "mlp.0.bias",
        "mlp.2.weight",
        "mlp.2.bias",
        "adaLN_modulation.weight",
        "adaLN_modulation.bias",
    }
    for index in range(n_blocks):
        keys.update(f"blocks.{index}.{suffix}" for suffix in block_suffixes)
    return keys


def _normalize_backbone_state(state: Mapping[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        while key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        if key.startswith("backbone."):
            key = key[len("backbone.") :]
        if key in normalized:
            raise RecipeError(f"teacher key normalization collision at {key!r}")
        normalized[key] = value
    return normalized


def _callable_sha256(function: object) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def adapt_teacher_checkpoint(
    launch: LaunchSpec,
    *,
    source_mask_index: int | None = None,
) -> dict[str, object]:
    """Adapt and atomically publish one server-side teacher checkpoint."""

    if launch.teacher is None or launch.adapted_teacher is None:
        raise RecipeError("teacher adaptation requires source and destination paths")
    _require_regular_file(launch.teacher, "teacher checkpoint")
    try:
        import torch
    except ImportError as error:
        raise RecipeError(
            "PyTorch is required only for real server teacher adaptation"
        ) from error
    try:
        payload = torch.load(
            str(launch.teacher), map_location="cpu", weights_only=False
        )
    except Exception as error:
        raise RecipeError(f"could not load teacher checkpoint: {error}") from error
    if not isinstance(payload, MutableMapping):
        raise RecipeError("teacher checkpoint must contain a mapping")
    raw_state = payload.get("state_dict", payload)
    if not isinstance(raw_state, Mapping):
        raise RecipeError("teacher checkpoint state_dict must be a mapping")
    state = _normalize_backbone_state(raw_state)
    expected = expected_backbone_keys()
    _validate_state(state, expected)
    base_vocab = BASE_VOCAB_SIZES[launch.recipe.dataset]
    observed_vocab, hidden = _validate_state(state, expected)
    if launch.recipe.teacher_adapter == "uniform_to_absorbing":
        target_vocab = base_vocab + 1
        if observed_vocab not in {base_vocab, target_vocab}:
            raise RecipeError(
                "uniform teacher vocabulary is "
                f"{observed_vocab}, expected {base_vocab} or {target_vocab}"
            )
        adapted = uniform_to_absorbing(
            state,
            target_vocab_size=target_vocab,
            expected_keys=expected,
        )
        transform = uniform_to_absorbing
        mask_index = target_vocab - 1
    elif launch.recipe.teacher_adapter == "masked_to_absorbing":
        target_vocab = base_vocab + 1
        if source_mask_index is None:
            source_mask_index = (
                target_vocab - 1 if observed_vocab == target_vocab else 103
            )
        adapted = masked_to_absorbing(
            state,
            source_mask_index=source_mask_index,
            target_vocab_size=target_vocab,
            expected_keys=expected,
        )
        transform = masked_to_absorbing
        mask_index = target_vocab - 1
    else:
        raise RecipeError(f"unknown teacher adapter {launch.recipe.teacher_adapter!r}")
    adapted_vocab, adapted_hidden = _validate_state(adapted, expected)
    if adapted_vocab != base_vocab + 1 or adapted_hidden != hidden:
        raise RecipeError("adapted teacher shape does not match absorbing backbone")

    destination = launch.adapted_teacher
    ensure_safe_directory(destination.parent)
    _require_regular_or_missing(destination, "adapted teacher checkpoint")
    adapted_payload = {"state_dict": adapted}
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    _require_regular_or_missing(temporary, "temporary adapted checkpoint")
    try:
        torch.save(adapted_payload, temporary)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    record = {
        "schema_version": 1,
        "source_path": str(launch.teacher),
        "source_sha256": sha256_file(launch.teacher),
        "output_path": str(destination),
        "output_sha256": sha256_file(destination),
        "teacher_family": launch.recipe.teacher_family,
        "transformation": launch.recipe.teacher_adapter,
        "transformation_sha256": _callable_sha256(transform),
        "source_mask_index": source_mask_index,
        "target_mask_index": mask_index,
        "source_vocab_size": observed_vocab,
        "target_vocab_size": adapted_vocab,
        "hidden_size": hidden,
        "state_keys_sha256": hashlib.sha256(
            "\n".join(sorted(adapted)).encode("utf-8")
        ).hexdigest(),
    }
    atomic_json_write(destination.with_suffix(".provenance.json"), record)
    return record


def _path_components(path: Path) -> list[Path]:
    components: list[Path] = []
    current = path
    while True:
        components.append(current)
        if current.parent == current:
            return list(reversed(components))
        current = current.parent


def _reject_symlink_components(path: Path, label: str) -> None:
    for component in _path_components(path):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise RecipeError(f"{label} contains a symlink component: {component}")


def _project_path(root: Path, value: Path, label: str) -> Path:
    candidate = value if value.is_absolute() else root / value
    candidate = candidate.absolute()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise RecipeError(f"{label} must remain inside project root") from error
    _reject_symlink_components(candidate, label)
    return candidate


def _require_regular_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise RecipeError(f"{label} is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RecipeError(f"{label} is not a safe regular file: {path}")


def _require_regular_or_missing(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RecipeError(f"{label} is not a safe regular file: {path}")


def _verify_source(launch: LaunchSpec) -> None:
    _require_regular_file(launch.entrypoint, "upstream entrypoint")
    completed = subprocess.run(
        ["git", "-C", str(launch.cwd), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode or completed.stdout.strip() != launch.recipe.source_commit:
        raise RecipeError(
            f"{launch.recipe.source} source is not pinned commit "
            f"{launch.recipe.source_commit}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(launch.cwd), "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=False,
    )
    dirty_lines = _source_dirty_lines(dirty.stdout)
    if dirty.returncode or dirty_lines:
        raise RecipeError(f"{launch.recipe.source} source checkout is dirty")


def _source_dirty_lines(status_output: str) -> list[str]:
    """Return source status lines excluding Python bytecode cache byproducts."""

    return [
        line
        for line in status_output.splitlines()
        if line.strip() and not _is_untracked_python_bytecode_cache(line)
    ]


def _is_untracked_python_bytecode_cache(status_line: str) -> bool:
    if not status_line.startswith("?? "):
        return False
    path = status_line[3:].strip()
    parts = PurePosixPath(path).parts
    return "__pycache__" in parts or path.endswith((".pyc", ".pyo"))


def _processed_dataset(root: Path, dataset: str) -> tuple[Path, Path]:
    name = "lm1b-bert-128" if dataset == "lm1b" else "owt-gpt2-1024"
    processed = root / "data" / "processed" / name
    manifest = root / "data" / "manifests" / f"{dataset}.json"
    if processed.is_symlink() or not processed.is_dir():
        raise RecipeError(f"processed dataset is missing: {processed}")
    _require_regular_file(manifest, "processed dataset manifest")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if value.get("dataset") != dataset:
        raise RecipeError("processed dataset manifest identifies another dataset")
    if int(value.get("sequence_length", -1)) != SEQUENCE_LENGTHS[dataset]:
        raise RecipeError("processed dataset sequence length differs from recipe")
    return processed, manifest


def _link_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise RecipeError(f"training cache path is unsafe: {destination}")
        source_files = {
            path.relative_to(source): path
            for path in source.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        destination_files = {
            path.relative_to(destination): path
            for path in destination.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        if set(source_files) != set(destination_files):
            raise RecipeError(f"training cache inventory differs: {destination}")
        for relative, source_file in source_files.items():
            destination_file = destination_files[relative]
            source_stat = source_file.stat()
            destination_stat = destination_file.stat()
            if source_stat.st_size != destination_stat.st_size:
                raise RecipeError(f"training cache file size differs: {destination_file}")
            if source_stat.st_dev != destination_stat.st_dev or source_stat.st_ino != destination_stat.st_ino:
                if sha256_file(source_file) != sha256_file(destination_file):
                    raise RecipeError(
                        f"training cache file digest differs: {destination_file}"
                    )
        return
    ensure_safe_directory(destination)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_symlink():
            raise RecipeError(f"processed dataset contains a symlink: {path}")
        if path.is_dir():
            ensure_safe_directory(target)
        elif path.is_file():
            ensure_safe_directory(target.parent)
            try:
                os.link(path, target)
            except OSError:
                shutil.copyfile(path, target)
        else:
            raise RecipeError(f"processed dataset contains a special file: {path}")


def _remove_safe_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        raise RecipeError(f"training cache path is unsafe: {path}")
    for child in path.rglob("*"):
        if child.is_symlink():
            raise RecipeError(f"training cache contains a symlink: {child}")
    shutil.rmtree(path)


def _dataset_columns(path: Path) -> set[str]:
    from datasets import load_from_disk

    dataset = load_from_disk(str(path))
    return set(dataset.column_names)


def _materialize_text_training_split(
    source: Path,
    destination: Path,
    *,
    sequence_length: int,
    cache_is_current: bool,
) -> None:
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise RecipeError(f"training cache path is unsafe: {destination}")
        if cache_is_current and {"input_ids", "attention_mask"} <= _dataset_columns(
            destination
        ):
            return
        _remove_safe_tree(destination)

    from datasets import Sequence as DatasetSequence
    from datasets import Value, load_from_disk

    dataset = load_from_disk(str(source))
    if "attention_mask" not in dataset.column_names:
        features = deepcopy(dataset.features)
        features["attention_mask"] = DatasetSequence(
            Value("uint8"), length=sequence_length
        )

        def add_attention_mask(batch):
            return {
                "attention_mask": [
                    [1] * len(input_ids) for input_ids in batch["input_ids"]
                ]
            }

        dataset = dataset.map(
            add_attention_mask,
            batched=True,
            features=features,
            load_from_cache_file=False,
            desc="Adding all-one attention masks for upstream training",
        )

    staged = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    _remove_safe_tree(staged)
    ensure_safe_directory(destination.parent)
    try:
        dataset.save_to_disk(str(staged))
        if destination.exists():
            _remove_safe_tree(destination)
        os.replace(staged, destination)
    finally:
        _remove_safe_tree(staged)


def _prepare_training_cache(launch: LaunchSpec, root: Path) -> dict[str, object]:
    processed, manifest = _processed_dataset(root, launch.recipe.dataset)
    processed_manifest_sha = sha256_file(manifest)
    if launch.recipe.source == "rdlm":
        if launch.data_path.is_symlink() or not launch.data_path.is_dir():
            raise RecipeError(
                "RDLM requires the pinned raw Hugging Face dataset cache; "
                "run scripts/fetch_data.py first"
            )
        return {
            "mode": "upstream_raw_retokenization",
            "processed_preflight": str(processed),
            "processed_manifest_sha256": processed_manifest_sha,
            "cache": str(launch.data_path),
        }

    ensure_safe_directory(launch.data_path)
    cache_record = launch.data_path / "dlb_training_cache.json"
    cache_is_current = False
    if cache_record.is_file() and not cache_record.is_symlink():
        try:
            current = json.loads(cache_record.read_text(encoding="utf-8"))
            cache_is_current = (
                current.get("processed_manifest_sha256") == processed_manifest_sha
                and current.get("format") == "hf_dataset_with_attention_mask_v1"
            )
        except (OSError, json.JSONDecodeError):
            cache_is_current = False
    if launch.recipe.dataset == "lm1b":
        aliases = {
            "lm1b_train_bs128_wrapped.dat": processed / "train",
            "lm1b_test_bs128_wrapped.dat": processed / "validation",
        }
    else:
        aliases = {
            "openwebtext-train_train_bs1024_wrapped.dat": processed / "train",
            "openwebtext-valid_validation_bs1024_wrapped.dat": processed
            / "validation",
        }
    for name, source in aliases.items():
        if not source.is_dir():
            raise RecipeError(f"processed split is missing: {source}")
        _materialize_text_training_split(
            source,
            launch.data_path / name,
            sequence_length=launch.recipe.sequence_length,
            cache_is_current=cache_is_current,
        )
    record = {
        "mode": "materialized_bridge",
        "format": "hf_dataset_with_attention_mask_v1",
        "processed_path": str(processed),
        "processed_manifest_sha256": processed_manifest_sha,
        "cache": str(launch.data_path),
        "aliases": {name: str(path) for name, path in aliases.items()},
    }
    atomic_json_write(launch.data_path / "dlb_training_cache.json", record)
    return record


def _launch_identity(
    launch: LaunchSpec,
    data_record: Mapping[str, object],
    *,
    composed_config_sha256: str,
) -> dict[str, object]:
    teacher_sha = (
        sha256_file(launch.teacher)
        if launch.teacher is not None and launch.teacher.is_file()
        else None
    )
    recipe_value = asdict(launch.recipe)
    value = {
        "schema_version": 1,
        "recipe": recipe_value,
        "command": list(launch.command),
        "cwd": str(launch.cwd),
        "source_commit": launch.recipe.source_commit,
        "data": dict(data_record),
        "teacher_sha256": teacher_sha,
        "teacher_transformation_sha256": (
            _callable_sha256(
                uniform_to_absorbing
                if launch.recipe.teacher_adapter == "uniform_to_absorbing"
                else masked_to_absorbing
            )
            if launch.recipe.teacher_adapter is not None
            else None
        ),
        "composed_config_sha256": composed_config_sha256,
        "seed": launch.seed,
        "devices": launch.devices,
        "nodes": launch.nodes,
        "per_device_batch_size": launch.per_device_batch_size,
        "gradient_accumulation": launch.gradient_accumulation,
        "effective_global_batch_size": launch.effective_global_batch_size,
    }
    value["identity_sha256"] = hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    return value


def _completed_is_reusable(output: Path, identity: Mapping[str, object]) -> bool:
    marker = output / "completed.json"
    if marker.is_symlink() or not marker.is_file():
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if value.get("identity_sha256") != identity.get("identity_sha256"):
        raise RecipeError("completed output belongs to a different recipe identity")
    checkpoint = output / str(value.get("checkpoint", ""))
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise RecipeError("completed output checkpoint is missing or unsafe")
    if sha256_file(checkpoint) != value.get("checkpoint_sha256"):
        raise RecipeError("completed output checkpoint digest differs")
    config = output / str(value.get("config", ""))
    if config.is_symlink() or not config.is_file():
        raise RecipeError("completed output config is missing or unsafe")
    if sha256_file(config) != value.get("config_sha256"):
        raise RecipeError("completed output config digest differs")
    if value.get("config_sha256") != identity.get("composed_config_sha256"):
        raise RecipeError("completed output config differs from current composition")
    return True


def _compose_config(launch: LaunchSpec, environment: Mapping[str, str]) -> str:
    compose = list(launch.command[:3]) + ["--cfg", "job", "--resolve"] + list(
        launch.command[3:]
    )
    completed = subprocess.run(
        compose,
        cwd=launch.cwd,
        env=dict(environment),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RecipeError(
            "Hydra config composition failed before launch: "
            + completed.stderr[-4000:]
        )
    if not completed.stdout.strip():
        raise RecipeError("Hydra produced an empty composed config")
    return completed.stdout


def _write_text_atomic(path: Path, payload: str) -> None:
    _require_regular_or_missing(path, "text output")
    ensure_safe_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    _require_regular_or_missing(temporary, "temporary text output")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        _require_regular_or_missing(path, "text output")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _open_safe_append(path: Path):
    ensure_safe_directory(path.parent)
    _require_regular_or_missing(path, "log output")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    return os.fdopen(descriptor, "ab", buffering=0)


def _copy_atomic(source: Path, destination: Path) -> None:
    _require_regular_file(source, "produced checkpoint")
    _require_regular_or_missing(destination, "canonical checkpoint")
    ensure_safe_directory(destination.parent)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    _require_regular_or_missing(temporary, "temporary canonical checkpoint")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _expected_checkpoint(launch: LaunchSpec) -> tuple[Path, str]:
    recipe = launch.recipe
    if recipe.source == "rdlm":
        index = recipe.max_steps // recipe.checkpoint_every
        return launch.output / "checkpoints" / f"checkpoint_{index}.pth", (
            f"checkpoints/checkpoint_{index}.pth"
        )
    if recipe.model in {"mdlm_sdtt", "duo_di4c", "mdlm_di4c"}:
        if recipe.sampling_step is None:
            raise RecipeError("distilled recipe has no selected sampling step")
        relative = f"student_checkpoints/{recipe.sampling_step}.ckpt"
        candidates = (
            launch.output / relative,
            launch.output / "checkpoints" / f"0-{recipe.sampling_step}.ckpt",
            launch.output / "checkpoints" / "last.ckpt",
        )
        for candidate in candidates:
            if candidate.is_file() and not candidate.is_symlink():
                return candidate, relative
        return candidates[0], relative
    return launch.output / "checkpoints" / "last.ckpt", "model.ckpt"


def execute_launch(launch: LaunchSpec, *, root: Path) -> str:
    """Run one prepared recipe on a server, returning ``completed`` or ``skipped``."""

    _verify_source(launch)
    data_record = _prepare_training_cache(launch, root)
    if launch.teacher is not None:
        _require_regular_file(launch.teacher, "teacher checkpoint")
    ensure_safe_directory(launch.output)
    environment = dict(os.environ)
    environment.update(
        {
            "DLB_ROOT": str(root),
            "HF_HOME": str(root / "data" / "raw" / "huggingface"),
            "HF_DATASETS_CACHE": str(
                root / "data" / "raw" / "huggingface" / "datasets"
            ),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    upstream_python_path = str(launch.entrypoint.parents[1])
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        upstream_python_path
        if not existing_python_path
        else upstream_python_path + os.pathsep + existing_python_path
    )
    composed = _compose_config(launch, environment)
    composed_sha256 = hashlib.sha256(composed.encode("utf-8")).hexdigest()
    identity = _launch_identity(
        launch, data_record, composed_config_sha256=composed_sha256
    )
    if _completed_is_reusable(launch.output, identity):
        return "skipped"
    complete_marker = launch.output / "completed.json"
    if complete_marker.exists():
        raise RecipeError("refusing to overwrite a non-reusable completed run")

    adaptation = None
    if launch.recipe.teacher_adapter is not None:
        adaptation = adapt_teacher_checkpoint(launch)
    atomic_json_write(launch.output / "recipe.json", asdict(launch.recipe))
    atomic_json_write(launch.output / "launch_argv.json", list(launch.command))
    atomic_json_write(
        launch.output / "provenance.json",
        {
            **identity,
            "teacher_adaptation": adaptation,
            "prepared_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    )
    config_path = launch.output / "config.yaml"
    _write_text_atomic(config_path, composed)
    with _open_safe_append(launch.output / "stdout.log") as stdout_file, _open_safe_append(
        launch.output / "stderr.log"
    ) as stderr_file:
        completed = subprocess.run(
            launch.command,
            cwd=launch.cwd,
            env=environment,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )
    if completed.returncode:
        if completed.returncode < 0:
            return_code = 128 + abs(completed.returncode)
        else:
            return_code = completed.returncode
        raise RecipeProcessError(
            f"upstream training exited with status {return_code}", return_code
        )
    produced, relative = _expected_checkpoint(launch)
    _require_regular_file(produced, "selected training checkpoint")
    canonical = launch.output / relative
    if produced != canonical:
        _copy_atomic(produced, canonical)
    marker = {
        "schema_version": 1,
        "identity_sha256": identity["identity_sha256"],
        "checkpoint": relative,
        "checkpoint_sha256": sha256_file(canonical),
        "config": "config.yaml",
        "config_sha256": sha256_file(config_path),
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    atomic_json_write(complete_marker, marker)
    return "completed"


def _dry_record(launch: LaunchSpec) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dry_run": True,
        "model": launch.recipe.model,
        "dataset": launch.recipe.dataset,
        "source": str(launch.cwd),
        "source_commit": launch.recipe.source_commit,
        "entrypoint": str(launch.entrypoint),
        "data": str(launch.data_path),
        "teacher": str(launch.teacher) if launch.teacher is not None else None,
        "adapted_teacher": (
            str(launch.adapted_teacher)
            if launch.adapted_teacher is not None
            else None
        ),
        "output": str(launch.output),
        "command": list(launch.command),
        "effective_global_batch_size": launch.effective_global_batch_size,
        "gradient_accumulation": launch.gradient_accumulation,
        "recipe": asdict(launch.recipe),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        required=True,
        choices=("flm", "duo", "mdlm", "candi", "rdlm", "fmlm", "duo_dcd", "mdlm_sdtt", "di4c"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", choices=("lm1b", "owt"), required=True)
    parser.add_argument("--model", choices=("duo_di4c", "mdlm_di4c"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--teacher", "--teacher-checkpoint", dest="teacher", type=Path)
    parser.add_argument(
        "--teacher-family", choices=("continuous_flm", "uniform_duo", "masked_mdlm")
    )
    parser.add_argument("--devices", type=int)
    parser.add_argument("--nodes", type=int, default=int(os.getenv("DLB_TRAIN_NODES", "1")))
    parser.add_argument("--per-device-batch-size", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--steps-per-round", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upstream-override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _assert_locked(name: str, supplied: object, expected: object) -> None:
    if supplied is not None and supplied != expected:
        raise RecipeError(
            f"{name}={supplied!r} differs from locked recipe value {expected!r}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.root.absolute()
    _reject_symlink_components(root, "project root")
    model = arguments.model if arguments.recipe == "di4c" else arguments.recipe
    if model is None:
        raise RecipeError("--model is required by the Di4C wrapper")
    if arguments.recipe != "di4c" and arguments.model is not None:
        raise RecipeError("--model is valid only for the Di4C wrapper")
    recipe = load_recipe(model, arguments.dataset)
    _assert_locked("global batch size", arguments.global_batch_size, recipe.global_batch_size)
    _assert_locked("learning rate", arguments.learning_rate, recipe.learning_rate)
    _assert_locked("rounds", arguments.rounds, recipe.rounds)
    _assert_locked("steps per round", arguments.steps_per_round, recipe.steps_per_round)
    _assert_locked("teacher family", arguments.teacher_family, recipe.teacher_family)
    allowed_overrides = {
        "is_di4c=true",
    }
    unknown = sorted(set(arguments.upstream_override) - allowed_overrides)
    if unknown:
        raise RecipeError(f"unsupported or unsafe upstream overrides: {unknown}")
    source_value = arguments.source or Path(recipe.source_path)
    output_value = arguments.output or Path(
        "checkpoints", "self_trained", recipe.dataset, recipe.model
    )
    source = _project_path(root, source_value, "source")
    output = _project_path(root, output_value, "output")
    teacher = (
        _project_path(root, arguments.teacher, "teacher")
        if arguments.teacher is not None
        else None
    )
    per_device = (
        arguments.per_device_batch_size or recipe.default_per_device_batch_size
    )
    devices = arguments.devices
    if devices is None:
        default_devices = "2" if recipe.model.endswith("_di4c") else "8"
        devices = int(os.getenv("DLB_TRAIN_DEVICES", default_devices))
    launch = build_launch(
        recipe,
        root=root,
        source=source,
        output=output,
        teacher=teacher,
        devices=devices,
        nodes=arguments.nodes,
        per_device_batch_size=per_device,
        seed=arguments.seed,
        resume=arguments.resume,
    )
    if arguments.dry_run:
        print(json.dumps(_dry_record(launch), sort_keys=True))
        return 0
    status = execute_launch(launch, root=root)
    print(json.dumps({"status": status, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RecipeProcessError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(error.exit_status)
    except RecipeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
