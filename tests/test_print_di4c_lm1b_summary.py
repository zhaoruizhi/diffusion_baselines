from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_print_di4c_lm1b_summary_filters_and_formats_rows(tmp_path: Path) -> None:
    summary = tmp_path / "results" / "summary"
    summary.mkdir(parents=True)
    (summary / "results_wide.csv").write_text(
        "\n".join(
            [
                "task_id,model,dataset,category,steps,sample_count,seed,"
                "generative_perplexity,unigram_entropy,self_bleu,"
                "generation_seconds_per_sample,provenance",
                "duo_di4c-lm1b-steps-1,duo_di4c,lm1b,few,1,1024,42,"
                "101.234,4.1,0.05,0.01234567,reference_reproduction",
                "mdlm_di4c-lm1b-steps-32,mdlm_di4c,lm1b,few,32,1024,42,"
                "996.576,4.384469,0.032817,0.25,reference_reproduction",
                "duo_di4c-owt-steps-1,duo_di4c,owt,few,1,1024,42,"
                "14.5,1.0,0.4,0.01,reference_reproduction",
                "flm-lm1b-steps-1,flm,lm1b,many,1,1024,42,"
                "119.34,4.16,0.1,0.02,reference_reproduction",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (summary / "failures.csv").write_text(
        "\n".join(
            [
                "status,task_id,model,dataset,category,steps,reason",
                "failed,mdlm_di4c-lm1b-steps-16,mdlm_di4c,lm1b,few,16,samples missing",
                "failed,flm-lm1b-steps-2,flm,lm1b,many,2,ignored",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "print_di4c_lm1b_summary.py"),
            "--root",
            str(tmp_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert completed.stdout == (
        "model,dataset,steps,ppl,entropy,self_bleu,seconds_per_sample\n"
        "duo_di4c,lm1b,1,101.23,4.1000,0.0500,0.012346\n"
        "mdlm_di4c,lm1b,32,996.58,4.3845,0.0328,0.250000\n"
        "\n"
        "# failures\n"
        "task_id,model,dataset,steps,reason\n"
        "mdlm_di4c-lm1b-steps-16,mdlm_di4c,lm1b,16,samples missing\n"
    )
