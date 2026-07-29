from pathlib import Path

import pytest
import yaml


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


@pytest.fixture
def source_manifest():
    with Path("artifacts/sources.yaml").open(encoding="utf-8") as manifest_file:
        return yaml.safe_load(manifest_file)


def test_source_manifest_has_exact_repositories(source_manifest):
    assert {name: source["url"] for name, source in source_manifest.items()} == EXPECTED
    assert all(len(source["commit"]) == 40 for source in source_manifest.values())
