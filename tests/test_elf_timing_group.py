"""Exercise the timing shell entry with fake GPU/runner tools, never CUDA."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_elf_timing_group.sh"


class TimingGroupTests(unittest.TestCase):
    def test_plan_matches_quality_matrix_without_inference(self):
        for group, count in (("owt", 14), ("seq2seq", 18)):
            result = subprocess.check_output(["bash", str(SCRIPT), "0", group, "--plan"], text=True)
            cells = [line.split() for line in result.splitlines()[2:]]
            self.assertEqual(len(cells), count)
            self.assertEqual(len({tuple(c) for c in cells}), count)
            if group == "owt":
                self.assertEqual([int(c[2]) for c in cells[:7]], [1, 2, 4, 8, 16, 32, 1024])
            else:
                self.assertEqual(sum(c[1] == "test" for c in cells), 4)

    def fake_run(self, fail=False, busy=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "scripts").mkdir()
        shutil.copy(ROOT / "scripts/run_elf_suite.sh", root / "scripts")
        binary = root / "bin"
        binary.mkdir()
        scripts = {
            "uname": "#!/bin/sh\necho Linux\n",
            "git": "#!/bin/sh\necho 0123456789012345678901234567890123456789\n",
            "nvidia-smi": '#!/bin/sh\ncase "$*" in\n*--query-gpu=uuid*) echo GPU-test-0;;\n*--query-compute-apps=pid*) if [ "$ELF_TEST_BUSY" = 1 ]; then echo 12345; fi;;\n*) echo GPU-snapshot;;\nesac\n',
            "fake-python": '''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['ELF_TEST_LOG'], 'a') as f:
    f.write(json.dumps({'args': sys.argv[1:], 'gpu': os.environ['CUDA_VISIBLE_DEVICES']}) + '\\n')
if os.environ['ELF_TEST_FAIL'] == '1' and '--steps' in sys.argv and sys.argv[sys.argv.index('--steps') + 1] == '2':
    sys.exit(7)
''',
        }
        for name, content in scripts.items():
            path = binary / name
            path.write_text(content)
            path.chmod(0o755)
        log = root / "calls.jsonl"
        env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"], DLB_ROOT=str(root),
                   ELF_PYTHON=str(binary / "fake-python"), ELF_TEST_LOG=str(log),
                   ELF_TEST_BUSY=str(int(busy)), ELF_TEST_FAIL=str(int(fail)),
                   ELF_STEPS="999", ELF_COMPILE="1", ELF_SEED="99", ELF_TIMING_PROMPT="128")
        result = subprocess.run(["bash", str(SCRIPT), "0", "owt"], env=env, text=True, capture_output=True)
        return result, [json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []

    def test_full_group_binds_gpu_and_overrides_stale_quality_environment(self):
        result, calls = self.fake_run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 15)  # 14 cells, then summary.
        for call in calls[:14]:
            args = call['args']
            self.assertEqual(call['gpu'], 'GPU-test-0')
            self.assertEqual(args[1], 'timing')
            self.assertEqual(args[args.index('--batch-size') + 1], '1')
            self.assertEqual(args[args.index('--seed') + 1], '42')
            self.assertEqual(args[args.index('--timing-prompt-index') + 1], '0')
            self.assertNotIn('--compile', args)
        self.assertIn('FINISHED', result.stdout)

    def test_runner_failure_is_not_hidden_by_tee(self):
        result, calls = self.fake_run(fail=True)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(calls), 2)
        self.assertNotIn('FINISHED', result.stdout)

    def test_busy_gpu_stops_before_inference(self):
        result, calls = self.fake_run(busy=True)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(calls)
        self.assertIn('compute processes', result.stderr)


if __name__ == '__main__':
    unittest.main()
