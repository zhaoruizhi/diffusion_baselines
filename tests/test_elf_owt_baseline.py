"""Regression tests for all-completion scoring and exact C64 prompt identity."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_elf_owt_baseline import c64_records
from evaluation.conditional_perplexity import compute_conditional_gen_ppl


class Tokenizer:
    def decode(self, ids, **kwargs):
        return ' '.join(map(str, ids))

    def __call__(self, texts, **kwargs):
        rows = [[int(x) for x in text.split()][:kwargs['max_length']] for text in texts]
        width = max(map(len, rows))
        return {'input_ids': [r + [0] * (width - len(r)) for r in rows],
                'attention_mask': [[1] * len(r) + [0] * (width - len(r)) for r in rows]}


class Scorer:
    def eval(self):
        return self

    def score_prompt_excluded_batch(self, ids, attention_mask, target_mask):
        nll, count = 0., 0
        for row, mask in zip(ids, target_mask):
            for token, target in zip(row[1:], mask[1:]):
                if target:
                    if token not in (2, 3):
                        raise AssertionError('Prompt or post-64 suffix was scored')
                    nll += math.log(2 if token == 2 else 8)
                    count += 1
        return nll, count


class C64Alignment(unittest.TestCase):
    def test_all_2048_completions_and_only_first_64_response_tokens(self):
        tok = Tokenizer()
        prompts = [{'prompt_id': i, 'prefix_token_ids': [9] * 64,
                    'reference_token_ids': [4] * 64} for i in range(1024)]
        schedule = [(i, 0) for i in range(1024)] + [(i, c) for c in range(1, 5) for i in range(256)]
        rows = [{'prompt_id': i, 'completion_id': c, 'input': tok.decode([9] * 64),
                 'reference': tok.decode([4] * 64)} for i, c in schedule]
        tokens = [[2 if c == 0 else 3] * 64 + [99] * 896 for i, c in schedule]
        records = c64_records(rows, prompts, tokens, tok)
        result = compute_conditional_gen_ppl(records, Scorer(), tok, tok, batch_size=32)
        self.assertEqual(result.sample_count, 2048)
        self.assertEqual(result.valid_token_count, 2048 * 64)
        self.assertAlmostEqual(result.perplexity, 4.0)

    def test_short_response_and_mismatched_prompt_are_not_silently_repaired(self):
        tok = Tokenizer()
        prompt = {'prompt_id': 0, 'prefix_token_ids': [9] * 64, 'reference_token_ids': [4] * 64}
        row = {'prompt_id': 0, 'completion_id': 0, 'input': tok.decode([9] * 64),
               'reference': tok.decode([4] * 64)}
        with self.assertRaisesRegex(ValueError, 'Short response'):
            c64_records([row], [prompt], [[2] * 63], tok)
        with self.assertRaisesRegex(ValueError, 'input differs'):
            c64_records([{**row, 'input': 'wrong'}], [prompt], [[2] * 64], tok)


if __name__ == '__main__':
    unittest.main()
