import pytest

from datasets import Dataset

from dlb.data import build_owt_split, pack_tokens, preprocess_split


class TinyBatchTokenizer:
    def __call__(self, texts, **kwargs):
        assert kwargs == {
            "add_special_tokens": False,
            "return_attention_mask": False,
            "return_token_type_ids": False,
        }
        table = {"first": [10, 11], "second": [12, 13, 14]}
        return {"input_ids": [table[text] for text in texts]}


def test_owt_split_is_last_100k_documents():
    """Catch shuffling or selecting any validation slice except the final 100k."""

    split = build_owt_split(total_documents=8_013_769)

    assert split.train == "train[:-100000]"
    assert split.validation == "train[-100000:]"


def test_owt_split_rejects_a_source_too_small_for_both_splits():
    """Catch silently producing an empty training split from a truncated snapshot."""

    with pytest.raises(ValueError, match="more than 100,000"):
        build_owt_split(total_documents=100_000)


def test_pack_tokens_reserves_boundaries():
    """Catch a packed block ending in document content rather than EOS."""

    blocks = list(
        pack_tokens([[10, 11], [12, 13, 14]], length=6, bos_id=101, eos_id=102)
    )

    assert blocks[0] == [101, 10, 11, 102, 12, 102]
    assert all(len(block) == 6 for block in blocks)


def test_pack_tokens_carries_overflow_and_drops_only_the_incomplete_tail():
    """Catch token loss when one document crosses a fixed-length boundary."""

    blocks = list(
        pack_tokens([[10, 11, 12, 13, 14, 15]], length=6, bos_id=101, eos_id=102)
    )

    assert blocks == [[101, 10, 11, 12, 13, 102]]


def test_preprocess_split_packs_real_dataset_rows_in_source_order(tmp_path):
    """Catch preprocessing that reorders documents or packs each batch separately."""

    source = Dataset.from_dict({"text": ["first", "second"]})

    processed = preprocess_split(
        source,
        tokenizer=TinyBatchTokenizer(),
        length=6,
        bos_id=101,
        eos_id=102,
        cache_dir=tmp_path,
        batch_documents=1,
    )

    assert processed.to_list() == [
        {"input_ids": [101, 10, 11, 102, 12, 102]},
    ]
