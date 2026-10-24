"""CPU regression checks for the chapter 2 teaching scripts, using synthetic text."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import tiktoken
import torch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "ch02_text_data"


@pytest.fixture(scope="module")
def gpt2():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def example_workspace(tmp_path, monkeypatch):
    # An explicit synthetic fixture, not the downloadable book corpus.
    text = (
        '"It\'s the last he painted, you know."Mrs. Gisburn said with pardonable pride. '
        "Hello, do you like tea? In the sunlit terraces of the palace. "
    ) * 4
    data_dir = tmp_path / "data" / "ch02"
    data_dir.mkdir(parents=True)
    (data_dir / "the-verdict.txt").write_text(text, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(EXAMPLES))
    return text


def test_simple_tokenizers_round_trip_and_unknown_tokens():
    examples = runpy.run_path(str(EXAMPLES / "simple_tokenizer.py"))
    vocab = {token: index for index, token in enumerate(
        ["Hello", ",", "world", "!", ":", ";", "<|unk|>", "<|endoftext|>"]
    )}
    v1 = examples["SimpleTokenizerV1"](vocab)
    v2 = examples["SimpleTokenizerV2"](vocab)

    for tokenizer in (v1, v2):
        assert tokenizer.encode("Hello, world!") == [0, 1, 2, 3]
        assert tokenizer.decode([0, 1, 2, 3]) == "Hello, world!"
        assert tokenizer.encode("") == []
        assert tokenizer.decode([]) == ""
    with pytest.raises(KeyError, match="unseen"):
        v1.encode("unseen")
    assert v2.encode("Hello: unseen; <|endoftext|>") == [0, 4, 6, 5, 7]
    assert v2.decode([0, 4, 6, 5, 7]) == "Hello: <|unk|>; <|endoftext|>"


def test_bpe_example_uses_real_gpt2_encoding(gpt2):
    assert gpt2.encode("Hello world!") == [15496, 995, 0]
    text = "你好，世界！ Akwirw ier 🙂"
    assert gpt2.decode(gpt2.encode(text)) == text

    example = runpy.run_path(str(EXAMPLES / "bpe_tokenizer.py"))
    assert example["decoded_text"] == example["text"]
    assert 50256 in example["integers"]
    assert gpt2.decode(example["exercise_ids"]) == "Akwirw ier"


@pytest.mark.parametrize("stride", [1, 4, 5])
def test_sliding_windows_have_exact_next_token_targets(gpt2, stride):
    dataset_class = runpy.run_path(str(EXAMPLES / "sliding_window.py"))["GPTDatasetV1"]
    text = "The quick brown fox jumps over the lazy dog."
    tokens = gpt2.encode(text)
    context = 4
    dataset = dataset_class(text, gpt2, max_length=context, stride=stride)
    assert len(dataset) == (len(tokens) - context - 1) // stride + 1

    for index, (inputs, targets) in enumerate(dataset):
        start = index * stride
        assert inputs.dtype == targets.dtype == torch.long
        assert inputs.device.type == targets.device.type == "cpu"
        assert inputs.tolist() == tokens[start:start + context]
        assert targets.tolist() == tokens[start + 1:start + context + 1]
        assert torch.equal(inputs[1:], targets[:-1])

    # A full input also needs a following token; incomplete targets are excluded.
    assert len(dataset_class(text, gpt2, max_length=len(tokens), stride=1)) == 0
    single = dataset_class(text, gpt2, max_length=len(tokens) - 1, stride=1)
    assert len(single) == 1
    assert single[0][1][-1].item() == tokens[-1]
    assert len(dataset_class("", gpt2, max_length=context, stride=stride)) == 0


def test_dataloader_batches_drop_last_and_seeded_shuffle(gpt2):
    create_loader = runpy.run_path(str(EXAMPLES / "sliding_window.py"))[
        "create_dataloader_v1"
    ]
    text = "The quick brown fox jumps over the lazy dog."
    sample_count = len(gpt2.encode(text)) - 4
    batch_size = sample_count - 1
    loader = create_loader(text, batch_size=batch_size, max_length=4, stride=1,
                           shuffle=False, drop_last=False)
    batches = list(loader)
    assert [tuple(inputs.shape) for inputs, _ in batches] == [(batch_size, 4), (1, 4)]
    assert all(torch.equal(inputs[:, 1:], targets[:, :-1]) for inputs, targets in batches)
    dropped = list(create_loader(text, batch_size=batch_size, max_length=4, stride=1,
                                 shuffle=False, drop_last=True))
    assert len(dropped) == 1
    assert torch.equal(dropped[0][0], batches[0][0])
    assert torch.equal(dropped[0][1], batches[0][1])

    def shuffled():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            return list(create_loader(text, batch_size=2, max_length=4, stride=1,
                                      shuffle=True, drop_last=False))

    first, second = shuffled(), shuffled()
    assert len(first) == len(second)
    for (x1, y1), (x2, y2) in zip(first, second, strict=True):
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)
    assert sorted(torch.cat([x for x, _ in first]).tolist()) == sorted(
        torch.cat([x for x, _ in batches]).tolist()
    )


def test_positional_embedding_script_shapes_values_and_repeatability(example_workspace, gpt2):
    def run_seeded():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            return runpy.run_path(str(EXAMPLES / "positional_embedding.py"))

    first = run_seeded()
    tokens = gpt2.encode(example_workspace)
    assert first["inputs"].tolist() == [tokens[start:start + 4] for start in range(0, 32, 4)]
    assert first["targets"].tolist() == [tokens[start + 1:start + 5]
                                        for start in range(0, 32, 4)]
    assert first["token_embeddings"].shape == (8, 4, 256)
    assert first["position_ids"].tolist() == [0, 1, 2, 3]
    assert first["pos_embeddings"].shape == (4, 256)
    actual = first["input_embeddings"]
    assert actual.shape == (8, 4, 256)
    assert actual.device.type == "cpu"
    token_weights = first["token_embedding_layer"].weight
    position_weights = first["pos_embedding_layer"].weight
    for batch in range(8):
        for position in range(4):
            expected = token_weights[first["inputs"][batch, position]] + position_weights[position]
            assert torch.equal(actual[batch, position], expected)
    second = run_seeded()
    assert torch.equal(actual, second["input_embeddings"])


def test_token_embedding_script_is_an_exact_weight_lookup():
    with torch.random.fork_rng(devices=[]):
        example = runpy.run_path(str(EXAMPLES / "tokenembedding.py"))
    layer = example["embedding_layer"]
    inputs = example["input_ids"]
    assert layer(inputs).shape == (4, 3)
    assert torch.equal(layer(inputs), layer.weight[inputs])


def test_tokenize_verdict_script_runs_with_synthetic_corpus(example_workspace):
    example = runpy.run_path(str(EXAMPLES / "tokenize_verdict.py"))
    assert example["raw_text"] == example_workspace
    assert example["int_to_str"][example["token_id"]] == "the"
    tokenizer = example["tokenizer"]
    assert tokenizer.encode(tokenizer.decode(example["ids"])) == example["ids"]
    tokenizer_v2 = example["tokenizer_v2"]
    encoded = tokenizer_v2.encode(example["text"])
    assert tokenizer_v2.str_to_int["<|endoftext|>"] in encoded
    assert tokenizer_v2.decode(encoded) == example["text"]
