.PHONY: install train overfit test lint tensorboard clean

install:
	python -m pip install -e ".[dev]"

train:
	python -m llm_from_scratch_lab.train_toy --config configs/toy_mlp.json

overfit:
	python -m llm_from_scratch_lab.train_toy --config configs/toy_mlp.json --overfit-one-batch

test:
	python -m pytest

lint:
	python -m ruff check .

tensorboard:
	tensorboard --logdir artifacts/runs

clean:
	python -c "from pathlib import Path; import shutil; shutil.rmtree(Path('artifacts'), ignore_errors=True)"
