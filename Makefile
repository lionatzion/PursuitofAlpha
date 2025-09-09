.PHONY: setup train backtest full

setup:
	python scaffold.py

train:
	python pipelines/train_pipeline.py

backtest:
	python pipelines/backtest_pipeline.py

full:
	python pipelines/full_pipeline.py
