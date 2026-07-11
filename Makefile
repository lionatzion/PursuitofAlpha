PYTHON ?= python
SCAFFOLD_ROOT ?= /tmp/pursuitofalpha-scaffold
MPLCONFIGDIR ?= /tmp/pursuitofalpha-matplotlib
XDG_CACHE_HOME ?= /tmp/pursuitofalpha-cache

export MPLCONFIGDIR
export XDG_CACHE_HOME

.PHONY: backtest entrypoints full scaffold scaffold-contract test train verify

train:
	$(PYTHON) -m pipelines.train_pipeline

backtest:
	$(PYTHON) -m pipelines.backtest_pipeline

full:
	$(PYTHON) -m pipelines.full_pipeline

scaffold:
	$(PYTHON) project_scaffold.py --root "$(SCAFFOLD_ROOT)"

entrypoints:
	$(PYTHON) -m pipelines.train_pipeline --help >/dev/null
	$(PYTHON) -m pipelines.backtest_pipeline --help >/dev/null
	$(PYTHON) -m pipelines.full_pipeline --help >/dev/null
	$(PYTHON) project_scaffold.py --help >/dev/null

scaffold-contract:
	$(PYTHON) -m pytest tests/test_project_scaffold.py -q

test:
	$(PYTHON) -m pytest

verify: entrypoints scaffold-contract test
