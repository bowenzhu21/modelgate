.PHONY: install test demo benchmark clean
PYTHON ?= python3
install:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install -e .
test:
	.venv/bin/python -m unittest discover -s tests -v
demo:
	.venv/bin/modelgate demo --out demo-output
benchmark:
	.venv/bin/modelgate benchmark --out demo-output/benchmark.json
clean:
	rm -rf runs build dist
