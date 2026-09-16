PY ?= python
CFG ?= configs/default.yaml

install:      ; $(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e . --no-deps
download:     ; $(PY) -m sshocks.cli data --download --config $(CFG)
data:         ; $(PY) -m sshocks.cli data --config $(CFG)
forecast:     ; $(PY) -m sshocks.cli forecast --config $(CFG)
changepoint:  ; $(PY) -m sshocks.cli changepoint --config $(CFG)
figures:      ; $(PY) -m sshocks.cli figures --config $(CFG)
all:          ; $(PY) -m sshocks.cli all --config $(CFG)
test:         ; $(PY) -m pytest -q tests
.PHONY: install download data forecast changepoint figures all test
