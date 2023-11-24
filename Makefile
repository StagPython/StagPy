PY=python3

BRANCH=$(shell git rev-parse --abbrev-ref HEAD)
VERSION=$(shell git describe --exact-match HEAD 2>/dev/null)

.PHONY: all clean again release
.PHONY: info infoenv infozsh infobash
.PHONY: notebook-kernel

all:
	@echo 'Run `make release` to release a new version'

release:
ifneq ($(BRANCH),master)
	@echo 'Please run git checkout master'
	@echo 'Then rerun make release'
else
ifeq ($(VERSION),)
	@echo -n 'Please tag HEAD with the desired version number'
	@echo ' (last version: $(shell git describe HEAD --abbrev=0))'
	@echo 'git tag -a vX.Y.Z'
	@echo 'Then rerun make release'
else
	@echo 'Release $(VERSION)'
	git push --follow-tags
	$(PY) -m pip install -U --user pip build setuptools twine wheel
	$(PY) -m build
	$(PY) -m twine upload dist/*
	-rm -rf dist/
endif
endif
