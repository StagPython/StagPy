PY=python3

VENV_DIR=stagpy_git
STAGPY=$(VENV_DIR)/bin/stagpy
VPY=$(VENV_DIR)/bin/python
VPIP=$(VPY) -m pip

BRANCH=$(shell git rev-parse --abbrev-ref HEAD)
VERSION=$(shell git describe --exact-match HEAD 2>/dev/null)

.PHONY: all clean again release
.PHONY: info infoenv infozsh infobash
.PHONY: notebook-kernel

all: $(STAGPY) info

$(STAGPY): setup.cfg pyproject.toml
	$(PY) -m venv $(VENV_DIR)
	$(VPIP) install -U pip
	$(VPIP) install -e .
	@$(STAGPY) version

notebook-kernel: $(STAGPY)
	$(VPIP) install -U ipykernel
	$(VPY) -m ipykernel install --user --name=stagpy-git

info: infozsh infobash infoenv

infoenv:
	@echo
	@echo 'Run'
	@echo '  source $(VENV_DIR)/bin/activate'
	@echo 'to use the development version of StagPy'

infozsh:
	@echo
	@echo 'Add'
	@echo ' source ~/.config/stagpy/zsh/_stagpy.sh'
	@echo 'to your ~/.zshrc to enjoy command line completion with zsh!'

infobash:
	@echo
	@echo 'Add'
	@echo ' source ~/.config/stagpy/bash/stagpy.sh'
	@echo 'to your ~/.bashrc to enjoy command line completion with bash!'

clean:
	-rm -rf $(VENV_DIR)
	-rm -rf stagpy.egg-info

again: clean all

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
