LINK_DIR=~/bin
LINK_NAME=stagpy-git
LINK=$(LINK_DIR)/$(LINK_NAME)

PY=python3

VENV_DIR=.venv_dev
STAGPY=$(VENV_DIR)/bin/stagpy
VPY=$(VENV_DIR)/bin/python
VPIP=$(VPY) -m pip

BRANCH=$(shell git rev-parse --abbrev-ref HEAD)
VERSION=$(shell git describe --exact-match HEAD 2>/dev/null)

.PHONY: all install clean uninstall again release
.PHONY: info infopath infozsh infobash

all: install

install: $(LINK) info
	@echo
	@echo 'Installation completed!'

$(LINK): $(STAGPY)
	@mkdir -p $(LINK_DIR)
	ln -sf $(PWD)/$(STAGPY) $(LINK)

$(STAGPY): setup.py
	$(PY) -m venv $(VENV_DIR)
	$(VPIP) install -U pip
	$(VPIP) install --no-use-pep517 -e .
	@$(STAGPY) version

info: infopath infozsh infobash

infopath:
	@echo
	@echo 'Add $(LINK_DIR) to your path to be able to call StagPy from anywhere'

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

clean: uninstall
	-rm -rf $(VENV_DIR)
	-rm -rf stagpy.egg-info

uninstall:
	-rm -rf ~/.config/stagpy/
	-rm -f $(LINK)

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
	$(PY) -m pip install -U --user pip setuptools twine wheel
	$(PY) setup.py sdist bdist_wheel
	$(PY) -m twine upload dist/*
	-rm -rf build/ dist/
endif
endif
