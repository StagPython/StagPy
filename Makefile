LINK_DIR=~/bin
LINK_NAME=stagpy-git
LINK=$(LINK_DIR)/$(LINK_NAME)

PY=python3

BLD_DIR=bld
VENV_DIR=$(BLD_DIR)/venv
STAGPY=$(VENV_DIR)/bin/stagpy
VPY=$(VENV_DIR)/bin/python
VPIP=$(VPY) -m pip

.PHONY: all install config clean uninstall autocomplete
.PHONY: info infopath infozsh infobash

OBJS=setup.py stagpy/*.py

all: install

$(BLD_DIR):
	@mkdir -p $@

install: $(LINK) config infopath autocomplete
	@echo
	@echo 'Installation completed!'

autocomplete: infozsh infobash

config: $(STAGPY)
	$(STAGPY) config --update

$(LINK): $(STAGPY)
	@mkdir -p $(LINK_DIR)
	ln -sf $(PWD)/$(STAGPY) $(LINK)

$(STAGPY): $(VENV_DIR) $(OBJS)
	$(VPIP) install -e .

$(VENV_DIR):
	$(PY) -m venv --system-site-packages $@
	$(VPIP) install -U pip

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
	-rm -rf $(BLD_DIR)
	-rm -rf stagpy.egg-info

uninstall:
	-rm -rf ~/.config/stagpy/
	-rm -f $(LINK)

again: clean all
