LINK_DIR=~/bin
LINK_NAME=stagpy-git
LINK=$(LINK_DIR)/$(LINK_NAME)

PY=python3

BLD_DIR=bld
VENV_DIR=$(BLD_DIR)/venv
STAGPY=$(VENV_DIR)/bin/stagpy
VPY=$(VENV_DIR)/bin/python
VPIP=$(VENV_DIR)/bin/pip

COMP=$(PWD)/$(VENV_DIR)/bin/register-python-argcomplete

.PHONY: all install config clean uninstall autocomplete
.PHONY: info infopath infozsh infobash

CONF_FILE=~/.config/stagpy/config
OBJS=setup.py stagpy/*.py
COMP_ZSH=$(BLD_DIR)/comp.zsh
COMP_SH=$(BLD_DIR)/comp.sh

all: install

$(BLD_DIR):
	@mkdir -p $@

install: $(LINK) config infopath autocomplete
	@echo
	@echo 'Installation completed!'

autocomplete: $(COMP_ZSH) $(COMP_SH) infozsh infobash

$(COMP_ZSH): $(LINK)
	@echo 'autoload bashcompinit' > $@
	@echo 'bashcompinit' >> $@
	@echo 'eval "$$($(COMP) $(LINK_NAME))"' >> $@

$(COMP_SH): $(LINK)
	@echo 'eval "$$($(COMP) $(LINK_NAME))"' > $@

config: $(STAGPY) $(CONF_FILE)
	@$(STAGPY) config --update
	@echo 'Config file updated!'

$(CONF_FILE):
	@$(STAGPY) config --create
	@echo 'Config file created!'

$(LINK): $(STAGPY)
	@mkdir -p $(LINK_DIR)
	ln -sf $(PWD)/$(STAGPY) $(LINK)

$(STAGPY): $(VENV_DIR) $(OBJS)
	$(VPY) -E setup.py install
	@echo 'Removing useless build files'
	@-rm -rf build dist stagpy.egg-info

$(VENV_DIR): $(BLD_DIR)/get-pip.py requirements.txt
	$(PY) -m venv --system-site-packages --without-pip $@
	$(VPY) -E $< -I
	$(VPIP) install -I argcomplete
	$(VPIP) install -r requirements.txt

$(BLD_DIR)/get-pip.py: $(BLD_DIR)
	@echo 'Dowloading get-pip.py...'
	@$(PY) downloadgetpip.py $@
	@echo 'Done'

info: infopath infozsh infobash

infopath:
	@echo
	@echo 'Add $(LINK_DIR) to your path to be able to call StagPy from anywhere'

infozsh:
	@echo
	@echo 'Add'
	@echo ' source $(PWD)/$(COMP_ZSH)'
	@echo 'to your ~/.zshrc to enjoy command line completion with zsh!'

infobash:
	@echo
	@echo 'Add'
	@echo ' source $(PWD)/$(COMP_SH)'
	@echo 'to your ~/.bashrc to enjoy command line completion with bash!'

clean: uninstall
	@echo 'Removing build files'
	@-rm -rf $(BLD_DIR)

uninstall:
	@echo 'Removing config files...'
	@-rm -rf ~/.config/stagpy/
	@echo 'Removing link...'
	@-rm -f $(LINK)
	@echo 'Done.'

again: clean all
