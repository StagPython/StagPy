LINK_DIR=~/bin
LINK_NAME=stagpy
LINK=$(LINK_DIR)/$(LINK_NAME)

VENV=stagpyvenv
STAGPY=$(VENV)/bin/stagpy

CPLT=$(PWD)/$(VENV)/bin/register-python-argcomplete
.PHONY: all install config clean uninstall autocomplete
.PHONY: info infopath infozsh infobash

OBJS=setup.py stagpy/*.py

all: install config infopath autocomplete

config: $(STAGPY)
	@$(STAGPY) config --create
	@echo 'Config file created!'

install: $(LINK)
	@echo
	@echo 'Installation completed!'

autocomplete: .comp.zsh .comp.sh infozsh infobash

.comp.zsh:
	@echo 'autoload bashcompinit' > $@
	@echo 'bashcompinit' >> $@
	@echo 'eval "$$($(CPLT) $(LINK_NAME))"' >> $@

.comp.sh:
	@echo 'eval "$$($(CPLT) $(LINK_NAME))"' > $@

$(LINK): $(STAGPY)
	@mkdir -p $(LINK_DIR)
	ln -sf $(PWD)/$(STAGPY) $(LINK)

$(STAGPY): $(VENV) $(OBJS)
	$(VENV)/bin/python setup.py install

$(VENV):
	python3 -m virtualenv --system-site-packages $@
	$@/bin/pip install -I argcomplete

info: infopath infozsh infobash

infopath:
	@echo
	@echo 'Add $(LINK_DIR) to your path to be able to call StagPy from anywhere'

infozsh:
	@echo
	@echo 'Add'
	@echo ' source $(PWD)/.comp.zsh'
	@echo 'to your ~/.zshrc to enjoy command line completion with zsh!'

infobash:
	@echo
	@echo 'Add'
	@echo ' source $(PWD)/.comp.sh'
	@echo 'to your ~/.bashrc to enjoy command line completion with bash!'

clean: uninstall
	@echo 'Removing build and virtualenv files'
	rm -rf build/ dist/ StagPy.egg-info/ $(VENV)
	rm -rf stagpy/__pycache__ .comp.zsh .comp.sh

uninstall:
	@echo 'Removing config file...'
	@rm -rf ~/.config/stagpy/
	@echo 'Removing link...'
	@rm -f $(LINK)
	@echo 'Done.'
