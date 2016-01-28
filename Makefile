LINK_DIR=~/bin
LINK_NAME=stagpy
LINK=$(LINK_DIR)/$(LINK_NAME)

# set venv to virtualenv with Python3.2
VENV_MOD=venv

VENV_DIR=stagpyvenv
STAGPY=$(VENV_DIR)/bin/stagpy

CPLT=$(PWD)/$(VENV_DIR)/bin/register-python-argcomplete

.PHONY: all install config clean uninstall autocomplete
.PHONY: info infopath infozsh infobash
.PHONY: novirtualenv

CONF_FILE=~/.config/stagpy/config
OBJS=setup.py stagpy/*.py

all: install

install: $(LINK) config infopath autocomplete
	@echo
	@echo 'Installation completed!'

autocomplete: .comp.zsh .comp.sh infozsh infobash

.comp.zsh:
	@echo 'autoload bashcompinit' > $@
	@echo 'bashcompinit' >> $@
	@echo 'eval "$$($(CPLT) $(LINK_NAME))"' >> $@

.comp.sh:
	@echo 'eval "$$($(CPLT) $(LINK_NAME))"' > $@

config: $(STAGPY) $(CONF_FILE) stagpy/config.py
	@$(STAGPY) config --update
	@echo 'Config file updated!'

$(CONF_FILE):
	@$(STAGPY) config --create
	@echo 'Config file created!'

$(LINK): $(STAGPY)
	@mkdir -p $(LINK_DIR)
	ln -sf $(PWD)/$(STAGPY) $(LINK)

$(STAGPY): $(VENV_DIR) $(OBJS)
	$</bin/python -E setup.py install

$(VENV_DIR): .get-pip.py requirements.txt
	python3 -m $(VENV_MOD) --system-site-packages --without-pip $@
	$@/bin/python -E $< -I
	$@/bin/pip install -I argcomplete
	$@/bin/pip install -r requirements.txt

.get-pip.py:
	@echo 'Dowloading get-pip.py...'
	@python3 download-get-pip.py
	@echo 'Done'

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
	rm -rf build/ dist/ StagPy.egg-info/ $(VENV_DIR)
	rm -rf stagpy/__pycache__
	rm -f .get-pip.py .comp.zsh .comp.sh .install.txt

uninstall:
	@echo 'Removing config file...'
	@rm -rf ~/.config/stagpy/
	@echo 'Removing link...'
	@rm -f $(LINK)
	@echo 'Done.'

novirtualenv: .get-pip.py requirements.txt $(OBJS)
	@echo 'Installing without virtual environment'
	python3 $< --user
	python3 -m pip install --user -r requirements.txt
	python3 setup.py install --user --record .install.txt
	@echo 'Installation finished!'
	@echo 'Have a look at .install.txt to know where'
	@echo 'StagPy has been installed'
	@echo
	@echo 'Add'
	@echo ' autoload bashcompinit'
	@echo ' bashcompinit'
	@echo ' eval "$$(register-python-argcomplete stagpy)"'
	@echo 'to your ~/.zshrc to enjoy command line completion with zsh!'
	@echo
	@echo 'Add'
	@echo ' eval "$$(register-python-argcomplete stagpy)"'
	@echo 'to your ~/.bashrc to enjoy command line completion with bash!'
