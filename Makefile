LINK_DIR=~/bin
LINK_NAME=stagpy
LINK=${LINK_DIR}/${LINK_NAME}

VENV=venv
STAGPY=${VENV}/bin/stagpy

.PHONY: all install config clean uninstall

OBJS=setup.py stagpy/*.py

all: install config

config: ${STAGPY}
	${STAGPY} config --create

install: ${STAGPY}
	ln -fs ${PWD}/${STAGPY} ${LINK}

${STAGPY}: ${VENV}
	${VENV}/bin/python setup.py install

${VENV}:
	python2 -m virtualenv --system-site-packages $@

clean: uninstall
	@echo 'Removing build and virtualenv files'
	rm -rf build/ dist/ StagPy.egg-info/ ${VENV}
	rm -f *.pyc stagpy/*.pyc

uninstall:
	@echo 'Removing config file...'
	@rm -rf ~/.config/stagpy/
	@rm -f ${LINK}
	@echo 'Removing link...'
	@echo 'Done.'
