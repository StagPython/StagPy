# Leave empty to install in virtualenv or as root
USR_FLAG=--user

.PHONY: all install config

all: install config

install:
	python2 setup.py install ${USR_FLAG}

config:
	./run.py config --create
