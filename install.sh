#!/bin/sh

linkDir=~/bin
linkName=stagpy

mkdir -p ${linkDir}
ln -fs ${PWD}/main.py ${linkDir}/${linkName}
chmod u+x main.py
./main.py config --create
