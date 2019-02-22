#!/bin/bash

curl -O https://www.manythings.org/anki/$1-eng.zip

mkdir -p data/$1
mv $1-eng.zip data/$1
cd data/$1/
unzip $1-eng.zip