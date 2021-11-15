#!/bin/bash
cd "$(dirname '$0')"
echo "Current Path : $(pwd)"
pip3 install --quiet pipenv
# pipenv commands
pipenv run pipenv install --dev
pipenv run pipenv sync