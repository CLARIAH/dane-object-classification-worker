#!/bin/sh

# use this script to use local development version of:
# - dane

poetry remove dane
poetry add ../DANE/dist/*.whl