#!/bin/sh

# This script needs to be run BEFORE you're pushing changes to the main branch
# - it makes sure to link up the correct DANE version (main branch)

if poetry remove dane; then
    echo "successfully uninstalled dane"
else
    echo "already uninstalled"
fi

poetry add git+https://git@github.com/CLARIAH/DANE.git#main