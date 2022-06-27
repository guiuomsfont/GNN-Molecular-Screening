#!/bin/bash
#
# Quick and dirty script to automate all steps for building/installing the package
# 
# (c) Grupo AIA / RTE
#     goms@aia.es
#

# For saner programming:
set -o nounset -o noclobber
set -o errexit -o pipefail


PKG="GNN_DTI"
SCRIPT_PATH=$(realpath "$0")
MY_LOCAL_REPO=$(dirname "$SCRIPT_PATH")


GREEN="\\033[1;32m"
NC="\\033[0m"
colormsg()
{
    echo -e "${GREEN}$1${NC}"
}
colormsg_nnl()
{
    echo -n -e "${GREEN}$1${NC}"
}


# Step 0: reminder to refresh your local workspace
echo "You're about to build & reinstall: $PKG  (remember to refresh your local repo if needed)"
read -p "Are you sure? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  exit
fi
pip install --upgrade pip wheel setuptools build

# Step 1: build
echo
colormsg "Building the package... "
cd "$MY_LOCAL_REPO" && python3 -m build
colormsg "OK."


# Step 2: install the package
echo
colormsg "Installing the package... "
pip uninstall "$PKG" 
pip install "$MY_LOCAL_REPO"/dist/*.whl
colormsg "OK."


# Step 3: upgrade all deps
echo
colormsg "Upgrading all dependencies... "
pip install -U --upgrade-strategy eager  "$PKG"
colormsg "OK."

