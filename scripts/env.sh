#! /bin/bash

################################################################################
# env.sh
#
# Set up an Anaconda virtualenv in the directory ./env
#
# Run this script from the root of the project, i.e.
#   ./scripts/env.sh
#
# Requires that conda be installed and set up for calling from bash scripts.
#
# Also requires that you set the environment variable CONDA_HOME to the
# location of the root of your anaconda/miniconda distribution.
################################################################################

PYTHON_VERSION=3.6

############################
# HACK ALERT *** HACK ALERT 
# The friendly folks at Anaconda thought it would be a good idea to make the
# "conda" command a shell function. 
# See https://github.com/conda/conda/issues/7126
# The following workaround will probably be fragile.
if [ -z "$CONDA_HOME" ]
then 
    echo "Error: CONDA_HOME not set"
    exit
fi
. ${CONDA_HOME}/etc/profile.d/conda.sh
# END HACK
############################

################################################################################
# Remove any previous outputs of this script

rm -rf ./env


################################################################################
# Create the environment
conda create -y --prefix ./env \
    python=${PYTHON_VERSION} \
    numpy \
    tensorflow \
    jupyterlab \
    pytest \
    keras \
    pillow \
    nomkl

echo << EOM
Anaconda virtualenv installed in ./env.
Run \"conda activate ./env\" to use it.
EOM

