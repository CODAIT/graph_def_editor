################################################################################
# test.sh
#
# Run regression tests for this project.
#
# Usage:
#   ./scripts/test.sh
#

#conda activate ./env
./env/bin/pytest --ignore=env | tee test.out
#conda deactivate

