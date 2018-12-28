# pge: Portable Graph Editor
### A port of the old TensorFlow `contrib.graph_editor` to use public APIs

TensorFlow versions prior to version 2.0 had a Python graph editor in
`contrib.graph_editor`. This functionality was removed in TensorFlow 2.0.
This project brings back the graph editor as a standalone Python package.

Unlike the original graph editor, this package depends only on public APIs of
TensorFlow. The editor uses the SavedModel format for both input and output.

TODO: Example usage

## Project status

**This project is a work in progress.**

Current status: 13 of 50 original regression tests passing.

## Contents of root directory:

* env: Not in git repo; create by running scripts/env.sh. Anaconda virtualenv
  for running tests and notebooks in this project.
* notebooks: Jupyter notebooks.
* pge: Source code for the Python package
* scripts: Useful shell and Python scripts
* tests: pytest tests. To run, create and activate




