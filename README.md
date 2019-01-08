# GraphDef Editor (gde)
### A port of the old TensorFlow `contrib.graph_editor` that operates over serialized graphs

TensorFlow versions prior to version 2.0 had a Python graph editor in
`contrib.graph_editor`. This functionality was removed in TensorFlow 2.0.
This project brings back the graph editor as a standalone Python package.

The original graph editor operated over TensorFlow's Python classes `Graph`,
`Variable`, `Operator`, etc., often poking into the internals of these classes. 
As a result of this design, the graph editor needed to be updated whenever the
underlying classes changed.

Unlike the original graph editor, the GraphDef Editor operates over 
*serialized* TensorFlow graphs. Although TensorFlow's serialization format is
not (currently) a public API, this format changes much less frequently than the
Python classes that the original graph editor depended on.

TODO: Example usage

## Project status

**This project is a work in progress.**

Current status: 16 of 50 original regression tests passing.

## Contents of root directory:

* env: Not in git repo; create by running scripts/env.sh. Anaconda virtualenv
  for running tests and notebooks in this project.
* notebooks: Jupyter notebooks.
* gde: Source code for the Python package
* scripts: Useful shell and Python scripts
* tests: pytest tests. To run, create and activate




