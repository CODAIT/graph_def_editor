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

Current status:

* All of the original project's regression tests pass
* Examples from the original project have not yet been ported

## Contents of root directory:

* LICENSE: This project is released under an Apache v2 license
* env: Not in git repo; create by running scripts/env.sh. Anaconda virtualenv
  for running tests and notebooks in this project.
* notebooks: Jupyter notebooks.
* gde: Source code for the Python package
* scripts: Useful shell and Python scripts
* tests: pytest tests. To run, create and activate

## IDE setup instructions

1. Install IntelliJ and the community Python plugin.
2. Run the script `scripts/env.sh` to create an Anaconda enviroment under `env`.
3. Import the root directory of this repository as a new project.
   Use the Anaconda environment at `env/bin/python` as the Python for
   the project.
4. In the "Project" view of IntelliJ, right-click on `env` and select 
   `Mark directory as ==> Excluded`. `env` shoud turn red.
5. Configure your editor to use 2 spaces for indents. Disable the PEP8 warnings
   in IntelliJ about indents not being a multiple of 4.
6. To run tests from within IntelliJ, open up the `Terminal` pane and type
   `./scripts/test.sh`. The outputs of the test run will be teed to the file
   `test.out` at the root of the project.




