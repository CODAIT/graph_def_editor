# GraphDef Editor
### A port of the old TensorFlow `contrib.graph_editor` that operates over serialized graphs

TensorFlow versions prior to version 2.0 had a Python graph editor in
`contrib.graph_editor`. This functionality is slated to be removed in 
TensorFlow 2.0, along with the rest of the `contrib` package (see the 
[RFC](https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md).)
This project brings back the graph editor as a standalone Python package.

The original graph editor operated over TensorFlow's Python classes `Graph`,
`Variable`, `Operator`, etc., often poking into the internals of these classes. 
As a result of this design, the graph editor needed to be updated whenever the
underlying classes changed.

The GraphDef Editor operates over *serialized* TensorFlow graphs represented as
`GraphDef` protocol buffer messages. Although TensorFlow's serialization format 
is technically not a public API, there is public 
[documentation](https://www.tensorflow.org/guide/extend/model_files) 
for its structure, and the format changes much less frequently than the Python 
classes that the original graph editor depended on. TensorFlow's C++ 
[Graph Transform Tool](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md)
also operates over serialized graphs.

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




