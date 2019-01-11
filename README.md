# GraphDef Editor
### A port of the TensorFlow `contrib.graph_editor` package that operates over serialized graphs

TensorFlow versions prior to version 2.0 had a Python graph editor in
`contrib.graph_editor`. This functionality is slated to be removed in 
TensorFlow 2.0, along with the rest of the `contrib` package (see the 
[RFC](https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md)).
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

Example usage:
```python
import tensorflow as tf
import graph_def_editor as gde
# Create a graph
tf_g = tf.Graph()
with tf_g.as_default():
  a = tf.constant(1.0, shape=[2, 3], name="a")
  c = tf.add(
      tf.placeholder(dtype=np.float32),
      tf.placeholder(dtype=np.float32),
      name="c")

# Serialize the graph
g = gde.Graph(tf_g.as_graph_def())

# Modify the graph.
# In this case we replace the two input placeholders with constants.
# One of the constants (a) is a node that was in the original graph.
# The other one (b) we create here.
b = gde.make_const(g, "b", np.full([2, 3], 2.0, dtype=np.float32))
gde.swap_inputs(g[c.op.name], [g[a.name], b.output(0)])

# Reconstitute the modified serialized graph as TensorFlow graph...
with g.to_tf_graph().as_default():
  # ...and print the value of c, which should be 2x3 matrix of 3.0's
  with tf.Session() as sess:
    res = sess.run(c.name)
    print("Result is:\n{}".format(res))
```

## Project status

**This project is a work in progress.**

Current status:

* All of the original project's regression tests pass.
* The simple example script from the original project runs.

## Contents of root directory:

* `LICENSE`: This project is released under an Apache v2 license
* `env`: Not in git repo; create by running `scripts/env.sh`. Anaconda virtualenv
  for running tests and notebooks in this project.
* `examples`: Example scripts.  To run these scripts from the root directory use,
  the command 
  ```
  PYTHONPATH=$PWD env/bin/python examples/script_name.py
  ```
  where `script_name.py` is the name of the example script.
* `notebooks`: Jupyter notebooks.
* `graph\_def\_editor`: Source code for the Python package
* `scripts`: Useful shell scripts for development.
* `setup.py`: Setup script to make this project pip-installable with 
   [`setuptools`](https://setuptools.readthedocs.io/en/latest/)
* `tests`: pytest tests. To run these tests, create `env` and run
  `scripts/test.sh`

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




