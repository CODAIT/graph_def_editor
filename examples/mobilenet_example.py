# Coypright 2018 IBM. All Rights Reserved.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Example of using the GraphDef editor and the Graph Transform Tool to prep a
copy of MobileNetV2 for inference.

Requires that the "Pillow" package be installed.

To run this example from the root of the project, type:
   PYTHONPATH=$PWD env/bin/python examples/mobilenet_example.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import graph_def_editor as gde
import numpy as np
# noinspection PyPackageRequirements
import PIL  # Pillow
import shutil
import tarfile
import textwrap
import urllib.request

from tensorflow.tools import graph_transforms

FLAGS = tf.flags.FLAGS


def _indent(s):
  return textwrap.indent(str(s), "    ")


_TMP_DIR = "/tmp/mobilenet_example"
_SAVED_MODEL_DIR = _TMP_DIR + "/original_model"
_FROZEN_GRAPH_FILE = "{}/frozen_graph.pbtext".format(_TMP_DIR)
_TF_REWRITES_GRAPH_FILE = "{}/after_tf_rewrites_graph.pbtext".format(_TMP_DIR)
_GDE_REWRITES_GRAPH_FILE = "{}/after_gde_rewrites_graph.pbtext".format(_TMP_DIR)
_AFTER_MODEL_FILES = [
  _FROZEN_GRAPH_FILE, _TF_REWRITES_GRAPH_FILE, _GDE_REWRITES_GRAPH_FILE
]
_USE_KERAS = False

# Panda pic from Wikimedia; also used in
# https://github.com/tensorflow/models/blob/master/research/slim/nets ...
#   ... /mobilenet/mobilenet_example.ipynb
_PANDA_PIC_URL = ("https://upload.wikimedia.org/wikipedia/commons/f/fe/"
                  "Giant_Panda_in_Beijing_Zoo_1.JPG")
_PANDA_PIC_FILE = _TMP_DIR + "/panda.jpg"

def _clear_dir(path):
  # type: (str) -> None
  if os.path.isdir(path):
    shutil.rmtree(path)
  os.mkdir(path)


def _protobuf_to_file(pb, path, human_readable_name):
  # type: (Any, str, str) -> None
  with open(path, "w") as f:
    f.write(str(pb))
  print("{} written to {}".format(human_readable_name, path))


def get_keras_frozen_graph():
  # type: () -> Tuple[tf.GraphDef, str, str]
  """
  Generate a frozen graph for the Keras MobileNet_v2 model.

  This should work, but does NOT work as of TensorFlow 1.12. The
  save_keras_model() function in TensorFlow creates a graph that the
  convert_variables_to_constants() function can't consume correctly.

  Returns GraphDef, input node name, output node name
  """
  # Start with the pretrained MobileNetV2 model from keras.applications,
  # wrapped as a tf.keras model.
  mobilenet = tf.keras.applications.MobileNetV2()
  # Because we're using a tf.keras model instead of a keras model,
  # the backing TensorFlow session (keras.backend.get_session()) will have a
  # ginormous graph with many unused nodes. The only supported API to filter
  # down that graph is to write the model out as a SavedModel "file". So
  # that's what we do here.
  # Note that save_keras_model() doesn't write the model to the path you told
  # it to use. It writes the model to a timestamped subdirectory and returns
  # the path of the subdirectory as a bytes object (NOT a string).
  actual_saved_model_directory_bytes = \
    tf.contrib.saved_model.save_keras_model(mobilenet, _SAVED_MODEL_DIR,
                                            as_text=True)
  print("Initial SavedModel file is at {}".format(tf.compat.as_str(
    actual_saved_model_directory_bytes)))
  # Now we need to freeze the graph, i.e. convert all variables to Const nodes.
  # The only supported way to do this with a Keras model is to write out a
  # SavedModel file, read the SavedModel file back into a fresh session,
  # and invoke the appropriate rewrite from tf.graph_util.
  with tf.Session() as sess:
    # save_keras_model() uses the "serve" tag for inference graphs, and the
    # names of the output nodes are the same as those returned by Model.outputs
    tf.saved_model.load(sess, tags=["serve"],
                        export_dir=actual_saved_model_directory_bytes)
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
      sess, sess.graph.as_graph_def(),
      output_node_names=[n.op.name for n in mobilenet.outputs])
  return (frozen_graph_def, mobilenet.inputs[0].op.name,
          mobilenet.outputs[0].op.name)


def get_slim_frozen_graph():
  # type: () -> Tuple[tf.GraphDef, str, str]
  """
  Obtains a MobileNet_v2 model from the TensorFlow model zoo

  Returns GraphDef, input op name, output op name
  """
  # Download a checkpoint if we don't have a cached one in our temp dir.
  # See
  # https://github.com/tensorflow/models/tree/master/research/slim/nets ...
  #             ... /mobilenet
  # for a full list of available checkpoints.
  _CHECKPOINT_NAME = "mobilenet_v2_1.0_224"
  _CHECKPOINT_URL = "https://storage.googleapis.com/mobilenet_v2/checkpoints" \
                    "/{}.tgz".format(_CHECKPOINT_NAME)
  _CHECKPOINT_TGZ = "{}/{}.tgz".format(_TMP_DIR, _CHECKPOINT_NAME)
  _FROZEN_GRAPH_MEMBER = "./{}_frozen.pb".format(_CHECKPOINT_NAME)

  if not os.path.exists(_CHECKPOINT_TGZ):
    urllib.request.urlretrieve(_CHECKPOINT_URL, _CHECKPOINT_TGZ)
  with tarfile.open(_CHECKPOINT_TGZ) as t:
    frozen_graph_bytes = t.extractfile(_FROZEN_GRAPH_MEMBER).read()
    return (tf.GraphDef.FromString(frozen_graph_bytes),
            "input", "MobilenetV2/Predictions/Reshape_1")


def run_graph(graph_proto, img, input_node, output_node):
  # type: (tf.GraphDef, np.ndarray, str, str) -> None
  """
  Run an example image through a MobileNet-like graph and print a summary of
  the results to STDOUT.

  graph_proto: GraphDef protocol buffer message holding serialized graph
  img: Preprocessed (centered by dividing by 128) numpy array holding image
  input_node: Name of input graph node
  output_node: Name of output graph node; should produce logits
  """
  img_as_batch = img.reshape(tuple([1] + list(img.shape)))
  with tf.Graph().as_default():
    with tf.Session() as sess:
      tf.import_graph_def(graph_proto, name="")
      result = sess.run(output_node + ":0", {input_node + ":0": img_as_batch})

  result = result.reshape(result.shape[1:])
  # print("Raw result is {}".format(result))
  sorted_indices = result.argsort()
  # print("Top 5 indices: {}".format(sorted_indices[-5:]))

  print("Rank      Label     Weight")
  for i in range(5):
    print("{:<10}{:<10}{}".format(i + 1, sorted_indices[-(i + 1)],
                                  result[sorted_indices[-(i + 1)]]))


def main(_):
  # Remove any detritus of previous runs of this script, but leave the temp
  # dir in place because the user might have a shell there.
  if not os.path.isdir(_TMP_DIR):
    os.mkdir(_TMP_DIR)
  _clear_dir(_SAVED_MODEL_DIR)
  for f in _AFTER_MODEL_FILES:
    if os.path.isfile(f):
      os.remove(f)

  # Obtain a frozen graph for a MobileNet model
  if _USE_KERAS:
    frozen_graph_def, input_node, output_node = get_keras_frozen_graph()
  else:
    frozen_graph_def, input_node, output_node = get_slim_frozen_graph()

  _protobuf_to_file(frozen_graph_def, _FROZEN_GRAPH_FILE, "Frozen graph")

  # Now run through some of TensorFlow's built-in graph rewrites.
  # For that we use the undocumented Python APIs under
  # tensorflow.tools.graph_transforms
  after_tf_rewrites_graph_def = graph_transforms.TransformGraph(
    frozen_graph_def,
    inputs=[input_node],
    outputs=[output_node],
    # Use the set of transforms recommended in the README under "Optimizing
    # for Deployment"
    transforms=['strip_unused_nodes(type=float, shape="1,299,299,3")',
                'remove_nodes(op=Identity, op=CheckNumerics)',
                'fold_constants(ignore_errors=true)',
                'fold_batch_norms',
                'fold_old_batch_norms']
  )

  _protobuf_to_file(after_tf_rewrites_graph_def,
                    _TF_REWRITES_GRAPH_FILE,
                    "Graph after built-in TensorFlow rewrites")

  # Now run the GraphDef editor's fold_batch_norms_up() rewrite
  g = gde.Graph(after_tf_rewrites_graph_def)
  gde.rewrite.fold_batch_norms(g)
  gde.rewrite.fold_old_batch_norms(g)
  gde.rewrite.fold_batch_norms_up(g)
  after_gde_graph_def = g.to_graph_def(add_shapes=True)

  _protobuf_to_file(after_gde_graph_def,
                    _GDE_REWRITES_GRAPH_FILE,
                    "Graph after fold_batch_norms_up() rewrite")

  # Dump some statistics about the number of each type of op
  print("            Number of ops in frozen graph: {}".format(len(
    frozen_graph_def.node)))
  print("    Number of ops after built-in rewrites: {}".format(len(
    after_tf_rewrites_graph_def.node)))
  print("Number of ops after GDE rewrites: {}".format(len(
    after_gde_graph_def.node)))

  # Run model before and after rewrite and compare results
  if not os.path.exists(_PANDA_PIC_FILE):
    print("Downloading {} to {}".format(_PANDA_PIC_URL, _PANDA_PIC_FILE))
    urllib.request.urlretrieve(_PANDA_PIC_URL, _PANDA_PIC_FILE)
  img = np.array(PIL.Image.open(_PANDA_PIC_FILE).resize((224, 224))).astype(
    np.float) # / 128 # - 1
  # Normalize each channel
  channel_means = np.mean(img, axis=(0, 1))

  print("Channel means are: {}".format(channel_means))
  print("Image shape is {}".format(img.shape))

  print("Frozen graph results:")
  run_graph(frozen_graph_def, img, input_node, output_node)
  print("Results after built-in rewrites:")
  run_graph(after_tf_rewrites_graph_def, img, input_node, output_node)
  print("Results after GDE rewrites:")
  run_graph(after_gde_graph_def, img, input_node, output_node)


if __name__ == "__main__":
  tf.app.run()
