# Coypright 2019 IBM. All Rights Reserved.
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
Example of using the GraphDef editor and the Graph Transform Tool to prep an
object detection model for easy deployment.

This script starts with the pre-trained object detection model from the
TensorFlow Models repository; see https://github.com/tensorflow/models/
blob/master/research/object_detection/g3doc/detection_model_zoo.md.

Specifically, we use the object detector trained on the COCO dataset with a
MobileNetV1 architecture.

The original model takes as input batches of equal-sized images, represented
as a single dense numpy array of binary pixel data.  The output of the
original model represents the object type as an integer. This script grafts on
pre- and post-processing ops to make the input and output format more amenable
to use in applications. After these ops are added, the resulting graph takes a
single image file as an input and produces string-valued object labels.

To run this example from the root of the project, type:
   PYTHONPATH=$PWD env/bin/python examples/coco_example.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tensorflow as tf
import graph_def_editor as gde
import numpy as np
# noinspection PyPackageRequirements
import PIL  # Pillow
import shutil
import tarfile
from typing import List
import textwrap
import urllib.request

from tensorflow.tools import graph_transforms

FLAGS = tf.flags.FLAGS


def _indent(s):
  return textwrap.indent(str(s), "    ")


# Parameters of input graph
_LONG_MODEL_NAME = "ssd_mobilenet_v1_coco_2018_01_28"
_MODEL_TARBALL_URL = ("http://download.tensorflow.org/models/object_detection/" 
                      + _LONG_MODEL_NAME + ".tar.gz")
# Path to frozen graph within tarball
_FROZEN_GRAPH_MEMBER = _LONG_MODEL_NAME + "/frozen_inference_graph.pb"
_INPUT_NODE_NAMES = ["image_tensor"]
_OUTPUT_NODE_NAMES = ["detection_boxes", "detection_classes",
                      "detection_scores", "num_detections"]

_HASH_TABLE_INIT_OP_NAME = "hash_table_init"

# Label map for decoding label IDs in the output of the graph
_LABEL_MAP_URL = ("https://raw.githubusercontent.com/tensorflow/models/"
                  "f87a58cd96d45de73c9a8330a06b2ab56749a7fa/research/"
                  "object_detection/data/mscoco_label_map.pbtxt")


# Locations of intermediate files
_TMP_DIR = "/tmp/coco_example"
_SAVED_MODEL_DIR = _TMP_DIR + "/original_model"
_FROZEN_GRAPH_FILE = "{}/frozen_graph.pbtext".format(_TMP_DIR)
_PRE_POST_GRAPH_FILE = "{}/pre_and_post.pbtext".format(_TMP_DIR)
_TF_REWRITES_GRAPH_FILE = "{}/after_tf_rewrites_graph.pbtext".format(_TMP_DIR)
_GDE_REWRITES_GRAPH_FILE = "{}/after_gde_rewrites_graph.pbtext".format(_TMP_DIR)
_AFTER_MODEL_FILES = [
  _FROZEN_GRAPH_FILE, _TF_REWRITES_GRAPH_FILE, _GDE_REWRITES_GRAPH_FILE
]
# Panda pic from Wikimedia; also used in
# https://github.com/tensorflow/models/blob/master/research/slim/nets ...
#   ... /mobilenet/mobilenet_example.ipynb
_PANDA_PIC_URL = ("https://upload.wikimedia.org/wikipedia/commons/f/fe/"
                  "Giant_Panda_in_Beijing_Zoo_1.JPG")


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


def _fetch_or_use_cached(file_name, url):
  # type: (str, str) -> str
  """
  Check for a cached copy of the indicated file in our temp directory.

  If a copy doesn't exist, download the file.

  Arg:
    file_name: Name of the file within the temp dir, not including the temp
      dir path
    url: Full URL from which to download the file, including remote file
      name, which can be different from file_name

  Returns the path of the cached file.
  """
  cached_filename = "{}/{}".format(_TMP_DIR, file_name)
  if not os.path.exists(cached_filename):
    print("Downloading {} to {}".format(url, cached_filename))
    urllib.request.urlretrieve(url, cached_filename)
  return cached_filename


def _get_frozen_graph():
  # type: () -> tf.GraphDef
  """
  Obtains the starting version of the model from the TensorFlow model zoo

  Returns GraphDef
  """
  tarball = _fetch_or_use_cached("{}.tar.gz".format(_LONG_MODEL_NAME),
                                 _MODEL_TARBALL_URL)

  print("Original model files at {}".format(tarball))
  with tarfile.open(tarball) as t:
    frozen_graph_bytes = t.extractfile(_FROZEN_GRAPH_MEMBER).read()
    return tf.GraphDef.FromString(frozen_graph_bytes)


def _build_preprocessing_graph_def():
  # type: () -> tf.GraphDef
  """
  Build a TensorFlow graph that performs the preprocessing operations that
  need to happen before the main graph, then convert to a GraphDef.

  Returns:
    Python object representation of the GraphDef for the preprocessing graph.
    Input node of the graph is the placeholder "raw_image", and the output is
    the node with the name "preprocessed_image".
  """
  # At the moment, the only preprocessing we need to perform is converting
  # JPEG/PNG/GIF files to numpy arrays.
  img_decode_g = tf.Graph()
  with img_decode_g.as_default():
    raw_image = tf.placeholder(tf.string, name="raw_image")

    # Downstream code hardcodes RGB
    _NUM_CHANNELS = 3

    # The TensorFlow authors, in their infinite wisdom, created two generic
    # image-decoder ops. tf.image.decode_imaage() returns a 4D tensor when it
    # receives a GIF and a 3D tensor for every other file type. This means
    # that you need complicated shape-checking and reshaping logic downstream
    # for it to be of any use in an inference context.
    # The other op is tf.image.decode_png(). In spite of its name, this op
    # actually handles PNG, JPEG, and non-animated GIF files. For now, we use
    # this op for simplicity.
    decoded_image = tf.image.decode_png(raw_image, _NUM_CHANNELS)

    # Downstream code expects a batch of equal-sized images. For now, we
    # generate a single-image batch.
    decoded_image_batch = tf.expand_dims(decoded_image, 0,
                                         name="preprocessed_image")

  return img_decode_g.as_graph_def()


def _build_postprocessing_graph_def():
  # type: () -> tf.GraphDef
  """
  Build the TensorFlow graph that performs postprocessing operations that
  should happen after the main graph.

  Returns:
    Python object representation of the GraphDef for the postprocessing graph.
    The graph has one input placeholder called "detection_classes" and
    an output op called "decoded_detection_classes".
    The graph will also have an op called "hash_table_init" that initializes
    the mapping table. This op MUST be run exactly once before the
    "decoded_detection_classes" op will work.
  """
  label_file = _fetch_or_use_cached("labels.pbtext", _LABEL_MAP_URL)

  # Category mapping comes in pbtext format. Translate to the format that
  # TensorFlow's hash table initializers expect (key and value tensors).
  with open(label_file, "r") as f:
    raw_data = f.read()
  # Parse directly instead of going through the protobuf API dance.
  records = raw_data.split("}")
  records.pop(-1)  # Remove empty record at end
  records = [r.replace("\n", "") for r in records]  # Strip newlines
  regex = re.compile(r"item {  name: \".+\"  id: (.+)  display_name: \"(.+)\"")
  keys = []
  values = []
  for r in records:
    match = regex.match(r)
    keys.append(int(match.group(1)))
    values.append(match.group(2))

  result_decode_g = tf.Graph()
  with result_decode_g.as_default():
    # The original graph produces floating-point output for detection class,
    # even though the output is always an integer.
    float_class = tf.placeholder(tf.float32, shape=[None],
                                 name="detection_classes")
    int_class = tf.cast(float_class, tf.int32)
    key_tensor = tf.constant(keys, dtype=tf.int32)
    value_tensor = tf.constant(values)
    table_init = tf.contrib.lookup.KeyValueTensorInitializer(
      key_tensor,
      value_tensor,
      name=_HASH_TABLE_INIT_OP_NAME)
    hash_table = tf.contrib.lookup.HashTable(
      table_init,
      default_value="Unknown"
    )
    _ = hash_table.lookup(int_class, name="decoded_detection_classes")

  return result_decode_g.as_graph_def()


def _graft_pre_and_post_processing_to_main_graph(g):
  # type: (gde.Graph) -> None
  """
  Attach pre- and post-processing subgraphs to the main graph.

  Args:
    g: GDE representation of the core graph. Modified in place.
  """
  # Build the pre- and post-processing subgraphs and import into GDE
  pre_g = gde.Graph(_build_preprocessing_graph_def())
  post_g = gde.Graph(_build_postprocessing_graph_def())

  # Replace the graph's input placeholder with the contents of our
  # pre-processing graph.
  name_of_input_node = _INPUT_NODE_NAMES[0]
  gde.copy(pre_g, g)
  gde.reroute_ts(g.get_node_by_name("preprocessed_image").output(0),
                 g.get_node_by_name(name_of_input_node).output(0))
  g.remove_node_by_name(name_of_input_node)
  g.rename_node("raw_image", name_of_input_node)

  # Tack on the postprocessing graph at the original output and rename
  # the postprocessed output to the original output's name
  # The original graph produces an output called "detection_classes".
  # The postprocessing graph goes from "detection_classes" to
  # "decoded_detection_classes".
  # The graph after modification produces decoded classes under the original
  # "detection_classes" name. The original output is renamed to
  # "raw_detection_classes".
  g.rename_node("detection_classes", "raw_detection_classes")
  gde.copy(post_g, g)
  gde.reroute_ts(g.get_node_by_name("raw_detection_classes").output(0),
                 g.get_node_by_name("detection_classes").output(0))
  g.remove_node_by_name("detection_classes")
  g.rename_node("decoded_detection_classes", "detection_classes")


def _apply_graph_transform_tool_rewrites(g, input_node_names,
                                         output_node_names):
  # type: (gde.Graph, List[str], List[str]) -> tf.GraphDef
  """
  Use the [Graph Transform Tool](
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/
  graph_transforms/README.md)
  to perform a series of pre-deployment rewrites.

  Args:
     g: GDE representation of the core graph.
     input_node_names: Names of placeholder nodes that are used as inputs to
       the graph for inference. Placeholders NOT on this list will be
       considered dead code.
     output_node_names: Names of nodes that produce tensors that are outputs
       of the graph for inference purposes. Nodes not necessary to produce
       these tensors will be considered dead code.

  Returns: GraphDef representation of rewritten graph.
  """
  # Invoke the Graph Transform Tool using the undocumented Python APIs under
  # tensorflow.tools.graph_transforms
  after_tf_rewrites_graph_def = graph_transforms.TransformGraph(
    g.to_graph_def(),
    inputs=input_node_names,
    outputs=output_node_names,
    # Use the set of transforms recommended in the README under "Optimizing
    # for Deployment"
    transforms=['strip_unused_nodes(type=float, shape="1,299,299,3")',
                'remove_nodes(op=Identity, op=CheckNumerics)',
                'fold_constants(ignore_errors=true)',
                'fold_batch_norms',
                'fold_old_batch_norms']
  )
  return after_tf_rewrites_graph_def


def _graph_has_op(g, op_name):
  # type: (tf.Graph, str) -> bool
  """
  A method that really ought to be part of `tf.Graph`. Returns true of the
  indicated graph has an op by the indicated name.
  """
  all_ops_in_graph = g.get_operations()
  return any(op_name == o.name for o in all_ops_in_graph)


def _run_coco_graph(graph_proto, img):
  # type: (tf.GraphDef, np.ndarray) -> None
  """
  Run an example image through a TensorFlow graph and print a summary of
  the results to STDOUT.

  Only works for the graphs used in this example.

  graph_proto: GraphDef protocol buffer message holding serialized graph
  img: input image, either as a numpy array or a JPEG binary string
  """
  image_tensor_name = _INPUT_NODE_NAMES[0] + ":0"
  output_tensor_names = [n + ":0" for n in _OUTPUT_NODE_NAMES]
  with tf.Graph().as_default():
    with tf.Session() as sess:
      tf.import_graph_def(graph_proto, name="")

      # Initialize hash tables if present. Assumes that the init op is called
      # "hash_table_init"
      if _graph_has_op(tf.get_default_graph(), _HASH_TABLE_INIT_OP_NAME):
        sess.run(_HASH_TABLE_INIT_OP_NAME)

      results = sess.run(output_tensor_names, {image_tensor_name: img})

  bboxes, classes, scores, num_detections = results
  if len(classes.shape) > 1:
    # Results are a batch of length 1; unnest.
    bboxes = bboxes[0]
    classes = classes[0]
    scores = scores[0]
    num_detections = num_detections[0]

  # The num_detections output tells how much of the other output tensors is
  # used. The remaining rows of the tensors contain garbage. Print out the
  # non-garbage rows.
  print("Rank      Label               Weight    Bounding Box")
  for i in range(int(num_detections)):
    clazz = classes[i]  # "class" is a reserved word in Python
    if isinstance(clazz, bytes):
      clazz = clazz.decode("UTF-8")

    print("{:<10}{:<20}{:<10f}{}".format(
      i + 1, clazz, scores[i], bboxes[i]))


def main(_):
  # Remove any detritus of previous runs of this script, but leave the temp
  # dir in place because the user might have a shell there.
  if not os.path.isdir(_TMP_DIR):
    os.mkdir(_TMP_DIR)
  _clear_dir(_SAVED_MODEL_DIR)
  for f in _AFTER_MODEL_FILES:
    if os.path.isfile(f):
      os.remove(f)

  # We start with a frozen graph for the model. "Frozen" means that all
  # variables have been converted to constants.
  frozen_graph_def = _get_frozen_graph()

  # Wrap the initial GraphDef in a gde.Graph so we can examine it.
  frozen_graph = gde.Graph(frozen_graph_def)
  input_node_names = [n.name for n in
                      gde.filter_ops_by_optype(frozen_graph, "Placeholder")]
  # TODO: Devise an automatic way to find the outputs
  output_node_names = _OUTPUT_NODE_NAMES + [_HASH_TABLE_INIT_OP_NAME]
  print("Input names: {}".format(input_node_names))
  print("Output names: {}".format(output_node_names))

  _protobuf_to_file(frozen_graph_def, _FROZEN_GRAPH_FILE, "Frozen graph")

  # Graft the preprocessing and postprocessing graphs onto the beginning and
  # end of the inference graph.
  g = gde.Graph(frozen_graph_def)
  _graft_pre_and_post_processing_to_main_graph(g)
  after_add_pre_post_graph_def = g.to_graph_def()
  _protobuf_to_file(after_add_pre_post_graph_def, _PRE_POST_GRAPH_FILE,
                    "Graph with pre- and post-processing")

  # Now run through some of TensorFlow's built-in graph rewrites.
  after_tf_rewrites_graph_def = _apply_graph_transform_tool_rewrites(
    g, input_node_names, output_node_names)
  _protobuf_to_file(after_tf_rewrites_graph_def,
                    _TF_REWRITES_GRAPH_FILE,
                    "Graph after built-in TensorFlow rewrites")

  # Now run the GraphDef editor's graph prep rewrites
  g = gde.Graph(after_tf_rewrites_graph_def)
  gde.rewrite.fold_batch_norms(g)
  gde.rewrite.fold_old_batch_norms(g)
  gde.rewrite.fold_batch_norms_up(g)
  after_gde_graph_def = g.to_graph_def(add_shapes=True)
  _protobuf_to_file(after_gde_graph_def,
                    _GDE_REWRITES_GRAPH_FILE,
                    "Graph after GraphDef Editor rewrites")

  # Dump some statistics about the number of each type of op
  print("            Number of ops in frozen graph: {}".format(len(
    frozen_graph_def.node)))
  print(" Num. ops after adding pre- and post-proc: {}".format(len(
    after_add_pre_post_graph_def.node)))
  print("    Number of ops after built-in rewrites: {}".format(len(
    after_tf_rewrites_graph_def.node)))
  print("         Number of ops after GDE rewrites: {}".format(len(
    after_gde_graph_def.node)))

  # Run model before and after rewrite and compare results
  img_path = _fetch_or_use_cached("panda.jpg", _PANDA_PIC_URL)

  with open(img_path, "rb") as f:
    jpg_img = f.read()
  np_img_batch = np.expand_dims(np.array(PIL.Image.open(img_path)), axis=0)

  print("Frozen graph results:")
  _run_coco_graph(frozen_graph_def, np_img_batch)
  print("Results after adding pre/post-processing:")
  _run_coco_graph(after_add_pre_post_graph_def, jpg_img)
  print("Results after built-in rewrites:")
  _run_coco_graph(after_tf_rewrites_graph_def, jpg_img)
  print("Results after GDE rewrites:")
  _run_coco_graph(after_gde_graph_def, jpg_img)


if __name__ == "__main__":
  tf.app.run()
