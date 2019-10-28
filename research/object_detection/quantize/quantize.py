"""Tool to export a quantized object detection model for inference."""

import os
import glob
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from PIL import Image

from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2


def make_const6(const6_name="const6"):
  graph = tf.Graph()
  with graph.as_default():
    tf_6 = tf.constant(dtype=tf.float32, value=6.0, name=const6_name)
  return graph.as_graph_def()


def make_relu6(output_name, input_name, const6_name="const6"):
  """Make a relu6(x) = relu(x) - relu(x-6) op."""
  graph = tf.Graph()
  with graph.as_default():
    tf_x = tf.placeholder(tf.float32, [10, 10], name=input_name)
    tf_6 = tf.constant(dtype=tf.float32, value=6.0, name=const6_name)
    with tf.name_scope(output_name):
      tf_y1 = tf.nn.relu(tf_x, name="relu1")
      tf_y2 = tf.nn.relu(tf.subtract(tf_x, tf_6, name="sub1"), name="relu2")

  graph_def = graph.as_graph_def()
  graph_def.node[-1].name = output_name

  # Remove unsed nodes
  for node in graph_def.node:
    if node.name == input_name:
      graph_def.node.remove(node)
  for node in graph_def.node:
    if node.name == const6_name:
      graph_def.node.remove(node)
  for node in graph_def.node:
      if node.op == "_Neg":
        node.op = "Neg"
  return graph_def


def convert_relu6(graph_def, const6_name="const6"):
  """Convert relu6(x) into relu(x) - relu(6-x) op."""
  # Add constant 6
  has_const6 = False
  for node in graph_def.node:
    if node.name == const6_name:
      has_const6 = True
  if not has_const6:
    const6_graph_def = make_const6(const6_name=const6_name)
    graph_def.node.extend(const6_graph_def.node)

  for node in graph_def.node:
    if node.op == "Relu6":
      input_name = node.input[0]
      output_name = node.name
      relu6_graph_def = make_relu6(
        output_name, input_name, const6_name=const6_name
      )
      graph_def.node.remove(node)
      graph_def.node.extend(relu6_graph_def.node)
  return graph_def


def remove_node(graph_def, node):
  """Remove node from graph_def"""
  for n in graph_def.node:
    if node.name in n.input:
      n.input.remove(node.name)
  ctrl_name = '^' + node.name
  if ctrl_name in n.input:
    n.input.remove(ctrl_name)
  graph_def.node.remove(node)


def remove_op(graph_def, op_name):
  """Remove op from graph_def"""
  matches = [node for node in graph_def.node if node.op == op_name]
  for match in matches:
    remove_node(graph_def, match)


def f_force_nms_cpu(frozen_graph):
  """Force Non-max-NonMaxSuppression onto CPU"""
  for node in frozen_graph.node:
    if 'NonMaxSuppression' in node.name:
      node.device = '/device:CPU:0'
  return frozen_graph


def f_replace_relu6(frozen_graph):
  return convert_relu6(frozen_graph)


def f_remove_assert(frozen_graph):
  remove_op(frozen_graph, 'Assert')
  return frozen_graph


def quantize_model(frozen_graph_def,
                   force_nms_cpu=True,
                   replace_relu6=True,
                   remove_assert=True,
                   precision_mode='FP32',
                   minimum_segment_size=2,
                   max_workspace_size_bytes=1 << 32,
                   maximum_cached_engines=100,
                   calib_images_dir=None,
                   num_calib_images=None,
                   calib_batch_size=1,
                   calib_image_shape=None,
                   output_path=None):
  """Quantizes object detection model object detection model using TensorRT.

  In addition this methods also performs pre-tensorrt optimizations specific
  to the TensorFlow object detection API models.

  Args
  ----
      frozen_graph: A GraphDef representing the optimized model.
      force_nms_cpu: A boolean indicating whether to place NMS operations on
          the CPU.
      replace_relu6: A boolean indicating whether to replace relu6(x)
          operations with relu(x) - relu(x-6).
      remove_assert: A boolean indicating whether to remove Assert
          operations from the graph.
      precision_mode: A string representing the precision mode to use for
          TensorRT optimization.  Must be one of 'FP32', 'FP16', or 'INT8'.
      minimum_segment_size: An integer representing the minimum segment size
          to use for TensorRT graph segmentation.
      max_workspace_size_bytes: An integer representing the max workspace
          size for TensorRT optimization.
      maximum_cached_engines: An integer represenging the number of TRT engines
          that can be stored in the cache.
      calib_images_dir: A string representing a directory containing images to
          use for int8 calibration.
      num_calib_images: An integer representing the number of calibration
          images to use.  If None, will use all images in directory.
      calib_batch_size: An integer representing the batch size to use for calibration.
      calib_image_shape: A tuple of integers representing the height,
          width that images will be resized to for calibration.
      output_path: An optional string representing the path to save the
          optimized GraphDef to.
  Returns
  -------
      A GraphDef representing the optimized model.
  """
  # Apply optional graph modifications
  if force_nms_cpu:
      frozen_graph_def = f_force_nms_cpu(frozen_graph_def)
  if replace_relu6:
      frozen_graph_def = f_replace_relu6(frozen_graph_def)
  if remove_assert:
      frozen_graph_def = f_remove_assert(frozen_graph_def)

  # Object detection ouput names
  output_names = [
    "detection_boxes",
    "detection_classes",
    "detection_scores",
    "num_detections"
  ]

  # Record pre-tensorrt graph size and nodes
  pre_trt_graph_size = len(frozen_graph_def.SerializeToString())
  pre_trt_num_nodes = len(frozen_graph_def.node)
  start_time = time.time()

  # Converter
  converter = trt.TrtGraphConverter(
    input_graph_def=frozen_graph_def,
    nodes_blacklist=output_names,
    max_workspace_size_bytes=max_workspace_size_bytes,
    precision_mode=precision_mode,
    minimum_segment_size=minimum_segment_size,
    is_dynamic_op=True,
    maximum_cached_engines=maximum_cached_engines
  )
  frozen_graph_def = converter.convert()

  end_time = time.time()

  # Record post-trt graph size and nodes
  post_trt_graph_size = len(frozen_graph_def.SerializeToString())
  tftrt_num_nodes = len(frozen_graph_def.node)
  trt_num_nodes = len(
    [1 for n in frozen_graph_def.node if str(n.op)=="TRTEngineOp"]
  )

  print(f"Graph size (MB) (Native TF) {float(pre_trt_graph_size/(1<<20))}")
  print(f"Graph size (MB) (TRT) {float(post_trt_graph_size/(1<<20))}")
  print(f"Num nodes (Native TF) {pre_trt_num_nodes}")
  print(f"Num nodes (TFTRT Total) {tftrt_num_nodes}")
  print(f"Num nodes (TRT Only) {trt_num_nodes}")

  # Perform calibration for UINT8 precision
  if precision_mode == "INT8":
    if calib_images_dir is None:
      raise ValueError("calib_images_dir must be provided for INT8 optimization")
    image_paths = glob.glob(os.path.join(calib_images_dir, "*.jpg"))
    image_paths.extend(glob.glob(os.path.join(calib_images_dir, "*.png")))
    if len(image_paths) == 0:
      raise ValueError("No images were found in calib_images_dir")
    image_paths = image_paths[:num_calib_images]
    num_batches = len(image_paths) // calib_batch_size

    def feed_dict_fn():
      batch_images = []
      for path in image_paths[feed_dict_fn.index:feed_dict_fn.index+calib_batch_size]:
        image = _read_image(image_path, calib_image_shape)
        batch_images.append(image)
      feed_dict_fn.index += calib_batch_size
      return {"image_tensor:0": np.array(batch_images)}
    feed_dict_fn.index = 0

    print("Calibrating INT8...")
    start_time = time.time()
    frozen_graph_def = converter.calibrate(
      fetch_names=[x + ":0" for x in output_names],
      num_runs=num_batches,
      feed_dict_fn=feed_dict_fn
    )
    calibration_time = time.time()
    print(f"time (s) (trt_calibration) {calibration_time:.4f}")

  # Write optimized model to disk
  if output_path is not None:
    with open(output_path, "wb") as f:
      f.write(frozen_graph_def.SerializeToString())

  return frozen_graph_def


def _read_image(image_path, image_shape):
  image = Image.open(image_path).convert("RGB")
  if image_shape is not None:
    image = image.resize(image_shape[::-1])
  return np.array(image)
