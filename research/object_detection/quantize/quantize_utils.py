"""Quantization utility functions"""

import cv2

import numpy as np
import tensorflow as tf


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


def _read_image(image_path, image_shape):
  """Load single or three channel image image in YUV color space"""
  yuv = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2YUV)
  ychan = yuv[:,:,0]
  # Single channel
  if image_shape[-1] == 1:
    return np.expand_dims(ychan, -1)
  elif image_shape[-1] == 3:
    return cv2.cvtColor(ychan, cv2.COLOR_GRAY2RGB)
