"""Tool to export a quantized object detection model for inference."""

import os
import glob
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from google.protobuf import text_format

from object_detection import inputs
from object_detection import exporter
from object_detection import eval_util
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from object_detection.quantize import quantize_utils
from object_detection.metrics import coco_evaluation


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
      frozen_graph_def = quantize_utils.f_force_nms_cpu(frozen_graph_def)
  if replace_relu6:
      frozen_graph_def = quantize_utils.f_replace_relu6(frozen_graph_def)
  if remove_assert:
      frozen_graph_def = quantize_utils.f_remove_assert(frozen_graph_def)

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
        image = quantize_utils._read_image(path, calib_image_shape)
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


def benchmark_model(frozen_graph_def, pipeline_config_path):
  """Computes COCO evaluation and performance metrics on frozen graph.

  Args
  ----
      frozen_graph_def: A GraphDef representing the object detection model
      pipeline_config_path: path to the pipeline config file
      output_path: Optional string representing the output path to store
          evaluation and performance stats.

  Returns:
  --------
      statistics: a named dictionary of evaluation and performance metrics
      computed for the provided model.
  """
  configs = config_util.get_configs_from_pipeline_file(
    pipeline_config_path, config_override=None)
  model_config = configs["model"]
  eval_config = configs["eval_config"]
  eval_input_configs = configs["eval_input_configs"]

  # Create eval tf.data.Dataset
  # Note: this assumes we're using the first eval input configuration
  eval_dataset = inputs.create_eval_input_fn(
    eval_config=eval_config,
    eval_input_config=eval_input_configs[0],
    model_config=model_config)

  # Get graph and sess from graph def
  graph, sess = _load_model_from_graph_def(frozen_graph_def)

  # Input/output tensors
  tf_image_tensor = graph.get_tensor_by_name("image_tensor:0")
  tf_boxes = graph.get_tensor_by_name("detection_boxes:0")
  tf_scores = graph.get_tensor_by_name("detection_scores:0")
  tf_classes = graph.get_tensor_by_name("detection_classes:0")
  tf_num_detections = graph.get_tensor_by_name("num_detections:0")
  tf_masks = graph.get_tensor_by_name("detection_masks:0")
  detection_tensor_dict = {
    "boxes": tf_boxes,
    "image": tf_image_tensor,
    "scores": tf_scores,
    "classes": tf_classes,
    "num_detections": tf_num_detections,
    "masks": tf_masks
  }

  # Run inference on eval set
  # OPTIMIZE: currently we're doing batch_size=1 which slows things down
  image_ids = []
  gt_boxes = []
  gt_classes = []
  categories = []
  detection_boxes = []
  detection_scores = []
  detection_classes = []
  for features, labels in eval_dataset:
      image_ids.append(features[inputs.HASH_KEY].numpy()[0])
      gt_boxes.append(np.squeeze(labels["groundtruth_boxes"].numpy()))
      gt_classes.append(np.squeeze(labels["groundtruth_classes"].numpy()))
      categories.append({"id": 1, "name": "barcode"})
      # Run inference
      boxes, masks, scores, classes, num_detections = _detect(
        image=features["original_image"].numpy(),
        sess=sess,
        tensor_dict=detection_tensor_dict)
      detection_boxes.append(boxes)
      detection_scores.append(scores)
      detection_classes.append(classes)

  # Get COCO formatted groundtruth and detections
  groundtruth_dict = coco_tools.ExportGroundTruthToCOCO(
    image_ids=image_ids,
    groundtruth_boxes=gt_boxes,
    groundtruth_classes=gt_classes,
    categories=categories,
    output_path=None)
  detections_list = coco_tools.ExportDetectionsToCOCO(
    image_ids=image_ids,
    detection_boxes=detection_boxes,
    detection_scores=detection_scores,
    detection_classes=detection_classes,
    categories=categories,
    output_path=None)
  groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
  detections = groundtruth.LoadAnnotations(detections_list, custom_data=True)
  evaluator = coco_tools.COCOEvalWrapper(
    groundtruth, detections, agnostic_mode=False)
  metrics = evaluator.ComputeMetrics()


def _load_model_from_graph_def(graph_def):
  tf_config = tf.ConfigProto()
  tf_config.gpu_options.allow_growth = True
  tf_config.gpu_options.per_process_gpu_memory_fraction = 0.6

  graph = tf.Graph()
  with graph.as_default():
    tf.import_graph_def(graph_def)
  sess = tf.Session(graph=graph, config=tf_config)
  return graph, sess


def _detect(image, sess, tensor_dict):
  (boxes, masks, scores, classes, num_detections) = sess.run(
    [
      tensor_dict["boxes"],
      tensor_dict["masks"],
      tensor_dict["scores"],
      tensor_dict["classes"],
      tensor_dict["num_detections"]
    ],
    feed_dict={tensor_dict["image"]: image}
  )

  # Keep only num detections
  num_detections = int(np.squeeze(num_detections))
  masks = np.squeeze(masks)[:num_detections]
  masks = np.array(masks > 0.5, dtype=np.float32)
  boxes = np.squeeze(boxes)[:num_detections]
  scores = np.squeeze(scores)[:num_detections]
  classes = np.squeeze(classes)[:num_detections]

  return boxes, masks, scores, classes, num_detections
