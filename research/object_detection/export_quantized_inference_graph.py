"""Tool to export quantized object detection model for inference."""
import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.quantize import quantize
from object_detection.protos import pipeline_pb2

slim = tf.contrib.slim
flags = tf.app.flags


def main(_):
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(FLAGS.pipeline_config_path, "r") as f:
    text_format.Merge(f.read(), pipeline_config)
  text_format.Merge(FLAGS.config_override, pipeline_config)

  if FLAGS.input_shape:
      input_shape [
        int(dim) if dim != "-1" else None
        for dim in FLAGS.input_shape.split(",")
      ]
  else:
    input_shape = None

  # Export inference graph from checkpoint
  exporter.export_quantized_inference_graph(
    input_type=FLAGS.input_type,
    pipeline_config=pipeline_config,
    trained_checkpoint_prefix=FLAGS.trained_checkpoint_prefix,
    output_directory=FLAGS.output_directory,
    input_shape=input_shape,
    write_inference_graph=FLAGS.write_inference_graph
  )

  # Load frozen graph def
  frozen_graph_def_path = os.path.join(
    FLAGS.output_directory, "frozen_inference_graph.pb"
  )
  frozen_graph_def = tf.GraphDef()
  with open(frozen_graph_def_path, "rb") as f:
    frozen_graph_def.ParseFromString(f.read())

  # Quantize model
  frozen_graph_def = quantize.quantize_model(
    frozen_graph_def,
    precision_mode="INT8",
    calib_images_dir=FLAGS.calib_images_dir,
    num_calib_images=8,
    calib_batch_size=1,
    output_path=os.path.join(
      FLAGS.output_directory,"quantized_inference_graph.pb"
    )
  )

if __name__ == '__main__':
  tf.app.run()
