from flask import Flask, jsonify, request
import numpy as np
import os
import io
import base64
import logging

import sys
sys.path.append("..")
import tensorflow as tf
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


app = Flask(__name__)

@app.route('/', methods=["GET","POST"])
def func():
    print("qaq")
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            frame = base64.b64decode(request.get_data())
            image = Image.frombytes('RGB', (1280,720),frame) 
            image_np = np.asarray(image)
            # image_np = image_np.reshape((720,1080,3))
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            vec_ret = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
    return jsonify(vec_ret)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


if __name__ == '__main__':
    MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    IMAGE_SIZE = (12, 8)
    app.debug = False
    app.run('0.0.0.0')

