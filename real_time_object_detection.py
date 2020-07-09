import numpy as np
import os
import tensorflow as tf
import cv2
import argparse
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,
#     help="path to input video")
# ap.add_argument("-o", "--output", required=True,
#     help="path to output video")
# # ap.add_argument("-c", "--confidence", type=float, default=0.5,
# #     help="minimum probability to filter weak detections")
# # ap.add_argument("-t", "--threshold", type=float, default=0.3,
# #     help="threshold when applyong non-maxima suppression")
# args = vars(ap.parse_args())

# path to the frozen graph:
PATH_TO_FROZEN_GRAPH = 'frozen_inference_graph.pb'

# path to the label map
PATH_TO_LABEL_MAP = 'labelmap.pbtxt'

# number of classes 
NUM_CLASSES = 5

cap = cv2.VideoCapture(0)

#reads the frozen graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            # Read frame from camera
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detections
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=3)

        # Display output
            cv2.imshow('Object Detection', cv2.resize(image_np, (900, 900)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
