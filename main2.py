import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from object_detection.utils import visualization_utils as viz_utils
import argparse
from object_detection.utils import label_map_util


def load_image_into_numpy_array(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', metavar='model', type=str)
    parser.add_argument('--label_map', metavar='label_map', type=str)
    parser.add_argument('--input_image', metavar='input_image', type=str)
    args = parser.parse_args()

    detect_fn = tf.saved_model.load(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.label_map, use_display_name=True)
    image_np = load_image_into_numpy_array(args.input_image)

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()
    image_np_with_detections = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.40,
        agnostic_mode=False)
    return detections, image_np_with_detections


if __name__ == "__main__":
    main()
