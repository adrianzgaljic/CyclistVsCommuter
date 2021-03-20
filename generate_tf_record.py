import argparse
import tensorflow as tf
import pandas as pd
import os
from object_detection.utils import dataset_util
import io
from PIL import Image
from object_detection.utils import label_map_util

NO_CLASSES = 2
cyclist_cnt = 0
commuters_cnt = 0

def parse_data(data):
    grouped = data.groupby("filename")
    return [[filename, grouped.get_group(filename)] for filename in grouped.groups]

def parse_labl_map(path_to_label_map):
    label_map = label_map_util.load_labelmap(path_to_label_map)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NO_CLASSES, use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)
    label_map = {}
    for k, v in category_index.items():
        label_map[v.get("name")] = v.get("id")

    return label_map

def create_record(image_data, img_dir, label_map):
    global cyclist_cnt, commuters_cnt
    file_name = image_data[0]
    labels_data = image_data[1]
    with tf.io.gfile.GFile(os.path.join(img_dir, '{}'.format(file_name)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    print(width, height)
    filename = file_name.encode('utf8')
    image_format = b'jpg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for i, row in labels_data.iterrows():
        try:
            class_id = label_map[row['class']]
            classes.append(class_id)
        except:
            print("No class with the name {}".format(row['class']))
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode("utf8"))
        if row['class'] == "cyclist":
            cyclist_cnt += 1
        else:
            commuters_cnt += 1

    tf_example = tf.train.Example(features=tf.train.Features(
        feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
    return tf_example


parser = argparse.ArgumentParser(description='Create a TFRecord file for use with the TensorFlow Object Detection API.')
parser.add_argument('--csv_file', metavar='csv_file', type=str)
parser.add_argument('--label_map', metavar='label_map', type=str)
parser.add_argument('--img_dir', metavar='img_dir', type=str)
parser.add_argument('--output_record', metavar='output_record', type=str)
args = parser.parse_args()


record_writer = tf.io.TFRecordWriter(args.output_record)

csv_data = pd.read_csv(args.csv_file)
parsed_data = parse_data(csv_data)

img_dir = os.path.join(args.img_dir)
label_map = parse_labl_map(args.label_map)

for image_data in parsed_data:
    #if image_data[0].endswith("jpeg"):
    record = create_record(image_data, img_dir, label_map)
    record_writer.write(record.SerializeToString())

record_writer.close()

print("Created {} cyclist labels".format(cyclist_cnt))
print("Created {} commuters labels".format(commuters_cnt))

