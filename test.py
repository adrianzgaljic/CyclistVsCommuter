import argparse
import cv2
import detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', metavar='model', type=str)
    parser.add_argument('--label_map', metavar='label_map', type=str)
    parser.add_argument('--input_image', metavar='input_image', type=str)
    args = parser.parse_args()

    detections, image_np_with_detections = detector.detect(args.model, args.label_map, args.input_image)

    cv2.imshow("image with detections", image_np_with_detections)
    cv2.waitKey(0)



if __name__ == "__main__":
    main()