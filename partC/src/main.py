import cv2
import time
import argparse
import numpy as np
import seaborn as sns
import os

def load_input_image(image_path):
    test_img = cv2.imread(image_path)
    h, w, _ = test_img.shape

    return test_img, h, w


def yolov3(yolo_weights, yolo_cfg, coco_names):
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    classes = open(coco_names).read().strip().split("\n")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, classes, output_layers


def perform_detection(net, img, output_layers, w, h, confidence_threshold):
    blob = cv2.dnn.blobFromImage(img, 1 / 255., (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Object is deemed to be detected
            if confidence > confidence_threshold:
                # center_x, center_y, width, height = (detection[0:4] * np.array([w, h, w, h])).astype('int')
                center_x, center_y, width, height = list(map(int, detection[0:4] * [w, h, w, h]))
                # print(center_x, center_y, width, height)

                top_left_x = int(center_x - (width / 2))
                top_left_y = int(center_y - (height / 2))

                boxes.append([top_left_x, top_left_y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids


def draw_boxes_labels(boxes, confidences, class_ids, classes, img, colors, confidence_threshold, NMS_threshold):

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, NMS_threshold)

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            # print(len(colors[class_ids[i]]))
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # text = f"{class_ids[i]} -- {confidences[i]}"
            text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), FONT, 0.5, color, 2)

    return img


def dectection_video_file(webcam, video_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold):
    net, classes, output_layers = yolov3(yolo_weights, yolo_cfg, coco_names)
    # colors = np.random.uniform(0, 255, size=(len(classes), 3))
    colors = ((np.array(color_palette("husl", len(classes))) * 255)).astype(np.uint8)

    if webcam:
        video = cv2.VideoCapture(0)
        time.sleep(2.0)
        if (video.isOpened() == False): 
            print("Error opening camera")
            return
    else:
        video = cv2.VideoCapture(video_path)
    
        if (video.isOpened() == False): 
            print("Error reading video file")
            return

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    out_dir = os.path.join(os.path.dirname(video_path), '..', 'output')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = os.path.splitext(os.path.basename(video_path))[0] + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(os.path.join(out_dir, '..', out_name), fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, image = video.read()
        if ret == False:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        h, w, _ = image.shape
        boxes, confidences, class_ids = perform_detection(net, image, output_layers, w, h, confidence_threshold)
        final_img = draw_boxes_labels(boxes, confidences, class_ids, classes, image, colors, confidence_threshold, nms_threshold)
        cv2.imshow("Detection", final_img)
        out.write(final_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()


def detection_image_file(image_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold):
    img, h, w = load_input_image(image_path)
    net, classes, output_layers = yolov3(yolo_weights, yolo_cfg, coco_names)
    # colors = np.random.uniform(0, 255, size=(len(classes), 3))
    colors = ((np.array(color_palette("husl", len(classes))) * 255)).astype(np.uint8)
    boxes, confidences, class_ids = perform_detection(net, img, output_layers, w, h, confidence_threshold)
    final_img = draw_boxes_labels(boxes, confidences, class_ids, classes, img, colors, confidence_threshold, nms_threshold)
    cv2.imshow("Detection", final_img)
    final_img_dir = os.path.join(os.path.dirname(image_path), '..', 'output')
    if not os.path.exists(final_img_dir):
        os.makedirs(final_img_dir)
    cv2.imwrite("", os.path.join(final_img_dir, os.path.basename(image_path)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ## Arguments to give before running
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', help='Path to video file', default=None)
    ap.add_argument('--image', help='Path to the test images', default=None)
    ap.add_argument('--camera', help='To use the live feed from web-cam', type=bool, default=False)
    ap.add_argument('--weights', help='Path to model weights', type=str, default='../pre-trained-model/yolov3.weights')
    ap.add_argument('--configs', help='Path to model configs',type=str, default='../pre-trained-model/yolov3.cfg')
    ap.add_argument('--class_names', help='Path to class-names text file', type=str, default='../pre-trained-model/coco.names')
    ap.add_argument('--conf_thresh', help='Confidence threshold value', default=0.5)
    ap.add_argument('--nms_thresh', help='Confidence threshold value', default=0.4)
    args = vars(ap.parse_args())

    image_path = args['image']
    yolo_weights, yolo_cfg, coco_names = args['weights'], args['configs'], args['class_names']
    confidence_threshold = args['conf_thresh']
    nms_threshold = args['nms_thresh']


    if image_path is None and args['camera'] == False and args['video'] is None:
        print('No input provided to apply the model')

    if image_path:
        detection_image_file(image_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold)

    elif args['camera'] == True or args['video']:
        webcam = args['camera']
        video_path = args['video']
        dectection_video_file(webcam, video_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold)




