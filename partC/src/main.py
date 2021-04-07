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


def draw_boxes_labels(boxes, confidences, class_ids, classes, img, colors, confidence_threshold, NMS_threshold, disp_scores):

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, NMS_threshold)

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            # print(len(colors[class_ids[i]]))
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # text = f"{class_ids[i]} -- {confidences[i]}"
            text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i]) if disp_scores else "{}".format(classes[class_ids[i]])
            (text_w, text_h) = cv2.getTextSize(text, FONT, fontScale=0.6, thickness=2)[0]
            text_offset_x, text_offset_y = 4, -4
            cv2.rectangle(img, (x-1, y-1), (x + text_offset_x + text_w + 4, y + text_offset_y - text_h - 5), color, cv2.FILLED)
            cv2.putText(img, text, (x + text_offset_x, y + text_offset_y - 1), FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    return img


def dectection_video_file(webcam, video_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold, disp_scores):
    net, classes, output_layers = yolov3(yolo_weights, yolo_cfg, coco_names)
    # colors = np.random.uniform(0, 255, size=(len(classes), 3))
    colors = ((np.array(sns.color_palette("husl", len(classes))) * 255)).astype(int)

    if webcam:
        video = cv2.VideoCapture(0)
        time.sleep(1.0)
        if (video.isOpened() == False): 
            print("Error opening camera")
            return
        out_dir = os.path.join('..', 'output')
        out_name = 'webcam_out' + '.avi'
    else:
        video = cv2.VideoCapture(video_path)
        if (video.isOpened() == False): 
            print("Error reading video file")
            return
        out_dir = os.path.join(os.path.dirname(video_path), '..', 'output')
        out_name = os.path.splitext(os.path.basename(video_path))[0] + '.avi'

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(os.path.join(out_dir, out_name), fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, image = video.read()
        if ret == False:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        h, w, _ = image.shape
        boxes, confidences, class_ids = perform_detection(net, image, output_layers, w, h, confidence_threshold)
        final_img = draw_boxes_labels(boxes, confidences, class_ids, classes, image, colors, confidence_threshold, nms_threshold, disp_scores)
        cv2.imshow("Detection", final_img)
        out.write(final_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()


def detection_image_file(image_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold, disp_scores):
    img, h, w = load_input_image(image_path)
    net, classes, output_layers = yolov3(yolo_weights, yolo_cfg, coco_names)
    # colors = np.random.uniform(0, 255, size=(len(classes), 3))
    colors = ((np.array(sns.color_palette("husl", len(classes))) * 255)).astype(int)
    boxes, confidences, class_ids = perform_detection(net, img, output_layers, w, h, confidence_threshold)

    final_img = draw_boxes_labels(boxes, confidences, class_ids, classes, img, colors, confidence_threshold, nms_threshold, disp_scores)
    
    cv2.imshow("Detection", final_img)
    final_img_dir = os.path.join(os.path.dirname(image_path), '..', 'output')
    if not os.path.exists(final_img_dir):
        os.makedirs(final_img_dir)
    cv2.imwrite(os.path.join(final_img_dir, os.path.basename(image_path)), final_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ## Arguments to give before running
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', help='Path to video file', default=None)
    ap.add_argument('--image', help='Path to the test images', default=None)
    ap.add_argument('-camera', help='To use the live feed from web-cam', action="store_true", default=False)
    ap.add_argument('--weights', help='Path to model weights', type=str, default='../pre-trained-model/yolov3.weights')
    ap.add_argument('--configs', help='Path to model configs',type=str, default='../pre-trained-model/yolov3.cfg')
    ap.add_argument('--class_names', help='Path to class-names text file', type=str, default='../pre-trained-model/coco.names')
    ap.add_argument('--conf_thresh', help='Confidence threshold value', default=0.6)
    ap.add_argument('--nms_thresh', help='NMS threshold value', default=0.4)
    ap.add_argument('-ds', help='To display probability scores for the object detected', action='store_true', default=False)
    args = vars(ap.parse_args())

    image_path = args['image']
    yolo_weights, yolo_cfg, coco_names = args['weights'], args['configs'], args['class_names']
    confidence_threshold = args['conf_thresh']
    nms_threshold = args['nms_thresh']

    options_list = [image_path is not None, args['camera'] == True, args['video'] is not None]

    if sum(options_list) == 0:
        print('No input provided to apply the model')

    elif sum(options_list) > 1:
        print('Two or more input sources provided. Give only one of image, video and webcam source')

    if image_path:
        detection_image_file(image_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold, args['ds'])

    elif args['camera'] == True or args['video']:
        webcam = args['camera']
        video_path = args['video']
        dectection_video_file(webcam, video_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold, args['ds'])


