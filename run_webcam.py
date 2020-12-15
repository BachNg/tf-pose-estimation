import argparse
import logging
import time
from sort import *
import cv2
import numpy as np
from skeleton import *

import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt 
from scipy.spatial.distance import cdist

from keras.optimizers import *
from keras.models import Model
from keras.layers import *
from keras.layers.core import *
from tensorflow.keras.callbacks import *
from keras.layers.convolutional import *
from keras import backend as K
import keras
import tensorflow as tf

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
labels = []
with open('./labels.txt', 'r') as f:
    labels = [str(i).strip() for i in f]

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    mot_tracker = Sort(max_age= 5, 
                       min_hits=3,
                       iou_threshold= 0.3)
    C = Config()
    predict = []
    DD_Net = build_DD_Net(C)
    DD_Net.load_weights("./checkpoints/checkpoint_0.872.hdf5")

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))
    out = cv2.VideoWriter('./video_test.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5.0, (frame_width,frame_height))
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    order = [i for i in range(14)]
    human_actions = {}
    while True:
        ret_val, image = cam.read()
        #print(np.shape(image))
        #logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        dets = []
        new_hum = []
        for people in humans:

            box = people.get_upper_body_box(np.shape(image)[1], np.shape(image)[0])
            points = [i for i in people.body_parts]
            if all(i in points for i in order) and box is not None:
                dets.append([box["x"], box["y"], box["x"]+box["w"], box["y"]+ box["h"]])
                new_hum.append(people)
            else:
                continue

        if len(dets) > 0:
            trackers = mot_tracker.update(np.array(dets))
            for idx, d in enumerate(trackers):
                #print(d[4], ":", new_hum[idx].body_parts)
                joints = [[new_hum[idx].body_parts[i].x, new_hum[idx].body_parts[i].y] for i in order]
                if d[4] not in human_actions.keys():
                    human_actions[d[4]] = {}
                    human_actions[d[4]]['joints'] = []
                    human_actions[d[4]]['joints'].append(joints)
                else:
                    human_actions[d[4]]['joints'].append(joints)
                human_actions[d[4]]['det'] = d[:4]
                cv2.rectangle(image, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (255, 0, 0), 2)
                cv2.putText(image, "ID " + str(int(d[4])), (int(d[0]), int(d[1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        before = len(predict)
        for key in human_actions.keys():
            #print("dddddddddddddd", np.shape())
            if len(human_actions[key]['joints']) == 32:
                start = time.time()
                Test = np.expand_dims(human_actions[key]['joints'], axis=0)
                X_test_0,X_test_1 = data_generator(Test,C)
                prediction = DD_Net.predict([X_test_0, X_test_1])
                #print(np.argsort(prediction), prediction[0][np.argsort(prediction)[0][-1]], prediction[0][np.argsort(prediction)[0][-2]])
                predict.append(labels[int(np.argmax(prediction[0]))])
                print("Skeleton model time: ", time.time() - start)
                human_actions[key]['joints'] = human_actions[key]['joints'][16:]
        after = len(predict)
        space = 10
        for k in predict:
            cv2.putText(image, k, (10, 20+space),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            space = space + 20
        #logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        #logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        out.write(image)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        #logger.debug('finished+')
    out.release()
    cv2.destroyAllWindows()
