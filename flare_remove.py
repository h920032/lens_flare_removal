import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
from skimage.morphology import convex_hull_image
from flare_detect import *
from inpaint_model import InpaintCAModel
import tensorflow as tf
import neuralgym as ng
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='input image dictionary path', required=True)
parser.add_argument('--output', type=str, help='output image dictionary path', required=True)
args = parser.parse_args()

def is_flare_large(pixel):
    if pixel[0]>= 40 and pixel[0]<= 280 and pixel[1] >= 0.1 and pixel[2] >= 100:
        return True
    else:
        return False

def canny_convex(img1, flare_list):
    mask = np.zeros(img1.shape[0:2])
    for cir in flare_list:
        try:
            test = img1[round(cir[0]-3*cir[2]):round(cir[0]+3*cir[2]), round(cir[1]-3*cir[2]):round(cir[1]+3*cir[2])]
            small = cv2.Canny(test,100,300)
            if small is not None:
                chull = convex_hull_image(small)
                mask[round(cir[0]-3*cir[2]):round(cir[0]+3*cir[2]), round(cir[1]-3*cir[2]):round(cir[1]+3*cir[2])] = chull
        except:
            print(test)
    return np.where(mask==1,255,0)

def hsv(img1, flare_list):
    mask = np.zeros(img1.shape[0:2])
    for cir in flare_list:
        try:
            test = img1[round(cir[0]-3*cir[2]):round(cir[0]+3*cir[2]), round(cir[1]-3*cir[2]):round(cir[1]+3*cir[2])]
            m = np.zeros(test.shape[0:2])
            test_hsv = cv2.cvtColor(test.astype('float32'), cv2.COLOR_BGR2HSV)
            for i in range(test_hsv.shape[0]):
                for j in range(test_hsv.shape[1]):
                    if is_flare_large(test_hsv[i][j]):
                        m[i][j] = 255
            mask[round(cir[0]-3*cir[2]):round(cir[0]+3*cir[2]), round(cir[1]-3*cir[2]):round(cir[1]+3*cir[2])] = m
        except:
            print(test)
        
    kernel = np.ones((5, 5), 'uint8')
    mask = cv2.dilate(mask, kernel, iterations=3)
    return mask

def trand_inpainting(img,mask):
    result = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
    return result

def GAN_inpainting(img,mask_input):
    mask_output=np.zeros((mask_input.shape[0],mask_input.shape[1],3)).astype('uint8')
    mask_output[:,:,0] = mask_input
    mask_output[:,:,1] = mask_input
    mask_output[:,:,2] = mask_input
    
    FLAGS = ng.Config('./inpaint.yml')
    # ng.get_gpus(1)

    model = InpaintCAModel()
    image = cv2.resize(img, (img.shape[1]//2,img.shape[0]//2), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask_output, (mask_output.shape[1]//2,mask_output.shape[0]//2), interpolation = cv2.INTER_AREA)
    # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)

    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(FLAGS, input_image, reuse=tf.AUTO_REUSE)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable('./model_logs/release_places2_256', from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        return result[0][:, :, ::-1]

if __name__=="__main__":
    data = os.listdir(args.input)

    for d in data:
        img1 = cv2.imread(os.path.join(args.input,d))
        a = flare_list(img1)

        mask = hsv(img1, a).astype('uint8')

        #dst = trand_inpainting(img1,mask)
        dst = GAN_inpainting(img1,mask)
        print(dst)
        #output_img = img1.copy()
        #for i in a:
        #    output_img = cv2.circle(output_img, (int(i[1]),int(i[0])), round(i[2]), color=(255, 255, 0), thickness = 1)

        cv2.imwrite(os.path.join(args.output,'flare_'+d),dst)
