from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import numpy as np
import tensorflow as tf
import json
import sys
import pickle
from PIL import Image
import os
import compare
import copy
import argparse
import facenet
import align.detect_face
import io

# This script generates the scoring and schema files
# necessary to operationalize your model
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema

# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.


def init():
    global authorized_image_embeddings, pnet, rnet, onet, args, eval_graph

    model_location = './outputs/20180402-114759.pb'
    authorized_pictures = ['./outputs/IMG_9820.JPG',
                           './outputs/IMG_9822.JPG',
                           './outputs/ktXdtHO.JPG']

    argv = [model_location]
    for pic in authorized_pictures:
        argv.append(pic)
    args = compare.parse_arguments(argv)

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    authorized_images = load_and_align_data(args.image_files, args.image_size, args.margin)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(args.model)
            eval_graph = tf.get_default_graph()

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: authorized_images, phase_train_placeholder: False}
            authorized_image_embeddings = sess.run(embeddings, feed_dict=feed_dict)

def run(input_json):
    contains_match = False

    image_to_check = load_and_align_image_json(input_json, args.image_size, args.margin)

    if image_to_check is not None:
        image_to_check.resize((1, image_to_check.shape[0], image_to_check.shape[1], image_to_check.shape[2]))

        with tf.Session(graph=eval_graph) as sess:
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: image_to_check, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)

            args.image_files.append("input")
            nrof_images = len(args.image_files)

            embeddings_to_compare = np.append(authorized_image_embeddings, emb, axis=0)

            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, args.image_files[i]))
            print('')

            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(embeddings_to_compare[i, :], embeddings_to_compare[j, :]))))
                    if dist < .3:
                        contains_match = True
                    print('  %1.4f  ' % dist, end='')
                print('')

    prediction = str(contains_match)
    return json.dumps(str(prediction))


def load_and_align_data(image_paths, image_size, margin):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    tmp_image_paths = copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, remove ", image)
            continue
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    if len(img_list) > 0:
        return img_list
    return None

def load_and_align_image_json(image_json, image_size, margin):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    data = json.loads(image_json)
    image_list = data['image']
    image = np.array(image_list, dtype='uint8')
    img_size = np.asarray(image.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1:
        print("can't detect face, skipping input")
        return None
    det = np.squeeze(bounding_boxes[0, 0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - margin / 2, 0)
    bb[1] = np.maximum(det[1] - margin / 2, 0)
    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
    cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    prewhitened = facenet.prewhiten(aligned)
    return prewhitened


# Implement test code to run in IDE or Azure ML Workbench
if __name__ == '__main__':
    import time

    init()

    string_io = io.StringIO()
    image_to_check = misc.imread(os.path.expanduser('./outputs/IMG_9823.JPG'), mode='RGB')
    json.dump({'image': image_to_check.tolist()}, string_io)
    json_to_check = string_io.getvalue()
    start = time.time()
    run(json_to_check)
    end = time.time()
    print("elapsed time", end - start)
