"""
Very simple HTTP server in python for logging requests
Usage::
    ./server.py [<port>]
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import requests
from scipy import misc
import numpy as np
import tensorflow as tf
import json
import time
import imageio
from tensorflow.python.platform import gfile
import sys
import pickle
import os
import compare
import copy
import facenet
import align.detect_face
import io


class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/json')
        self.end_headers()

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        my_json = post_data.decode('utf8')
        data = json.loads(my_json)
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                str(self.path), str(self.headers), post_data.decode('utf-8'))
        result = eval(data)
        self._set_response()
        self.wfile.write(result.encode())


def run(server_class=HTTPServer, handler_class=S, port=8080):
    init()
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')


def init():
    global authorized_image_embeddings, pnet, rnet, onet, args, eval_graph

    model_location = "C:\\Users\\typollak\\Documents\\Hackathon2018-20180723T192115Z-001\\Hackathon2018\\facenet\\src\\VGGFace2\\20180402-114759.pb"
    authorized_pictures = ['C:\\Users\\typollak\\Documents\\Hackathon2018-20180723T192115Z-001\\Hackathon2018\\facenet\\src\\pics\\IMG_9822.JPG']

    argv = [model_location]
    for pic in authorized_pictures:
        argv.append(pic)
    args = compare.parse_arguments(argv)

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.3)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
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
    args.image_files.append("input")


def eval(input_json):
    contains_match = False

    start = time.time()
    image_to_check = load_and_align_image_json(input_json, args.image_size, args.margin)
    end = time.time()
    print("elapsed time", end - start)

    if image_to_check is not None:
        image_to_check.resize((1, image_to_check.shape[0], image_to_check.shape[1], image_to_check.shape[2]))

        start = time.time()
        with tf.Session(graph=eval_graph) as sess:
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: image_to_check, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)

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
                    if dist < 1 and dist != 0:
                        contains_match = True
                    print('  %1.4f  ' % dist, end='')
                print('')
        end = time.time()
        print("elapsed time", end - start)

    prediction = str(contains_match)
    return json.dumps(str(prediction))


def load_and_align_data(image_paths, image_size, margin):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    tmp_image_paths = copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = imageio.imread(image)
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

    start = time.time()
    data = json.loads(image_json)
    image_list = data
    image = np.array(image_list, dtype='uint8')
    end = time.time()
    print("elapsed time", end - start)
    img_size = np.asarray(image.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1:
        print("can't detect face, remove ", image)
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

if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()