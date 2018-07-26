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


class Model:
    def __init__(self, graph, embs, arguments):
        self.tf_graph = graph
        self.img_embs = embs
        self.args = arguments


def init():
    global authorized_image_embeddings, pnet, rnet, onet, args, eval_graph, model

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model.tf_graph)
        tf.import_graph_def(graph_def, input_map=None, name='')
        eval_graph = tf.get_default_graph()
        authorized_image_embeddings = model.img_embs
        args = model.args

        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)


def eval(input_json):
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


def load_and_align_image_json(image_json, image_size, margin):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    data = json.loads(image_json)
    image_list = data['input_json']
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

if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()