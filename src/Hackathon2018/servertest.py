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

with open('jsontest.pkl', 'rb') as f:
    data = pickle.load(f)
    url = 'http://127.0.0.1:8080'
    r = requests.post(url, verify=False, json=data)
    print(r.status_code, r.json())
