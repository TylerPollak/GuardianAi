import requests
import imageio
import json
import time
import os
import io

with open("C:\\Users\\typollak\\Documents\\Hackathon2018-20180723T192115Z-001\\Hackathon2018\\facenet\\src\\JSONTEST.json", 'rb') as f:
    json_to_check = json.load(f)
    url = 'http://127.0.0.1:8080'
    r = requests.post(url, verify=False, json=json_to_check)
    print(r.status_code, r.json())