import cv2
import base64
import requests
import json
import numpy as np
import time
from tolist import tolist

address = 'http://postb.in'
test_url = address + '/HRgojc0X'

content_type = 'application/json'
headers = {'content-type': content_type}

jsonList = []

cap = cv2.VideoCapture(0)

if (cap.isOpened()== False): 
  print("Error opening video stream or file")


# Capture Frame
while(cap.isOpened()):
 
  ret, frame = cap.read()
  if ret == True:
    cv2.imshow('frame',frame)

    # Split frame into blue, green, red lists
    #b, g , r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    
    # Serialize all lists into single json list
    #for i in range(0,len(b)):
        #jsonList.append({"blue" : b[i], "green" : g[i], "red" : r[i]})
        #jsonList = json.dumps(jsonList, indent=1)

    # Serialize
    jsonList = frame.tolist()
    jsonList = json.dumps(jsonList)
        
    # Send HTTP request
    response = requests.post(test_url, data=jsonList, headers=headers)
    
    # Get HTTP response from server (see webserver_intake.py)
    print(response.status_code)
    print(response.content)

    # Sleep for 1s
    time.sleep(1)

    # Press Q on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
  else: 
    break

cap.release()
out.release()
cv2.destroyAllWindows() 