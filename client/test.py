#!/usr/bin/python3

import cv2 
import time
import requests
import serial
import base64
from pylab import *
from numpy import *
from scipy.ndimage import filters
from PIL import Image
import json

cam = cv2.VideoCapture("/home/qzwlecr/1.avi")
idx = 0
times = 0
last = -1
'''ser = serial.Serial()
ser.port = '/dev/ttyS3'
ser.bandrate = 9600
ser.timeout = 1
ser.open()'''
def answer(vec_ret ):
    for i in vec_ret:
        if is_closing(i) :
            if last == -1:
                last = post(i)
                return post(i), post(i)
            else:
                last = post(i)
                return last, post(i)
    last = -1
    return -1,-1


def post(a):
    if a[1] < 1280/3:
        return 0
    if a[3] > (2/3 *1280):
        return 2
    if (a[1] > 1280/3 and a[3] < 2/3*1280):
        return 1

def is_closing(a):
    if (a[3]-a[1])*(a[2]-a[0]) > 128*720:
        return True
    else:
        return False

while True:
    vec_ret = []
    vec_ret_last = []
    ret, frame = cam.read()
    last = -1
    #print(ret)
    if idx % 48 == 0:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st = base64.b64encode(img)
        r = requests.post('http://139.219.185.108:5000/',data = st
        print("found positions:")
        if times == 0:
            vec_ret_last = json.loads(str(r.content, 'utf-8'))
            vec_ret = json.loads(str(r.content,'utf-8'))
        else:
            vec_ret_last = vec_ret
            vec_ret = json.loads(str(r.content,'utf-8'))
        print(vec_ret)
        ret1,ret2= answer(vec_ret)
        if ret1 != -1:
            continue
            #ser.write(('0'+ret1).encode())
            #ser.write(('0'+ret2).encode())
    idx = idx+1
cam.release()



