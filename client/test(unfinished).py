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
'''
ser = serial.Serial()
ser.port = '/dev/ttyS3'
ser.bandrate = 9600
ser.timeout = 1
ser.open()
'''
def Crops(image_now,image_last,vec_ret,vec_ret_last):
    now_ = []
    image_a = Image.fromarray(image_now)
    image_b = Image.fromarray(image_last)
    for a in vec_ret:
        now_.append(image_a.crop(a))
    last_ = []
    for b in vec_ret_last:
        last_.append(image_b.crop(b))
    cal_match =[]
    for i in now_:
        for j in last_:
            cal_match.append(match_(i,j))
    cal_match = cal_match.reshape(len(j),len(i))
    match_ret = []
    for i in cal_match:
        match_ret.append(i.index(max(i)))
    return match_ret

def match_(a,b):
    im1 = Image.fromarray(a)
    im2 = Image.fromarray(b)
    # resize to make matching faster
    #im1 = imresize(im1, (int(im1.shape[1]/2), int(im1.shape[0]/2)))
    #im2 = imresize(im2, (int(im2.shape[1]/2), int(im2.shape[0]/2)))

    wid = 5
    harrisim = compute_harris_response(im1, 5)
    filtered_coords1 = get_harris_points(harrisim, wid+1)
    d1 = get_descriptors(im1, filtered_coords1, wid)

    harrisim = compute_harris_response(im2, 5)
    filtered_coords2 = get_harris_points(harrisim, wid+1)
    d2 = get_descriptors(im2, filtered_coords2, wid)

    matches = match_twosided(d1, d2)
    return  len(matches[matches>0])

def post(a):
    if a[1] < 1280/3:
        return 0
    if a[3] > (2/3 *1280):
        return 2
    if (a[1] > 1280/3 and a[3] < 2/3*1280):
        return 1

def is_closing(a):
    if now_[a].size > 128*720:
        return True
    else:
        return False


def answer(match_ret):
    for i in match_ret:
        j = i.index()
        mins = post(vec_ret[i])-post(vec_ret_last[j])
        if mins == 0 :
            if is_closing(i) :
                return post(vec_ret[i]), post(vec_ret[i])
        else:
            if mins == -1:
                if is_closing(i):
                    return post(vec_ret[i]),post(vec_ret_last[j])
            else:
                if is_closing(i):
                    return post(vec_ret[i]),post(vec_ret_last[j])
    for i in now_:
        if is_closing(i):
            return post(vec_ret[i]),post(vec_ret[i])

def compute_harris_response(im,sigma=3):
    """ Compute the Harris corner detector response function
        for each pixel in a graylevel image. """

    # derivatives
    imx = zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)

    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx,sigma)
    Wxy = filters.gaussian_filter(imx*imy,sigma)
    Wyy = filters.gaussian_filter(imy*imy,sigma)

    # determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet / Wtr


def get_harris_points(harrisim,min_dist=10,threshold=0.1):
    """ Return corners from a Harris response image
        min_dist is the minimum number of pixels separating
        corners and image boundary. """

    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    # get coordinates of candidates
    coords = array(harrisim_t.nonzero()).T

    # ...and their values
    candidate_values = [harrisim[c[0],c[1]] for c in coords]

    # sort candidates (reverse to get descending order)
    index = argsort(candidate_values)[::-1]

    # store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1

    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
                        (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0

    return filtered_coords


def plot_harris_points(image,filtered_coords):
    """ Plots corners found in image. """

    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords],
                [p[0] for p in filtered_coords],'*')
    axis('off')
    show()


def get_descriptors(image,filtered_coords,wid=5):
    """ For each point return pixel values around the point
        using a neighbourhood of width 2*wid+1. (Assume points are
        extracted with min_distance > wid). """

    desc = []
    for coords in filtered_coords:
        patch = image[coords[0]-wid:coords[0]+wid+1,
                            coords[1]-wid:coords[1]+wid+1].flatten()
        desc.append(patch)

    return desc


def match(desc1,desc2,threshold=0.5):
    """ For each corner point descriptor in the first image,
        select its match to second image using
        normalized cross correlation. """

    n = len(desc1[0])

    # pair-wise distances
    d = -ones((len(desc1),len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
            d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
            ncc_value = sum(d1 * d2) / (n-1)
            if ncc_value > threshold:
                d[i,j] = ncc_value

    ndx = argsort(-d)
    matchscores = ndx[:,0]

    return matchscores


def match_twosided(desc1,desc2,threshold=0.5):
    """ Two-sided symmetric version of match(). """

    matches_12 = match(desc1,desc2,threshold)
    matches_21 = match(desc2,desc1,threshold)

    ndx_12 = where(matches_12 >= 0)[0]

    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1

    return matches_12


def appendimages(im1,im2):
    """ Return a new image that appends the two images side-by-side. """

    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0)
    # if none of these cases they are equal, no filling needed.

    return concatenate((im1,im2), axis=1)


def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
    """ Show a figure with lines joining the accepted matches
        input: im1,im2 (images as arrays), locs1,locs2 (feature locations),
        matchscores (as output from 'match()'),
        show_below (if images should be shown below matches). """

    im3 = appendimages(im1,im2)
    if show_below:
        im3 = vstack((im3,im3))

    imshow(im3)

    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
    axis('off')

def imresize(im,sz):
    """    Resize an image array using PIL. """
    pil_im = Image.fromarray(uint8(im))

    return array(pil_im.resize(sz))
"""
Example of detecting Harris corner points (Figure 2-1 in the book).
"""

while True:
    vec_ret = []
    vec_ret_last = []
    ret, frame = cam.read()
    nowImage = frame
    lastImage = frame
    print(ret)
    if idx % 48 == 0:
        lastImage = nowImage
        nowImage = frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st = base64.b64encode(img)
        # print(img.shape)
        r = requests.post('http://139.219.185.108:5000/',data = st)
        if times == 0:
            vec_ret_last = json.loads(str(r.content, 'utf-8'))
            vec_ret = json.loads(str(r.content,'utf-8'))
        else:
            vec_ret_last = vec_ret
            vec_ret = json.loads(str(r.content,'utf-8'))
        # print(r.content)
        match_ret=Crops(nowImage,lastImage,vec_ret,vec_ret_last)
        ret1,ret2= answer(match_ret)
        #ser.write(('0'+ret1).encode())
        #ser.write(('0' + ret2).encode())
    idx = idx+1
cam.release()




