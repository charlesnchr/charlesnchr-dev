import torch
import torch.nn as nn
from vgg import vgg19
import glob
from PIL import Image, ImageFile  # ImageOps
from io import BytesIO
import requests

import random
import numpy as np
import sklearn
import sklearn.utils
from sklearn.neighbors import NearestNeighbors

import traceback

from misc import log
import mlsim_lib
import datetime
import tifffile as tiff
import os
import hashlib
import time
import socket

ImageFile.LOAD_TRUNCATED_IMAGES = True

useCloud = False
features = []
files = []
model = None
opt = None
device = []
cachefolder = None

server_socket = None
microManagerPluginState = False

def GetCachefolder():
    global cachefolder
    return cachefolder

def SetCachefolder(value):
    global cachefolder
    cachefolder = value

def handle_microManagerPluginState(desiredState, port):
    global microManagerPluginState
    global server_socket

    if desiredState == 'on' and microManagerPluginState == False:
        th = threading.Thread(target=start_plugin_server, args=(port,))
        th.daemon = True
        th.start()
    elif desiredState == 'off' and microManagerPluginState == True:
        server_socket.close()
        microManagerPluginState = False


def set_microManagerPluginState(value):
    global microManagerPluginState
    microManagerPluginState = value



def SetUseCloud(val):
    global useCloud
    if int(val) == 1:
        useCloud = True
    else:
        useCloud = False
    log("useCloud is now %d " % useCloud)


def reconstruct(exportdir,filepaths, conn):
    global model
    global opt

    if model is None:
        opt = mlsim_lib.GetOptions_allRnd()
        model = mlsim_lib.LoadModel(opt)

    os.makedirs(exportdir,exist_ok=True)
    result_arr = []

    log('received filepaths %s' % filepaths)

    for idx, filepath in enumerate(filepaths):
        log('ITRERASTION %d %s' % (idx,filepath))
        # status reconstruction
        conn.send(("si%d,%d" % (idx, len(filepaths))).encode())
        ready = conn.recv(20).decode()

        outfile = '%s/%s' % (exportdir,
                             datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3])
        img = tiff.imread(filepath, key=range(9))
        sr, wf, out = mlsim_lib.EvaluateModel(model, opt, img, outfile)
        result_arr.append(sr)
    
    conn.send("sd".encode())  # status done
    ready = conn.recv(20).decode()

    return result_arr



## --------------------------------------------------------------------------------
##  For Micromanager plugin
## --------------------------------------------------------------------------------

import asyncio
import copy
count = -1
asyncmodels = []
num_models = 5
import threading

class AsyncModel:
    def __init__(self, opt):
        self.resultReady = True
        self.model = mlsim_lib.LoadModel(opt)
        self.result = np.zeros((480,512))
        self.dependency = None

    def getResult(self, opt, img):
        th = threading.Thread(target=mlsim_lib.EvaluateModelRealtimeAsync, args=(self, opt, img))
        th.daemon = True
        th.start()
        while not self.dependency.resultReady:
            time.sleep(0.01)
        self.dependency.resultReady = False
        return self.dependency.result


def reconstruct_image(img):
    global asyncmodels
    global opt
    global count

    if len(asyncmodels) == 0:
        opt = mlsim_lib.GetOptions_allRnd()

        for i in range(num_models):
            asyncmodels.append(AsyncModel(opt))
        for i in range(num_models):
            asyncmodels[i].dependency = asyncmodels[(i+1) % num_models]
    
    count += 1

    return asyncmodels[count % num_models].getResult(opt, img)




import numpy as np
import cv2
from PIL import Image
import pickle
import matplotlib.pyplot as plt

def receiveImage(conn,npixels,w,h,debugMode=False):
    
    bytesstr = None
    count = 0

    while True:
        conn.send('i'.encode())  # send image data
        data = conn.recv(2000)
        if bytesstr is None:
            bytesstr = data
        else:
            bytesstr += data

        count += 1
        # print('%d batch, added now %d' % (count,len(bytesstr)))
        if len(bytesstr) >= npixels:
            conn.send('c'.encode())

            if debugMode:
                print('received all bytes')
                pickle.dump(bytesstr,open('data.pkl','wb'))
                print('saved as npy')
                return None
            else:
                img = Image.frombuffer("RGBA",(w,h),bytesstr,"raw","BGRA").convert("RGB")
                return np.array(img)
            ## img = filters.sobel(np.array(img).mean(2))
            

 

def start_plugin_server(port):
    global server_socket
    global microManagerPluginState

    showLiveView = True
    debugMode = True
    host = 'localhost'
    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((host, port))  # bind host address and port together
    microManagerPluginState = True

    # configure how many client the server can listen simultaneously
    server_socket.listen(1)
    print('ML-SIM Micromanager now listening for connection')
    try:
        conn, address = server_socket.accept()  # accept new connection
    except:
        conn = None
        errmsg = traceback.format_exc()
        if "not a socket" in errmsg:
            log('Socket closed forcefully')
        else:
            log(errmsg)
        

    if microManagerPluginState and conn is not None:

        stackbuffer = []

        try:
            log('connection from %s' % str(address))

            # receive data stream. it won't accept data packet greater than 2048 bytes  
            count = 0
            t0 = time.perf_counter()
            fps = np.ones((10,))

            if showLiveView and not debugMode:
                plt.figure(figsize=(9,6),frameon=False)
                ax = plt.subplot(111,aspect = 'equal')
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

            while microManagerPluginState:
                data = conn.recv(2048).decode()
                vals = data.split(",")
                npixels = int(vals[0])
                w = int(vals[1])
                h = int(vals[2])
                bytesPerPixel = int(vals[3])
                numComponents = int(vals[4])
                # npixels = int.from_bytes(data,"big",signed=True)
                # print("Received",data)
                # print('received npixels',npixels)
                if npixels > 0:
                    img = receiveImage(conn,npixels,w,h,debugMode)
                    
                    if not debugMode:
                        stack = np.zeros((9,h,w))
                        stack[0,:,:] = img[:,:,0]
                        stack[1,:,:] = img[:,:,1]
                        stack[2,:,:] = img[:,:,2]
                        stack[3,:,:] = img[:,:,0]
                        stack[4,:,:] = img[:,:,1]
                        stack[5,:,:] = img[:,:,2]
                        stack[6,:,:] = img[:,:,0]
                        stack[7,:,:] = img[:,:,1]
                        stack[8,:,:] = img[:,:,2]
                        sr = reconstruct_image(stack)
                        print('here with',sr.shape)

                    if showLiveView and not debugMode:
                        plt.cla()
                        plt.gca().imshow(sr,cmap='magma')
                        plt.pause(0.01)
                    
                    # print('received img',img.size)
                    fps[count % 10] = 1 / (time.perf_counter() - t0)
                    count += 1
                    t0 = time.perf_counter()
                    print('img #%d (%dx%d) (%d,%d) - fps: %0.3f' % (count,w,h,bytesPerPixel,numComponents,fps.mean()))
                    continue
                else:
                    print('received',data,'exiting')
                    break
            
        except Exception as e:
            errmsg = traceback.format_exc()
            log(errmsg)
        
        microManagerPluginState = False
            # send_log(errmsg)
