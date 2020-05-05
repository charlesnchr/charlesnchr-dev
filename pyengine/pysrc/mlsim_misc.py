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
from mlsim_lib import *
import datetime
import tifffile as tiff
# import sklearn.neighbors
import os
import hashlib

# import dbm


ImageFile.LOAD_TRUNCATED_IMAGES = True

useCloud = False
libraryFolders = []
libraryFiles = []
libraryFiles_hash = ""
features = []
files = []
knn = []
model = None
opt = None
device = []
cachefolder = None

inprogress_calcFeatures = False
quit_calcFeatures = False


class VGGFeatureExtractor(nn.Module):
    def __init__(
        self,
        feature_layer=34,
        use_bn=False,
        use_input_norm=True,
        device=torch.device("cpu"),
    ):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        model = vgg19()
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(
                1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(
                1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        self.features = nn.Sequential(
            *list(model.features.children())[: (feature_layer + 1)]
        )
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


def md5file(filename):
    with open(filename, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()


# def GetModel():
#     model = VGGFeatureExtractor(feature_layer=37, use_bn=False, use_input_norm=False)
#     if torch.cuda.is_available():
#         log("using gpu")
#         model.cuda()
#         device = torch.device("cuda")
#     else:
#         log("using cpu")
#         device = torch.device("cpu")

#     return model, device


def GetLibraryFiles():
    global libraryFiles
    return libraryFiles


def GetCachefolder():
    global cachefolder
    return cachefolder


def inprogress_calcFeatures_get():
    global inprogress_calcFeatures
    return inprogress_calcFeatures


def quit_calcFeatures_set(val):
    global quit_calcFeatures
    quit_calcFeatures = val


def SetCachefolder(value):
    global cachefolder
    cachefolder = value


def SetUseCloud(val):
    global useCloud
    if int(val) == 1:
        useCloud = True
    else:
        useCloud = False
    log("useCloud is now %d " % useCloud)


def CalcImageFeatures(I):
    global model
    global device

    # I = io.imread(imgfile) / 255
    # I =ImageOps.fit(I,(300,300), method=Image.BICUBIC)
    Iscaled = np.array(I) / 255
    # I = transform.resize(I,(224,224,3))
    tns = torch.tensor(Iscaled).permute(2, 0, 1)
    t = model(tns.unsqueeze(0).float().to(device)).cpu()
    imgfeatures = t.reshape(-1,).numpy().astype("float32")
    return imgfeatures


def CalcImageFeatures_cloud(I):
    # I =ImageOps.fit(I,(300,300), method=Image.BICUBIC)
    img_io = BytesIO()
    I.save(img_io, "jpeg", quality=90)

    try:
        t = requests.post(
            "https://imageheal.com/api_vggImgFeatures",
            files={"image": ("image.png", img_io.getvalue())},
        )
        imgfeatures = np.frombuffer(t.content, "float32")
        return imgfeatures
    except:
        errmsg = traceback.format_exc()
        global useCloud
        useCloud = False
        log("cloud compute failed, using local")
        return CalcImageFeatures(I)


def GetImageFeatures(I):
    global useCloud

    if useCloud:
        imgfeatures = CalcImageFeatures_cloud(I)
    else:
        imgfeatures = CalcImageFeatures(I)
    # db[md5key] = imgfeatures.tobytes()

    return imgfeatures


def LoadOrCalcImageFeatures(I, md5key):
    global cachefolder

    if not os.path.isfile("%s/%s.npz" % (cachefolder, md5key)):
        imgfeatures = GetImageFeatures(I)
    else:
        imgfeatures = np.load("%s/%s.npz" % (cachefolder, md5key))["arr_0"]

    return imgfeatures


def LoadOrReadAndCalcImageFeatures(imgfile, md5key):
    global cachefolder

    if not os.path.isfile("%s/%s.npz" % (cachefolder, md5key)):
        I = ReadImage(imgfile)
        if not I is None:
            imgfeatures = GetImageFeatures(I)
        else:
            imgfeatures = None
    else:
        imgfeatures = np.load("%s/%s.npz" % (cachefolder, md5key))["arr_0"]

    return imgfeatures


def ReadImage(imgfile):
    try:
        I = Image.open(imgfile).convert("RGB")
        I = I.resize((300, 300), resample=Image.BICUBIC)
        return I
    except:
        errmsg = traceback.format_exc()
        log("Error reading %s - error: %s" % (imgfile, errmsg))
        return None


def GetSimilarImages(imgfile, conn):
    global features
    global libraryFolders
    global libraryFiles
    global knn

    if len(features) == 0 or len(libraryFiles) == 0 or isinstance(knn, list):
        log("no features/files available, aborting")
        return []

    try:
        md5key = md5file(imgfile)
    except:
        errmsg = traceback.format_exc()
        log(imgfile)
        return []

    imgfeatures = LoadOrReadAndCalcImageFeatures(imgfile, md5key)
    if imgfeatures is None:
        log("imgfeatures failed to compute, aborting")
        return []

    dists, indices = knn.kneighbors([imgfeatures])

    result_list = []

    for idx, dist in zip(indices[0], dists[0]):
        if dist < 0.9:  # could break earlier for more performance
            result_list.append(libraryFiles[idx])

    return result_list


def GetImagesFromText(term, conn):
    global features
    global libraryFiles
    global libraryFolders
    global knn

    if len(features) == 0 or len(libraryFiles) == 0 or isinstance(knn, list):
        log("no features/files available, aborting")
        return []

    # imgfile = term + '.jpg'
    # imgfeatures = GetImageFeatures(imgfile)
    try:
        t = requests.post(
            "https://imageheal.com/api_vggImgFeaturesSearchTerm",
            data={"search_term": term},
        )
        imgfeatures = np.frombuffer(t.content, "float32")
    except:
        errmsg = traceback.format_exc()
        return []

    dists, indices = knn.kneighbors([imgfeatures])

    result_list = []

    for idx, dist in zip(indices[0], dists[0]):
        if dist < 0.9:  # could break earlier for more performance
            result_list.append(libraryFiles[idx])

    return result_list


def compareFilepathHash(input_hash):
    global libraryFiles_hash
    log("Comparing hashes %s and %s" % (input_hash, libraryFiles_hash))
    return input_hash == libraryFiles_hash


def reconstruct(exportdir,filepaths, conn):
    global model
    global opt

    if model is None:
        opt = GetOptions_allRnd()
        model = LoadModel(opt)

    os.makedirs(exportdir,exist_ok=True)
    result_arr = []

    print('received filepaths', filepaths)

    for idx, filepath in enumerate(filepaths):
        print('ITRERASTION',idx,filepath)
        # status reconstruction
        conn.send(("si%d,%d" % (idx, len(filepaths))).encode())
        ready = conn.recv(20).decode()

        outfile = '%s/%s' % (exportdir,
                             datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3])
        img = tiff.imread(filepath, key=range(9))
        sr, wf, out = EvaluateModel(model, opt, img, outfile)
        result_arr.append(sr)

    conn.send("sd".encode())  # status done
    ready = conn.recv(20).decode()

    return result_arr


def calcFeatures(filepaths, filepaths_hash, conn):
    global inprogress_calcFeatures
    global quit_calcFeatures
    global libraryFiles

    if len(filepaths) < 20:
        # too few images for knn model
        conn.send("sd".encode())  # status done
        ready = conn.recv(20).decode()

    libraryFiles = filepaths

    if inprogress_calcFeatures:
        quit_calcFeatures = True
    else:
        try:
            _calcFeatures(filepaths_hash, conn)
            while quit_calcFeatures:
                _calcFeatures(filepaths_hash, conn)
            quit_calcFeatures = False
        except:
            errmsg = traceback.format_exc()
            inprogress_calcFeatures = False
            quit_calcFeatures = False

    conn.send("sd".encode())  # status done
    ready = conn.recv(20).decode()


def _calcFeatures(filepaths_hash, conn):
    global features
    global libraryFolders
    global libraryFiles
    global libraryFiles_hash
    global cachefolder
    global knn
    global model
    global device
    global inprogress_calcFeatures
    global quit_calcFeatures

    knn = []

    quit_calcFeatures = False
    inprogress_calcFeatures = True
    features = []
    os.makedirs(cachefolder, exist_ok=True)

    # if model == [] or device == []:
    #     conn.send("sm".encode())  # status loading model
    #     ready = conn.recv(20).decode()
    # model, device = GetModel()

    nimg = len(libraryFiles)
    files_to_index = []
    files_to_index_keys = []
    files_to_load = []
    files_to_load_keys = []

    # * Hashing loop (scans all files)
    for idx, file in enumerate(libraryFiles):
        if quit_calcFeatures:
            log("quiting early, %d" % nimg)
            inprogress_calcFeatures = False
            return

        try:
            md5key = md5file(file)
        except:
            errmsg = traceback.format_exc()
            log(file)
            continue

        if not os.path.isfile("%s/%s.npz" % (cachefolder, md5key)):
            files_to_index.append(file)
            files_to_index_keys.append(md5key)
        else:
            files_to_load.append(file)
            files_to_load_keys.append(md5key)

        percent = float(idx + 1) / float(nimg)
        percent_former = float(idx) / float(nimg)
        if nimg > 300 and int(percent * 100) > int(percent_former * 100):
            conn.send(("sh%d" % int(percent * 100)).encode())  # status hashing
            ready = conn.recv(20).decode()
        elif nimg > 150 and int(percent * 20) > int(percent_former * 20):
            conn.send(("sh%d" % int(percent * 100)).encode())  # status hashing
            ready = conn.recv(20).decode()
        elif int(percent * 10) > int(percent_former * 10):
            conn.send(("sh%d" % int(percent * 100)).encode())  # status hashing
            ready = conn.recv(20).decode()

    libraryFiles = []  # will only add valid images

    # * Loading loop
    for idx, (imgfile, md5key) in enumerate(zip(files_to_load, files_to_load_keys)):
        if quit_calcFeatures:
            log("quiting early %d" % nimg)
            inprogress_calcFeatures = False
            return

        try:
            imgfeatures = np.load("%s/%s.npz" % (cachefolder, md5key))["arr_0"]
            if len(imgfeatures) == 0:
                files_to_index.append(imgfile)
                files_to_index_keys.append(md5key)
                continue
        except:
            errmsg = traceback.format_exc()
            log("Error loading image features %s, error: %s" % (imgfile, errmsg))
            files_to_index.append(imgfile)
            files_to_index_keys.append(md5key)
            continue

        percent = float(idx + 1) / float(len(files_to_load))
        percent_former = float(idx) / float(len(files_to_load))
        if len(files_to_load) > 3000 and int(percent * 100) > int(percent_former * 100):
            conn.send(("sl%d" % int(percent * 100)).encode())  # status hashing
            ready = conn.recv(20).decode()
        elif len(files_to_load) > 500 and int(percent * 20) > int(percent_former * 20):
            conn.send(("sl%d" % int(percent * 100)).encode())  # status hashing
            ready = conn.recv(20).decode()
        elif int(percent * 5) > int(percent_former * 5):
            conn.send(("sl%d" % int(percent * 100)).encode())  # status hashing
            ready = conn.recv(20).decode()

        features.append(imgfeatures)
        libraryFiles.append(imgfile)

    # * Indexing loop
    import time
    for idx, (imgfile, md5key) in enumerate(zip(files_to_index, files_to_index_keys)):
        if quit_calcFeatures:
            log("quiting early %d" % nimg)
            inprogress_calcFeatures = False
            return

        # I = ReadImage(imgfile)
        # if not I is None:
        #     imgfeatures = GetImageFeatures(I)
        # else:
        #     continue

        # np.savez_compressed("%s/%s.npz" % (cachefolder, md5key), imgfeatures)

        time.sleep(0.5)

        # status indeixing
        conn.send(("si%d,%d" % (idx, len(files_to_index))).encode())
        ready = conn.recv(20).decode()

        # features.append(imgfeatures)
        libraryFiles.append(imgfile)

    # * Fitting
    if len(features) >= 20:
        conn.send("sc".encode())  # status computing search tree
        ready = conn.recv(20).decode()
        features = np.array(features)
        knn = NearestNeighbors(n_neighbors=20, metric="cosine")
        knn.fit(features)
        conn.send("sd".encode())  # status done
        ready = conn.recv(20).decode()
    else:
        log("too few images have been index to build search tree")
        conn.send("sd".encode())  # status done
        ready = conn.recv(20).decode()

    log("finished calculating features")
    libraryFiles_hash = filepaths_hash
    inprogress_calcFeatures = False
