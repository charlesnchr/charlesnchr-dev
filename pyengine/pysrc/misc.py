import requests
import datetime
import traceback


print('Initting misc')
fid = open('socketserver.log', 'a+')


def _get_fid():
    global fid
    return fid

def _set_fid(newfid):
    global fid
    fid = newfid

def log(text):
    print('%s, %s' % (datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S"), text))
    fid.write('\n%s, %s' %
              (datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S"), text))
    fid.flush()


import tifffile as tiff
from skimage import io, transform, exposure
import hashlib

def md5file(filename):
    with open(filename, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()

def GetThumb(args):
    filepath = args[0]
    try:
        md5val = md5file(filepath)
        cachefolder = mlsim.GetCachefolder()
        thumbpath = '%s/%s.jpg' % (cachefolder, md5val)

        # create thumb
        if not os.path.exists(thumbpath):
            img = tiff.imread(filepath, key=0)
            # img = cv2.resize(img, (512, 512))
            img = transform.resize(img,(512,512))
            p2 = np.percentile(img, 0.5)
            p98 = np.percentile(img, 99.5)
            img = exposure.rescale_intensity(img, in_range=(p2, p98))
            print('MIN',np.min(img),'MAX',np.max(img))
            img = (255*img).astype('uint8')
            img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(thumbpath, img)

        dim = tiff.TiffFile(filepath).series[0].shape
        res = 't\n%s\n%s\n%s' % (filepath, thumbpath, dim)
        return res
    except:
        log('Error in GetThumb, %s, %s' % (filepath,traceback.format_exc()))
        res = 't\n%s\n%s\n%s' % (filepath, '0','N/A')
        return res