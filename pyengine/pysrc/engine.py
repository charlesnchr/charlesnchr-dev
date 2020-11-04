import socket
import sys
import threading
import traceback
import os
import os.path
import shutil
import time
import zipfile
from skimage import io, transform, exposure
import tifffile as tiff
import hashlib
import mlsim_misc as mlsim
from misc import log
import ernet
import numpy as np
import cv2
from PIL import Image
import io
import pickle 
import matplotlib.pyplot as plt
from skimage import filters

def md5file(filename):
    with open(filename, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()


def start_communication(conn, address):
    def exit():
        conn.send('1'.encode())  # kill signal
        exitconfirmation = conn.recv(20).decode()
        if exitconfirmation == '1':
            return True
        else:
            return False

    try:
        log('connection from %s' % str(address))

        # receive data stream. it won't accept data packet greater than 2048 bytes
        data = conn.recv(2048).decode()
        log('received data: %s' % data)
        args = data.split('\n')
        if len(args) < 2:
            return

        cmd = args[0]
        args = args[1:]
        if cmd == 'GetThumb':
            log('GetThumb: %s' % str(data))
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

                conn.send(('t\n%s\n%s\n%s' % (filepath, thumbpath, dim)).encode())
            except:
                log('Error in GetThumb, %s, %s' % (filepath,traceback.format_exc()))
                conn.send(('t\n%s\n%s\n%s' % (filepath, '0','N/A')).encode())
                
            conn.recv(20).decode()  # ready to exit
            if exit():
                print('Can now close thread')
        elif cmd == 'Plugin_MLSIM':
            log('ML-SIM Reconstruct: %s' % str(data))
            exportdir = args[0]
            
            filepaths = []
            while True:
                conn.send('p'.encode())  # send paths confirmations
                data = conn.recv(2048)
                if chr(data[0]) == 'x':  # decodes the ASCII code
                    print('received all paths', len(filepaths))
                    break
                else:
                    # equivalent to chr(10), i.e. ASCII code 10
                    res_chunk = data.split('\n'.encode())
                    # filepaths.extend(res_chunk)
                    for e in res_chunk:
                        filepaths.append(e.decode())
            
            if len(filepaths) == 0:
                if exit():
                    print('Can now close thread')
            else:
                try:
                    log('now calling recon')
                    reconpaths = mlsim.reconstruct(exportdir,filepaths,conn)
                    log('sending back %s' % reconpaths)
                    conn.send(('2' + '\n'.join(reconpaths)).encode())
                except:
                    errmsg = traceback.format_exc()
                    log("Error in reconstruct %s" % errmsg)
                    conn.send(('2' + '\n'.join([])).encode())

                conn.recv(20).decode()  # ready to exit
                if exit():
                    print('Can now close thread')                
        elif cmd == 'Plugin_ERNet':
            log('Plugin_ERNet: %s' % str(data))
            exportdir = args[0]
            weka_colours = args[1] == 'true'
            stats_tubule_sheet = args[2] == 'true'
            save_in_original_folders = args[3] == 'true'
            
            filepaths = []
            while True:
                conn.send('p'.encode())  # send paths confirmations
                data = conn.recv(2048)
                if chr(data[0]) == 'x':  # decodes the ASCII code
                    print('received all paths', len(filepaths))
                    break
                else:
                    # equivalent to chr(10), i.e. ASCII code 10
                    res_chunk = data.split('\n'.encode())
                    # filepaths.extend(res_chunk)
                    for e in res_chunk:
                        filepaths.append(e.decode())
            
            if len(filepaths) == 0:
                if exit():
                    print('Can now close thread')
            else:
                try:
                    outpaths = ernet.segment(exportdir,filepaths,conn,weka_colours,stats_tubule_sheet,save_in_original_folders)
                    log('sending back %s' % outpaths)
                    conn.send(('2' + '\n'.join(outpaths)).encode())
                except:
                    errmsg = traceback.format_exc()
                    log("Error in reconstruct %s" % errmsg)
                    conn.send(('2' + '\n'.join([])).encode())

                conn.recv(20).decode()  # ready to exit
                if exit():
                    print('Can now close thread')                   
        elif cmd == 'exportToFolder' or cmd == 'exportToZip':
            print('export to folder', args[1:])
            if len(args) < 3:
                if exit():
                    print('Can now close thread')

            exportdir = args[0]
            files = args[1:]

            newdir = os.path.join(exportdir, 'ML-SIM Export ' +
                                  time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(newdir, exist_ok=True)

            for file in files:
                basename = os.path.basename(file)
                outfile = os.path.join(newdir, basename)
                file_exists = os.path.isfile(outfile)
                countidx = 2
                while file_exists:
                    rootname, ext = os.path.splitext(basename)
                    newbasename = '%s_%d.%s' % (rootname, countidx, ext)
                    outfile = os.path.join(newdir, newbasename)
                    file_exists = os.path.isfile(outfile)
                    countidx += 1
                shutil.copy2(file, outfile)

            if cmd == 'exportToZip':
                zipname = os.path.join(
                    exportdir, 'ML-SIM Export ' + time.strftime("%Y%m%d-%H%M%S") + '.zip')
                zippath = os.path.join(exportdir, zipname)
                zipf = zipfile.ZipFile(zippath, 'w', zipfile.ZIP_DEFLATED)
                for root, dirs, files in os.walk(newdir):
                    for file in files:
                        zipf.write(os.path.join(root, file), file)
                zipf.close()
                shutil.rmtree(newdir, ignore_errors=True)

                conn.send(('z%s' % zippath).encode())  # file window can open
            else:
                conn.send(('e%s' % newdir).encode())  # file window can open

            conn.recv(20).decode()  # ready to exit
            if exit():
                print('Can now close thread')
        else:  # cmd unknown
            if exit():
                print('Can now close thread')

        conn.close()  # close the connection
        log('exited thread %s' % str(address))
    except Exception as e:
        errmsg = traceback.format_exc()
        log(errmsg)
        # send_log(errmsg)


def receiveImage(conn,npixels,w,h):
    
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
            # print('received all bytes')
            # pickle.dump(bytesstr,open('data.pkl','wb'))
            # print('saved as npy')
            conn.send('c'.encode())
            img = Image.frombuffer("RGBA",(w,h),bytesstr,"raw","BGRA").convert("RGB")
            # img = filters.sobel(np.array(img).mean(2))
            
            return np.array(img)


def start_plugin_server(conn, address):
    try:
        log('connection from %s' % str(address))



        # receive data stream. it won't accept data packet greater than 2048 bytes  
        count = 0
        t0 = time.perf_counter()
        fps = np.ones((10,))

        plt.figure(figsize=(9,6),frameon=False)
        ax = plt.subplot(111,aspect = 'equal')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        while True:
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
                img = receiveImage(conn,npixels,w,h)
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
                sr = mlsim.reconstruct_image(stack)
                print('here with',sr.shape)

                # plt.cla()
                # plt.gca().imshow(sr,cmap='magma')
                # plt.pause(0.01)
                
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
        # send_log(errmsg)


def socketserver():
    log('starting socketserver')
    host = 'localhost'
    port = int(sys.argv[1])

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((host, port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(1)
    while True:
        print('now listening')
        conn, address = server_socket.accept()  # accept new connection
        th = threading.Thread(target=start_plugin_server, args=(conn, address))
        th.daemon = True
        th.start()


if __name__ == '__main__':

    log(os.getcwd())
    if len(sys.argv) > 2:
        mlsim.SetUseCloud(sys.argv[2])

    # if no argument given # cachefolder = '/Users/cc/ML-SIM/Library/tempresize' # if no argument given
    cachefolder = os.getenv('APPDATA') + '/ML-SIM-Library/tempresize'

    if len(sys.argv) > 3:
        cachefolder = sys.argv[3]

    # first time launching new version? clean up cache
    mlsim.SetCachefolder(cachefolder)
    os.makedirs(cachefolder, exist_ok=True)

    idx = cachefolder.split('/ML-SIM/')[-1]
    if idx.isdigit():
        idx = int(idx)
    else:
        idx = cachefolder.split('\\ML-SIM\\')[-1]
        if idx.isdigit():
            idx = int(idx)
        else:
            idx = 0  # don't clean up

    for i in range(1, idx):
        oldfolder = cachefolder.replace(
            '/ML-SIM/' + str(idx), '/ML-SIM/' + str(i))
        if os.path.isdir(oldfolder):
            shutil.rmtree(oldfolder, ignore_errors=True)

    socketserver()
