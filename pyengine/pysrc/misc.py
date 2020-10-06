import requests
import datetime

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
