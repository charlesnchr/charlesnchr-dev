import requests
import datetime

print('Initting misc')
loggedIn_emailaddr = ''
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


def set_email(emailaddr):
    global loggedIn_emailaddr
    loggedIn_emailaddr = emailaddr

def get_email():
    global loggedIn_emailaddr
    return loggedIn_emailaddr

