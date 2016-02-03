"""download the official get-pip.py script"""
from urllib.request import urlopen
from shutil import copyfileobj

URL = 'https://bootstrap.pypa.io/get-pip.py'
with urlopen(URL) as urlstream, open('.get-pip.py', 'wb') as out_file:
    copyfileobj(urlstream, out_file)
