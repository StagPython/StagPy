"""download the official get-pip.py script"""
from urllib.request import urlopen
from shutil import copyfileobj
from sys import argv

if len(argv) > 1:
    OUT_NAME = argv[1]
else:
    OUT_NAME = 'get-pip.py'

URL = 'https://bootstrap.pypa.io/get-pip.py'
with urlopen(URL) as urlstream, open(OUT_NAME, 'wb') as out_file:
    copyfileobj(urlstream, out_file)
