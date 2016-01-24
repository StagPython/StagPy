"""download the official get-pip.py script"""
from urllib.request import urlopen
from shutil import copyfileobj

url = 'https://bootstrap.pypa.io/get-pip.py'
with urlopen(url) as urlstream, open('.get-pip.py', 'wb') as out_file:
        copyfileobj(urlstream, out_file)
