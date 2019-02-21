import os
from toCSV import convert

lasArr = []
oripath = "/home/lenovo/Downloads/"
for root, dirs, files in os.walk(oripath):
    for file in files:
        if file.endswith(".las"):
             laspath = os.path.join(root, file)
             print(laspath)
             convert(laspath)
