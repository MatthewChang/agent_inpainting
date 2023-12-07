# Script to unzip all zip files recursivly in place in the specified folder
# Usage: python unzip_visor.py <folder>

import os
import zipfile
import sys
from glob import glob
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('folder')
args = parser.parse_args()

files = glob(f"{args.folder}/**/*.zip",recursive=True)
for file in files:
    try:
        with zipfile.ZipFile(file,"r") as zip_ref:
            zip_ref.extractall(os.path.dirname(file))
    except zipfile.BadZipFile:
        print(f"Bad zip file: {file}")
        continue
    except:
        print(f"Unexpected error: {sys.exc_info()[0]}")
        continue
    print(f"Unzipped {file}")

