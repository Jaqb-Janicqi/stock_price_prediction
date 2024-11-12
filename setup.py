import subprocess
import sys
import os


# assert python 3.11.X
if sys.version_info < (3, 11):
    print('Please use Python 3.11.X')
    sys.exit(1)

requirements = [
    'pip3 install -r requirements.txt',
    'pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124'
]
for command in requirements:
    subprocess.run(command, shell=True)
