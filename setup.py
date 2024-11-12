import subprocess
import sys

torch_commands = [
    'pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124',
    'pip3 install -r requirements.txt'
]
# assert python 3.11.X
if sys.version_info < (3, 11):
    print('Please use Python 3.11.X')
    sys.exit(1)

for command in torch_commands:
    subprocess.run(command, shell=True)