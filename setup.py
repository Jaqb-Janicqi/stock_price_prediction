import subprocess
import sys
import os
import zipfile


def install_talib():
    try:
        import talib
        return
    except ImportError:
        pass

    os_version = sys.platform
    # check if linux or unix
    if not os_version.startswith('linux'):
        import requests
        subprocess.check_call(["pip", "debug", "--verbose"])
        directory = 'talib'
        os.makedirs(directory, exist_ok=True)
        url = "https://github.com/TA-Lib/ta-lib-python/files/12437040/TA_Lib-0.4.28-cp311-cp311-win_amd64.whl.zip"
        f_path = os.path.join(directory, "TA_Lib-0.4.28-cp311-cp311-win_amd64.whl.zip")
        with open(f_path, "wb") as f:
            f.write(requests.get(url).content)
        with zipfile.ZipFile(f_path, "r") as zip_ref:
            zip_ref.extractall(directory)
        subprocess.check_call(["pip", "install", os.path.join(directory, "TA_Lib-0.4.28-cp311-cp311-win_amd64.whl")])
    else:
        url = "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
        import urllib.request
        urllib.request.urlretrieve(url, "ta-lib-0.4.0-src.tar.gz")
        subprocess.run(["tar", "-xzf", "ta-lib-0.4.0-src.tar.gz"])
        os.chdir("ta-lib")
        subprocess.run(["./configure", "--prefix=/usr"])
        subprocess.run(["make"])
        subprocess.run(["sudo", "make", "install"])
        subprocess.run([sys.executable, "-m", "pip", "install", "ta-lib"])
        
    # check if talib is installed properly
    try:
        import talib
    except ImportError:
        print('Automatic TA-Lib installation failed. Please install TA-Lib manually.')
        sys.exit(1)


def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def run_setup():
    # assert python 3.11.X
    if sys.version_info < (3, 11):
        print('Please use Python 3.11.X')
        sys.exit(1)

    # install requirements
    install_requirements()

    # check if talib is installed and install if not
    install_talib()
    

if __name__ == '__main__':
    run_setup()