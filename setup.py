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
        # https://www.wheelodex.org/projects/talib-binary/wheels/talib_binary-0.4.19-cp37-cp37m-manylinux1_x86_64.whl/
        url = 'https://files.pythonhosted.org/packages/00/61/a68a9276a3c166df8717927780d994496ee4cb5299903a409f93689a2b4e/talib_binary-0.4.19-cp37-cp37m-manylinux1_x86_64.whl'
        subprocess.check_call(["pip", "install", url])
    
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