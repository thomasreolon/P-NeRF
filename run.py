import os, sys
DEFAULT_EPOCHS = 350
import requests
import os.path as osp
import zipfile
ROOT_PATH = osp.dirname(os.path.abspath(__file__))

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def n_epochs():
    n = input("Please insert number of ephocs... (0 for default) ")
    if(isinstance(n, int)):
        if(n == 0):
            return DEFAULT_EPOCHS
        else:
            return n
    else:
        sys.stdout.write("Please insert a valid value\n")
        return n_epochs()

def checkpoints():
    if not os.path.exists(ROOT_PATH+"/checkpoints"):
        print("Downloading checkpoints...\n")
        os.makedirs(ROOT_PATH+"/checkpoints")
        download_file_from_google_drive("1UO_rL201guN6euoWkCOn-XpqR2e8o6ju", ROOT_PATH+"/checkpoints/pixel_nerf_weights.zip")
        print("Extracting...")
        with zipfile.ZipFile(ROOT_PATH+"/checkpoints/pixel_nerf_weights.zip", 'r') as zip_ref:
            zip_ref.extractall(ROOT_PATH+"/checkpoints")
        os.rmdir(ROOT_PATH+"/checkpoints/srn_car")
        os.remove(ROOT_PATH+"/checkpoints/pixel_nerf_weights.zip")

    n = input("Select the checkpoint to use:\n0) train from scratch\n1) human\n2) chair\n")
    if(isinstance(n, int) and n in [0,1,2]):
        return n
    else:
        sys.stdout.write("Please insert a valid value\n")
        return checkpoints()

# activate python env
activate_this_file = "pixelnerf_env/bin/activate_this.py"
execfile(activate_this_file, dict(__file__=activate_this_file))

os.system('python src/scripts/preproc.py')
ep = n_epochs()
chk = checkpoints()
# os.system('python src/scripts/train.py -n srn_chair -c /home/thomasreolon/P-NeRF/conf/exp/custom.conf -D /home/thomasreolon/P-NeRF/input/dataset --epochs {ep} -V 8 --gpu_id=0'.format(ep = ep))

