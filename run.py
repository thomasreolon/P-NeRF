import os, sys
import requests
import os.path as osp
import zipfile
import shutil
from datetime import datetime

DEFAULT_EPOCHS = 350
ROOT_PATH = osp.dirname(os.path.abspath(__file__))
os.chdir(ROOT_PATH)

# util function to download checkpoints from GDrive
def download_file_from_google_drive(id, destination):
    # get link ID
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    # make request
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    
    # save result
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)      

# asks the user how many epochs to do
def n_epochs():
    n = input("Please insert number of ephocs... (-1 for default)\n>> ")
    try:
        n = int(n)
        if(n == -1):
            return DEFAULT_EPOCHS
        else:
            return n
    except:
        sys.stdout.write("Please insert a valid value\n")
        return n_epochs()

# download pretrained models
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

    n = input("Select the checkpoint to use:\n0) train from scratch\n1) human\n2) chair\nor: write the name of the checkpoint (eg. run_0)\n >> ")
    try:
        chk_type = int(n)
        run_name = f'run_{chk_type}'

        assert chk_type in {0,1,2}
        if chk_type == 1 and not osp.exists(ROOT_PATH+f'/checkpoints/{run_name}'):
            raise Exception("A pretrained model for humans is not available yet, please select another one")
        elif chk_type == 2 and not osp.exists(ROOT_PATH+f'/checkpoints/{run_name}'):
            shutil.copytree(ROOT_PATH+'/checkpoints/srn_chair', ROOT_PATH+f'/checkpoints/{run_name}')
        return run_name, chk_type
    except ValueError:
        return n, None  # n is the name of the model
    except Exception as e:
        sys.stdout.write(f"Please insert a valid value:\n{e}")
        return checkpoints()

if __name__=='__main__':

    # activate python env ((this works only for our pc))
    res = input("Do you want to activate pixelnerf environment? [y/N]\n>> ")
    if(res.lower()=='y'):
        activate_this_file = "pixelnerf_env/bin/activate_this.py"
        execfile(activate_this_file, dict(__file__=activate_this_file))

    # get number of epochs
    ep = n_epochs()

    # download checkpoints if necessary
    run_name, chk_type = checkpoints()

    # create dataset using videos in folder input
    chk_type = (chk_type==2) and 56 or 0
    os.system('python src/scripts/preproc.py --coco_class {chk_type}')

    # set up script to run the training
    os.system(f'python src/scripts/train.py -n {run_name} -c ./conf/exp/custom.conf -D ./input/dataset --epochs {ep} -V 8 --gpu_id=0')

