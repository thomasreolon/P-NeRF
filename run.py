import os, sys
import requests
import os.path as osp
import zipfile
import shutil
from io import StringIO

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
        return run_name, (chk_type==2 and 56 or 0)
    except ValueError:
        return n, 56  # n is the name of the model,   56 --> default chairs
    except Exception as e:
        sys.stdout.write(f"Please insert a valid value:\n{e}")
        return checkpoints()

if __name__=='__main__':
    # activate python env ((this works only for our pc))
    activate_this_file = "pixelnerf_env/bin/activate_this.py"
    if(osp.exists(activate_this_file)):
        execfile(activate_this_file, dict(__file__=activate_this_file))

    ##### DEFAULT SETTINGS, usage =  ||  python run.py --default run_123  ||
    if len(sys.argv) == 3 and sys.argv[1] == '--default':
        run_name, chk_type, ep, gen_vid = sys.argv[2], 56, 0, 'y'
    else:
        # download checkpoints if necessary
        run_name, chk_type = checkpoints()

        # get number of epochs
        ep = n_epochs()

        # ask before if you want the video
        gen_vid = input("Do you want to generate the video? [y/N]\n>> ")

    # create dataset using videos in folder input
    proceed = (len(sys.argv)>1 and sys.argv[1] == '--default') and -1 or 0
    os.system(f'python src/scripts/preproc.py --coco_class {chk_type} --proceed {proceed}')

    # set up script to run the training
    os.system(f'python src/scripts/train.py -n {run_name} -c ./conf/exp/custom.conf -D ./input/dataset --epochs {ep} --gpu_id=0 --resume')

    # generationg video
    if(gen_vid.lower()=='y'):
        os.system(f'python src/scripts/gen_video.py -n {run_name} --gpu_id=0 --split test -P "6 4" -D ./input/dataset -S 0')
