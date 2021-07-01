import os, sys
import requests
import os.path as osp
import zipfile
import shutil
import argparse

DEFAULT_EPOCHS = 350
ROOT_PATH = osp.dirname(os.path.abspath(__file__))
os.chdir(ROOT_PATH+'/..')

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
def checkpoints(base_model, run_name):
    # download pretrained checkpoints
    if not os.path.exists(ROOT_PATH+"/checkpoints"):
        print("Downloading checkpoints...\n")
        os.makedirs(ROOT_PATH+"/checkpoints")
        download_file_from_google_drive("1UO_rL201guN6euoWkCOn-XpqR2e8o6ju", ROOT_PATH+"/checkpoints/pixel_nerf_weights.zip")
        print("Extracting...")
        with zipfile.ZipFile(ROOT_PATH+"/checkpoints/pixel_nerf_weights.zip", 'r') as zip_ref:
            zip_ref.extractall(ROOT_PATH+"/checkpoints")
        os.rmdir(ROOT_PATH+"/checkpoints/srn_car")
        os.remove(ROOT_PATH+"/checkpoints/pixel_nerf_weights.zip")

    # create new checkpoint from pretrained
    if base_model == 'person' and not osp.exists(ROOT_PATH+f'/checkpoints/{run_name}'):
        raise Exception("A pretrained model for humans is not available yet, please select another one")
    elif base_model == 'chair' and not osp.exists(ROOT_PATH+f'/checkpoints/{run_name}'):
        shutil.copytree(ROOT_PATH+'/checkpoints/srn_chair', ROOT_PATH+f'/checkpoints/{run_name}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_name",
        "-n",
        type=str,
        help="checkpoint folder name",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=0,
        help="for how many epochs to train the model",
    )
    parser.add_argument(
        "--base_model",
        "-m",
        type=str,
        default=None,
        help="use a pretrained checkpoint as base",
    )
    parser.add_argument("--preprocess", action="prep_true", help="create dataset")
    parser.add_argument("--gen_video",  action="vide_true", help="make final video")
    parser.add_argument("--scratch",  action="scratch_true", help="override previous progress")

    args = parser.parse_args()

    # download checkpoints if necessary
    checkpoints(args.base_model, args.run_name)

    if args.preprocess:
        chk_type = (args.base_model=='chair' and 56) or (args.base_model=='person' and 0) or (args.base_model=='car' and 2)
        os.system(f'python src/scripts/preproc.py --coco_class {chk_type}')

    # set up script to run the training
    resume = (not args.scratch) and '--resume' or ''
    os.system(f'python src/scripts/train.py -n {args.run_name} -c ./conf/exp/custom.conf -D ./input/dataset --epochs {args.epochs} --gpu_id=0 {resume}')

    # generationg video
    if(args.gen_video):
        os.system(f'python src/scripts/gen_video.py -n {args.run_name} --gpu_id=0 --split test -P "6 4" -D ./input/dataset -S 0')
