# P-NeRF
Slight modifications to the PixelNeRF official repository, to use their code for unitn videos


## Fast Scripts

```sh
python src/scripts/preproc.py    
python src/scripts/train.py -n unitn_chair -D input/dataset --gpu_id=0 --resume
python src/scripts/gen_video.py -n srn_chair --gpu_id=0 --split test -P 6 -D dataset/chairs/chairs -S 0
```
