# Logo_Eraser

# Prerequirement
```
!pip install openmim
!mim install mmcv-full
```

## Demo
| Before | After |
| ------ | ----- |
| <img src="https://github.com/Dohyeon-Kim1/Logo_Eraser/assets/63627639/5a921ae5-1fca-4860-8a32-835ee5f8c569" width="200" height="200"> | <img src="https://github.com/Dohyeon-Kim1/Logo_Eraser/assets/63627639/8910ad65-958a-4072-9b5f-80d306cbd6cd" width="200" height="200"> |

# Checkpoint
```
%cd Logo_Eraser
" If you don't have any save folders, then you should make save directories. ex) mkdir save "
CUDA_VISIBLE_DEVICES=1 Python3 main.py --logo_ckpt your_path --inpaint_ckpt your_path --video_path your_path --device cuda --save_dir Logo_Eraser/save

logo-checkpoint: https://huggingface.co/spaces/nathanjc/Logo_detection_YoloV7/resolve/main/logo_detection.pt
inpaint-checkpoint: https://drive.usercontent.google.com/download?id=10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3&export=download&authuser=0
```

