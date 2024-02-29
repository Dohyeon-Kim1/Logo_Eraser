# Logo_Eraser

# Prerequirement
```
!pip install openmim
!mim install mmcv-full
```
## Demo
<video width="320" height="240" controls>
  <source src="./figs/your_video.mp4" type="video/mp4">
# Checkpoint
```
%cd Logo_Eraser
" If you don't have any save folders, then you should make save directories. ex) mkdir save "
CUDA_VISIBLE_DEVICES=1 Python3 main.py --logo_ckpt your_path --inpaint_ckpt your_path --video_path your_path --device cuda --save_dir Logo_Eraser/save

logo-checkpoint: https://huggingface.co/spaces/nathanjc/Logo_detection_YoloV7/resolve/main/logo_detection.pt
inpaint-checkpoint: https://drive.usercontent.google.com/download?id=10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3&export=download&authuser=0
```

