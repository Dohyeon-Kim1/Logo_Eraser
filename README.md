# Logo_Eraser

# Prerequirement
```
!pip install openmim
!mim install mmcv-full
```
## Demo
![teaser](./figs/teaser.gif)
https://github.com/Dohyeon-Kim1/Logo_Eraser/assets/63627639/b3ab3a4f-25c6-4ce4-bcce-c308524bdbcb

https://github.com/Dohyeon-Kim1/Logo_Eraser/assets/63627639/359cde88-2f2f-4860-8560-163359c9d26f
# Checkpoint
```
%cd Logo_Eraser
" If you don't have any save folders, then you should make save directories. ex) mkdir save "
CUDA_VISIBLE_DEVICES=1 Python3 main.py --logo_ckpt your_path --inpaint_ckpt your_path --video_path your_path --device cuda --save_dir Logo_Eraser/save

logo-checkpoint: https://huggingface.co/spaces/nathanjc/Logo_detection_YoloV7/resolve/main/logo_detection.pt
inpaint-checkpoint: https://drive.usercontent.google.com/download?id=10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3&export=download&authuser=0
```

