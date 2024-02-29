# Logo_Eraser
```
본 toy project는 프로메테우스의 기초 스터디 프로젝트의 일환으로 
logo detection part : video 속 logo를 detect 한다.
logo inpainting part: detect된 logo를 지운다.
두가지를 조합하여 동영상을 입력으로 주었을 때, 그에 대한 출력으로 로고가
지워진 동영상을 얻는 것이 목적입니다.
logo detection part에서는 Yolov7모델을 이용 하였고,
logo inpainting part에서는 End-to-End framework for Flow-Guided 
Video Inpainting(e2fgvi)를 이용하였습니다. 
아래의 내용들을 잘 활용하신다면 좋은 결과를 얻으실 수 있을 것입니다.
```
# Prerequirement
알아서 필요한거 pip install -q 하시면 됩니다.
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

