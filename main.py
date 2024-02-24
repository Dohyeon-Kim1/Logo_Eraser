import argparse
import cv2
import numpy as np
import os
from model.load import YoloDetection, LogoVideoInpainting
from utils.data import extract_frames, read_mask

def main(args):
    model1 = YoloDetection(args.logo_ckpt)
    # prepare dataset
    video_path = args.video_path
    frames, size = extract_frames(video_path)
    masks = read_mask(frames, model1, size) 
    video_length = len(frames)
    
    model2 = LogoVideoInpainting(device=args.device, inpaint_ckpt = args.inpaint_ckpt)
    new_video = model2.generate(frames, masks)
    
    ###
    print('Saving videos...')
    save_dir_name = args.save_dir
    ext_name = '_results.mp4'
    save_base_name = args.video_path.split('/')[-1]
    save_name = save_base_name.replace(
        '.mp4', ext_name)
    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)
    save_path = os.path.join(save_dir_name, save_name)
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             24, size)
    for f in range(video_length):
        comp = new_video[f].astype(np.uint8)
        writer.write(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
    writer.release()
    print(f'Finish test! The result video is saved in: {save_path}.')
    ###


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="logo_eraser")
    parser.add_argument("--logo_ckpt", type=str, default="", help="https://huggingface.co/spaces/nathanjc/Logo_detection_YoloV7/resolve/main/logo_detection.pt")
    parser.add_argument("--inpaint_ckpt", type=str, default="", help= "https://drive.usercontent.google.com/download?id=10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3&export=download&authuser=0")
    parser.add_argument("--video_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="the location in which model train or infernce")
    parser.add_argument("--save_dir", type = str, default = "")

    args = parser.parse_args()

    main(args)

    



