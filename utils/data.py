import torch
import cv2
import numpy as np
from PIL import Image


def extract_frames(video_path):
    video_capture = cv2.VideoCapture(video_path)
    frames = []

    while True:
        success, frame = video_capture.read()
        if not success: # 프레임을 성공적으로 읽어오지 않았다면(=False, 더이상 읽을 프레임이 없다)
            break

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        w,h = pil_image.size
        pil_image = pil_image.resize((w//4, h//4),Image.NEAREST) # 1/4배
        frames.append(pil_image)

    video_capture.release()
    return frames, frames[0].size


def create_mask(frame, bboxes):
    w, h = frame.size
    mask = np.array(bbox2mask((w,h), bboxes))
    mask = np.array(mask > 0).astype(np.uint8)          # mask 아닌 부분이 0. T/F를 1/0으로 바꿔줌
    mask = cv2.dilate(mask, cv2.getStructuringElement(  # dilate: 마스크(1인 부분) 확장시켜주는 함수 -> mask는 원래보다 큰게 좋음
            cv2.MORPH_CROSS, (3, 3)), iterations=4)
    return Image.fromarray(mask*255)


def bbox2mask(img_size, bboxes): # img_size = (h ,w) # xywh = 0 ~ 1
  mask = torch.zeros(img_size[1], img_size[0], dtype=torch.float32)
  for i in range(len(bboxes)):
      x_, y_, w_, h_ = bboxes[i][2:]
      x, y, w, h = int(img_size[0]*x_), int(img_size[1]*y_), int(img_size[0]*w_ / 2), int(img_size[1]*h_ / 2)
      mask[y-h:y+h, x-w:x+w] += 1.0
  return mask

def read_mask(frames, model, size):
    masks = []

    for mp in frames:
        m = mp.resize(size, Image.NEAREST)
        mask = bbox2mask(size, model.detect_img(m))
        mask = np.array(mask)
        mask = np.array(mask > 0).astype(np.uint8)          # mask 아닌 부분이 0. T/F를 1/0으로 바꿔줌
        mask = cv2.dilate(mask, cv2.getStructuringElement(  # dilate: 마스크(1인 부분) 확장시켜주는 함수 -> mask는 원래보다 큰게 좋음
            cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(mask*255))          # PIL이미지의 list가 됨
    return masks
