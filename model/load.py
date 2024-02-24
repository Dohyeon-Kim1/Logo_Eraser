import torch
import importlib
import numpy as np
from tqdm import tqdm

from model.e2fgvi.e2fgvi_hq import InpaintGenerator
from utils.e2fgvi import to_tensors, get_ref_index
from utils.yolov7 import attempt_load, letterbox, non_max_suppression, scale_coords, xyxy2xywh


class LogoVideoInpainting:
  def __init__(self, device, inpaint_ckpt):
    model = InpaintGenerator().to(device)
    ckpt_path = inpaint_ckpt
    data = torch.load(ckpt_path, map_location = device)
    model.load_state_dict(data)
    model.eval()
    self.model = model.to(device)
    self.neighbor_stride = 5
    self.device = device

  def generate(self, frames, masks):
    imgs = to_tensors()(frames).unsqueeze(0) * 2 - 1  # 배치 차원을 받기 위해 unsqueeze
    frames = [np.array(f).astype(np.uint8) for f in frames]  # 이미지는 uint8을 많이 사용: 8bit짜리 0~255 + array로 바꿔줌
    # -> PIL 이미지를 array로 바꿈
    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2)
                    for m in masks]
    masks = to_tensors()(masks).unsqueeze(0)
    imgs, masks = imgs.to(self.device), masks.to(self.device)
    video_length = len(frames)
    comp_frames = [None] * video_length # comp_frames 리스트의 각 요소는 프레임을 나타내는 NumPy 배열.
    for f in tqdm(range(0, video_length, self.neighbor_stride)):
    # 이전 이후 프레임 얼마나 참고할지의 인덱스
      neighbor_ids = [i for i in range(max(0, f-self.neighbor_stride), min(video_length, f+self.neighbor_stride+1))]
      # 바로 전보다 조금 더 전에 있는 것도 확인하기 위한 인덱스 추출해주는 함수(위에 있음)
      ref_ids = get_ref_index(f, neighbor_ids, video_length)
      selected_imgs = imgs[:1, neighbor_ids+ref_ids, :, :, :]
      selected_masks = masks[:1, neighbor_ids+ref_ids, :, :, :]
      with torch.no_grad():
          masked_imgs = selected_imgs*(1-selected_masks)       # 마스크된 이미지 뽑기
          # masked_imgs = masked_imgs.permute(0,1,2,4,3)
          pred_img, _ = self.model(masked_imgs, len(neighbor_ids))  # 모델 돌리기

          # 돌린 모델 후처리
          pred_img = (pred_img + 1) / 2 #->0~1
          pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255  # 토치만 채널이 맨 앞에 있고 나머지는 채널이 맨 뒤. H,W,C 따라서 순서 바꿔주기 + 0~255

          for i in range(len(neighbor_ids)):
              idx = neighbor_ids[i]
              img = np.array(pred_img[i]).astype(
                  np.uint8)*binary_masks[idx] + frames[idx] * (1-binary_masks[idx])
              if comp_frames[idx] is None:
                  comp_frames[idx] = img
              else:
                  comp_frames[idx] = comp_frames[idx].astype(
                      np.float32)*0.5 + img.astype(np.float32)*0.5
    return comp_frames


class YoloDetection():
    def __init__(self, ckpt) :
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = attempt_load(ckpt, map_location=device)
        self.device = device

    ## input: PIL.Image
    def detect_img(self, img):
        img0 = np.array(img)[:,:,::-1]
        img = letterbox(img0, 640, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # 이미지를 float32로 변환
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        results = []  # 결과를 저장할 리스트를 초기화

        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        for i, det in enumerate(pred):  # detections per image

            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (conf, cls, *xywh)  # label format
                    results.append(line)  # 결과를 리스트에 추가

        return results  # 결과를 반환
