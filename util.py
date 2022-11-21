# 필요한 패키지 import
import numpy as np # 파이썬 행렬 수식 및 수치 계산 처리 모듈
import cv2 # OpenCV(실시간 이미지 프로세싱) 모듈
import torch # Torch(머신러닝 라이브러리) 모듈
import torchvision

# 32-pixel-multiple 사각형으로 이미지 resize
def letterbox(image, new_shape):
    # shape 추출
    shape = image.shape[:2] # 현재 입력 이미지 shape(width, height)
    print(shape)
    new_shape = (new_shape, new_shape) # 새로운 shape

    # scale 비율
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # padding 계산
    unpadding = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    padding_width, padding_height = new_shape[1] - unpadding[0], new_shape[0] - unpadding[1]
    padding_width, padding_height = np.mod(padding_width, 32), np.mod(padding_height, 32) # np.mod() : 나눗셈의 나머지를 반환
    padding_width /= 2
    padding_height /= 2

    # 이미지 resize
    if shape[::-1] != unpadding: # shape의 역순([::-1])과 unpadding이 다를 경우
        image = cv2.resize(image, unpadding, interpolation=cv2.INTER_LINEAR)
    
    # 상, 하, 좌, 우 padding 값 계산
    top, bottom = int(round(padding_height - 0.1)), int(round(padding_height + 0.1))
    left, right = int(round(padding_width - 0.1)), int(round(padding_width + 0.1))

    # cv2.copyMakeBorder : 이미지의 가장자리 생성
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # 이미지 반환
    return image

# NMS(Non-Maximum Suppression) : object detector가 예측한 bounding box 중에서 정확한 bounding box를 선택하도록 하는 기법
def non_max_suppression(prediction, confidence, threshold):
    candidates = prediction[..., 4] > confidence # 확률보다 큰 모든 후보
    
    # 예측 계산
    prediction = prediction[0][candidates[0]]
    prediction[:, 5:] *= prediction[:, 4:5]
    i, j = (prediction[:, 5:] > confidence).nonzero().T # torch.nonzero() : 0이 아닌 값의 인덱스를 저장한 2차원 Tensor, T : 전치 행렬(행과 열을 변경)

    # boxes 설정 
    # [center x, center y, width, height] -> [x1, y1, x2, y2]
    boxes = torch.zeros_like(prediction[:, :4]) # torch.zeros_like() : 동일한 크기(shape)만큼 0 값으로 채워진 Tensor 생성
    boxes[:, 0] = prediction[:, :4][:, 0] - prediction[:, :4][:, 2] / 2 # x1
    boxes[:, 1] = prediction[:, :4][:, 1] - prediction[:, :4][:, 3] / 2 # y1
    boxes[:, 2] = prediction[:, :4][:, 0] + prediction[:, :4][:, 2] / 2 # x2
    boxes[:, 3] = prediction[:, :4][:, 1] + prediction[:, :4][:, 3] / 2 # y2

    # Detections 추출(boxes, confidences, classes)
    prediction = torch.cat((boxes[i], prediction[i, j + 5, None], j[:, None].float()), dim=1) # torch.cat() : 두 개의 Tensor를 연결(dim : 연결할 차원의 인덱스)

    # NMS 수행(torch.ops.torchvision.nms)
    classes = prediction[:, 5:6]
    confidences = prediction[:, 4]
    boxes = prediction[:, :4] + classes
    index = torch.ops.torchvision.nms(boxes, confidences, threshold)

    # NMS를 적용힌 예측 결과 반환
    return prediction[index]
