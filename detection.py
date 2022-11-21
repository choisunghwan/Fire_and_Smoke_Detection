# 필요한 패키지 import
import argparse  # 명령행 파싱(인자를 입력 받고 파싱, 예외처리 등) 모듈
import json
from datetime import datetime
import cv2  # OpenCV(실시간 이미지 프로세싱) 모듈
import pymssql
import torch  # Torch(머신러닝 라이브러리) 모듈
from websocket import create_connection
from util import *

# 실행을 할 때 인자값 추가


ws = create_connection("ws://192.168.206.195:8001")

# MSSQL 접속
conn = pymssql.connect(server="49.247.6.52", user="user2", password="user2", database="DB2", charset='utf8')

# Connection 으로부터 Cursor 생성
cursor = conn.cursor()
print(cursor)
# cursor.execute("INSERT INTO dbo.TB_PASSENGER_CP (UP_CP, DOWN_CP) values('0','0')")
# conn.commit()
# cursor.execute("SELECT max(PASSENGER_ID_CP) FROM dbo.TB_PASSENGER_CP")
# row3 = cursor.fetchone()


ap = argparse.ArgumentParser() # 인자값을 받을 인스턴스 생성
# 입력받을 인자값 등록
ap.add_argument("-i", "--input", type=str, help="input 비디오 경로")
ap.add_argument("-o", "--output", type=str, help="output 비디오 경로") # 비디오 저장 경로
ap.add_argument("-c", "--confidence", type=float, default=0.4, help="최소 확률")
# 퍼셉트론 : 입력 값과 활성화 함수를 사용해 출력 값을 다음으로 넘기는 가장 작은 신경망 단위
# - 입력 신호가 뉴런에 보내질 때 가중치가 곱해짐
# - 그 값들을 더한 값이 한계값을 넘어설 때 1을 출력
# - 이 때 한계값을 임계값이라고 함
ap.add_argument("-t", "--threshold", type=float, default=0.2, help="임계값")
ap.add_argument("-r", "--resolution", type=str, default=640, help="input 해상도(정확도를 높이려면 증가, 속도를 높이려면 감소)")
# 입력받은 인자값을 args에 저장
args = vars(ap.parse_args())

# CUDA 사용 가능 여부
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print("[CUDA 사용]")
    device = 'cuda:0'
else:
    print("[CPU 사용]")
    device = 'cpu'

# 모델 경로
# modelPath = "model.pt"
modelPath="C:\\Users\\csh\\Desktop\\SMARTFACTORY\\fire_and_smoke_detection\\fire_and_smoke_detection\\model.pt"


# 모델 로드
# - Deep Learning Model을 학습할 때 사용하는 대부분의 Framework 들은 기본적으로 32-bit Floating Point(FP32)로 학습
model = torch.load(modelPath, map_location=device)['model'].float().fuse().eval()
if device == 'cuda:0':
    model.half() # 계산량을 줄이기 위해 FP32 대신 CUDA에서 지원하는 Half Precision(FP16) 사용

# input 비디오 경로가 제공되지 않은 경우 webcam
if not args.get("input", False):
    print("[webcam 시작]")
    vs = cv2.VideoCapture(0)

# input 비디오 경로가 제공된 경우 video
else:
    print("[video 시작]")
    vs = cv2.VideoCapture(args["input"])

# 해상도 변경
vs.set(3, 640)
vs.set(4, 480)

writer = None
(W, H) = (None, None)


onFireCnt = 0
offFireCnt = 0

# 비디오 스트림 프레임 반복
while True:
    # 프레임 읽기
    ret, frame = vs.read()

    # 읽은 프레임이 없는 경우 종료
    if args["input"] is not None and frame is None:
        break

    # 프레임 크기
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    
    # 이미지 전처리
    image = frame # 이미지
    image_orig = image # 원본 이미지
    image = letterbox(image, new_shape=args["resolution"]) # 32-pixel-multiple 사각형으로 이미지 resize
    image = image[:, :, ::-1].transpose(2, 0, 1) # [:,:,::-1] : 각 pixel RGB 값 역순, transpose(2, 0, 1) : 이미지 행렬을 주어진 순서대로 축을 변경 (높이, 너비, 깊이) -> (깊이, 높이, 너비)
    image = np.ascontiguousarray(image) # np.ascontiguousarray() : 이미지 행렬을 복사하기위해 메모리의 연속 배열을 반환(속도 향상)
    image = torch.from_numpy(image).to(device) # from_numpy() : numpy array를 Tensor 자료형으로 변경(numpy array와 메모리를 공유(데이터 복사가 아닌 참조)하여 데이터를 변경시 둘 다 변경), to() : CPU 또는 GPU 사용
    image = image.half() if device == 'cuda:0' else image.float()  # cuda:0를 사용하는 경우 dtype(데이터 타입)을 uint8(8비트의 부호 없는 정수형 배열)에서 FP16으로 변경
    image /= 255.0  # 각 인덱스 값을 [0 ~ 255]에서 [0.0 ~ 1.0]로 변경
    image = image.unsqueeze(0) # 첫 번째 차원의 1 차원 추가
    
    try:
        # NMS(Non-Maximum Suppression) 적용
        # - NMS : object detector가 예측한 bounding box 중에서 정확한 bounding box를 선택하도록 하는 기법
        prediction = non_max_suppression(model(image)[0], args["confidence"], args["threshold"])

        # 예측 결과가 있는 경우
        if len(prediction):
            onFireCnt += 1
            offFireCnt = 0
            # scale 비율
            ratio = min(image.shape[2:][0] / image_orig.shape[0], image.shape[2:][1] / image_orig.shape[1])

            # padding 계산
            padding = (image.shape[2:][1] - (image_orig.shape[1] * ratio)) / 2, (image.shape[2:][0] - (image_orig.shape[0] * ratio)) / 2

            # rescale 적용
            prediction[:, :4][:, [0, 2]] -= padding[0] # x padding
            prediction[:, :4][:, [1, 3]] -= padding[1] # y padding
            prediction[:, :4][:, :4] /= ratio
            
            # Detection된 수 만큼 반복
            for x1, y1, x2, y2, confidence, class_ in prediction:
                # Detection 정보 출력
                if model.names[int(class_)] == 'fire': # fire
                    cv2.rectangle(frame, (int(x1) - 1, int(y1) - 21), (int(x2) + 1, int(y1) + 1), (50, 50, 200), -1)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (50, 50, 200), 2)
                    
                else: # smoke
                    cv2.rectangle(frame, (int(x1) - 1, int(y1) - 21), (int(x2) + 1, int(y1) + 1), (100, 100, 100), -1)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (100, 100, 100), 2)
                    print("연기 발생")
                    d = [{"req":"onSmoke", "contents":"1층 서버실 연기 발생"}]
                    json_data = json.dumps(d)
                    ws.send(json_data)
                # cv2.putText(frame, model.names[int(class_)] + " : " + str(round(float(confidence), 2)), (int(x1) + 5, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                cv2.putText(frame, model.names[int(class_)], (int(x1) + 5, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            print(onFireCnt)
            if onFireCnt >= 10:
                # 데이터 전송
                cursor.execute("INSERT INTO dbo.TB_FIREDETECT_LOG_CP (LOG_FIREDETECTTIME_CP, LOG_FIREDETECT_CONTENT_CP) values('"+datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]+"','1층 서버실')")
                conn.commit()
                print("화재가 발생")
                d = [{"req":"onFire", "contents":"1층 서버실 화재 발생"}]
                json_data = json.dumps(d)
                ws.send(json_data)
                onFireCnt = 0
        else:
            onFireCnt = 0
            offFireCnt += 1
            print(offFireCnt)
            if offFireCnt >= 30:
                print("정상 상태")
                offFireCnt = 0
                d = [{"req":"offFire", "contents":"정상"}]
                json_data = json.dumps(d)
                ws.send(json_data)

            #print(model.names)

    except Exception as error:
        print("에러 : {0}".format(error))
    
    # 프레임 크기 변경
    frame = cv2.resize(frame, (640, 480))
    
    # 프레임 출력
    cv2.imshow("Fire Detection", frame)

    # 'q' 키를 입력하면 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        d = [{"req":"offFireDetector", "contents":"카메라가 꺼져있습니다."}]
        json_data = json.dumps(d)
        ws.send(json_data)
        break

    # output video 설정
    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    # 비디오 저장
    if writer is not None:
        writer.write(frame)



# 종료
vs.release()
conn.close()
ws.close()
cv2.destroyAllWindows()
print("종료")