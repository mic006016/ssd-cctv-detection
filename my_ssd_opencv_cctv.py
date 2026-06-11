cam_str = ("v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, format=(string)YUY2,framerate=30/1 ! videoconvert ! video/x-raw,width=640,height=480,format=BGR ! appsink")

import cv2
import numpy as np
#1. 시스템, opencv, 그리고 하위 디렉토리 vision에 있는 ssd 패키지 임포트
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer

#2.  opencv, system 패키지 임포트
import cv2
import sys

import time

#3.  타임스탬프가 포함된 파일 이름 생성 함수
def getLogFileName():
	timestamps = time.time()
	return "log_{}.txt".format(timestamps)
	
#4.  로그파일에 넣을 연월일시분초 문자열
def getTimeLog():
	secs = time.time()
	tm = time.localtime(secs)
	return "{}:{}:{}:{}:{}:{}".format(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)

#5. 객체인식 결과를 이미지에 표시하는 함수
def imageProcessing(frame, predictor, class_names, f):
    # 추론을 위해 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 모델에 이미지 파일을 입력하여 추론
    boxes, labels, probs = predictor.predict(image, 10, 0.4)

    # 추론 결과인 박스와 클래스 명을 이미지에 표시
    for i in range(boxes.size(0)):
        # 신뢰도 0.5 이상의 박스만 표시
        if(probs[i]>0.5):
            # 타임로그, 라벨, 신뢰도를 로그에 남김.
            Data = "[{}] classID: {} conf:{}\n".format(getTimeLog(), labels[i], probs[i])
            f.write(Data)
            # 바운딩박스 표시           
            box = boxes[i, :]
            
            # 1. 텐서에서 순수 숫자를 뽑아내어 확실한 정수로 변환
            x1 = int(box[0].item())
            y1 = int(box[1].item())
            x2 = int(box[2].item())
            y2 = int(box[3].item())
            
            # 라벨 인덱스도 안전하게 정수로 변환
            label_idx = int(labels[i].item())
            
            # 2. 순수한 정수 좌표로 사각형 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 4)
            
            label = f"{class_names[label_idx]}: {probs[i]:.2f}"
            
            # 3. 텍스트도 정수 좌표로 입력
            cv2.putText(frame, label,
                (x1 + 20, y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # 폰트크기
                (255, 0, 255),
                2)  # 선의 유형


    return frame

#6. 영상 딥러닝 프로세싱 함수
def videoProcess(openpath, net_type, model_path, label_path):
    # 라벨 파일을 읽어 클래스 이름들 세팅
    class_names = [name.strip() for name in open(label_path, encoding='cp949', errors='ignore').readlines()]

    # 모델 파일을 읽어 모델 로딩
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)

    net.load(model_path)

    # 네트워크 지정
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

    # 카메라나 영상으로부터 이미지를 갖고오기 위해 연결 열기
    cap = cv2.VideoCapture(openpath)
    if cap.isOpened():
        print("Video Opened")
    else:
        print("Video Not Opened")
        print("Program Abort")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    # 영상보여주기 위한 opencv 창 생성
    cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)
	
    # 로그파일 생성
    logfilename = getLogFileName()
    f = open(logfilename, 'w')

    try:
        while cap.isOpened():
            # 이미지 프레임 읽어오기
            ret, frame = cap.read()
            if ret: 
                # 이미지 프로세싱 진행한 후 그 결과 이미지 보여주기			
                result = imageProcessing(frame, predictor, class_names, f)
                cv2.imshow("Output", result)
            else:
                break

            if cv2.waitKey(int(1000.0/120)) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:  
        print("key int")
        cap.release()
        cv2.destroyAllWindows()
        return
    # 프로그램 종료 후 사용한 리소스를 해제한다.
    cap.release()

    cv2.destroyAllWindows()
    f.close()
    return
   
#7. 인자가 4보다 작으면 종료, 인자가 4면 카메라 추론 시작, 인자가 4보다 크면 영상파일 추론
if len(sys.argv) < 4:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path>')
    sys.exit(0)

if len(sys.argv) == 4:
    gst_str = cam_str
    print("camera 0")

else:
    gst_str = sys.argv[4]
    print(gst_str)

net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]

#gst_str = "./data/run3.mp4"

videoProcess(gst_str, net_type, model_path, label_path)

