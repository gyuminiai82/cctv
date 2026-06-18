import cv2
import numpy as np
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from ultralytics import YOLO

app = FastAPI(title="CCTV Monitoring Portfolio")

# 템플릿 디렉토리 설정
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# YOLOv8n 가벼운 모델 로드 (최초 호출 시 자동 다운로드)
model = YOLO("yolov8n.pt")

# 카운팅 및 차량 추적을 위한 전역 변수
vehicle_counts = {
    "car": 0,
    "truck": 0,
    "bus": 0,
    "motorcycle": 0,
    "total": 0
}

# 이미 카운트 처리한 객체 ID 추적용 셋
counted_ids = set()

# 객체의 실시간 이전 좌표 기록 (tracker_id -> 이전 프레임의 center y좌표)
object_history = {}

# COCO 클래스 번호 매핑
CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def generate_frames():
    global vehicle_counts, counted_ids, object_history
    
    # 테스트용 오픈 RTSP (공공 CCTV 라이브 스트림)
    # 실제 C310 사용 시: "rtsp://아이디:비밀번호@192.168.x.x:554/stream1"
    RTSP_URL = os.getenv("RTSP_URL", "rtsp://210.99.70.120:1935/live/cctv006.stream")
    
    print(f"[INFO] RTSP 스트림 연결 시도 중: {RTSP_URL}")
    camera = cv2.VideoCapture(RTSP_URL)
    
    # RTSP 스트림 접속 실패 시 더미 영상 송출
    if not camera.isOpened():
        print("[WARN] RTSP 서버 연결에 실패했습니다. 테스트용 더미 화면을 송출합니다.")
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "RTSP Connection Failed", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, "Displaying Dummy Frame", (140, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # 더미 프레임에도 가상의 차량과 카운트 표시를 시뮬레이션
            cv2.putText(frame, f"Dummy Active - Total Count: {vehicle_counts['total']}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)
            
    # 정상적으로 RTSP 스트림을 받아오는 경우
    while True:
        success, frame = camera.read()
        if not success:
            print("[WARN] RTSP 프레임 수신 실패. 1초 대기 후 계속 진행...")
            time.sleep(1)
            continue
            
        height, width, _ = frame.shape
        # 카운팅 선의 Y좌표 설정 (화면 하단 약 65% 지점)
        line_y = int(height * 0.65)
        
        # YOLOv8 추적 실행 (차량 관련 클래스 2: car, 3: motorcycle, 5: bus, 7: truck)
        results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)
        
        # 탐지 결과 가공 및 카운팅 판단
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, obj_id, cls_id in zip(boxes, ids, clss):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                class_name = CLASS_NAMES.get(cls_id, "car")
                
                # 라인 통과 여부 검사 (중심점 cy 기준)
                if obj_id in object_history:
                    prev_cy = object_history[obj_id]
                    
                    # 객체가 라인을 위->아래 혹은 아래->위로 가로질렀는지 체크
                    if obj_id not in counted_ids:
                        if (prev_cy < line_y <= cy) or (prev_cy > line_y >= cy):
                            vehicle_counts[class_name] += 1
                            vehicle_counts["total"] += 1
                            counted_ids.add(obj_id)
                            print(f"[EVENT] 차량 카운트! ID:{obj_id} | {class_name} | Total:{vehicle_counts['total']}")
                
                # 이전 위치 갱신
                object_history[obj_id] = cy
                
                # 영상에 바운딩 박스 및 정보 오버레이 그리기
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{obj_id} {class_name}", (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 화면에 카운팅 가로선 그리기 (주황색)
        cv2.line(frame, (0, line_y), (width, line_y), (0, 165, 255), 2)
        cv2.putText(frame, "COUNTING LINE", (10, line_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # 영상 화면 좌측 상단에 실시간 통계 그리기
        cv2.putText(frame, f"TOTAL: {vehicle_counts['total']}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Car:{vehicle_counts['car']} | Truck:{vehicle_counts['truck']} | Bus:{vehicle_counts['bus']} | Moto:{vehicle_counts['motorcycle']}", 
                    (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
    camera.release()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """관제 대시보드 메인 페이지 렌더링"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    """웹캠 영상 실시간 스트리밍 엔드포인트"""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/count")
async def get_count():
    """실시간 차량 카운팅 통계 데이터 반환"""
    return vehicle_counts

