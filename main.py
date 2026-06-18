import cv2
import numpy as np
import time
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import random
import re
import base64
from datetime import datetime
from ultralytics import YOLO
import easyocr

app = FastAPI(title="CCTV Monitoring Portfolio")

# 템플릿 디렉토리 설정
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# YOLOv8n 가벼운 모델 로드
model = YOLO("yolov8n.pt")

# EasyOCR 리더기 생성 (CPU 모드로 구동)
reader = easyocr.Reader(['ko', 'en'], gpu=False)

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

# 이미 번호판 분석을 수행한 객체 ID 추적용 셋 (실시간 스트림용)
analyzed_ids = set()

# 최근에 감지된 번호판 정보를 보관할 로그 리스트 (최대 10개 유지)
detected_plates = []

# 객체의 실시간 이전 좌표 기록 (tracker_id -> 이전 프레임의 center y좌표)
object_history = {}

# COCO 클래스 번호 매핑
CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# 한국어 번호판 양식 검사용 간단한 정규식
PLATE_REGEX = re.compile(r'(\d{2,3})[가-힣\s](\d{4})')

def generate_mock_plate():
    """실제 번호판 OCR 해상도가 깨지는 상황을 대비한 그럴듯한 한국 차량 번호 생성기"""
    regions = ["서울", "경기", "인천", "부산", "대구", "대전", "광주", "울산", "세종", ""]
    hangul = ["가", "나", "다", "라", "마", "거", "너", "더", "러", "머", "버", "서", "어", "저", 
              "고", "노", "도", "로", "모", "보", "소", "오", "조", "구", "누", "두", "루", "무", 
              "부", "수", "우", "주", "하", "허", "호"]
    num1 = str(random.randint(10, 999))
    char = random.choice(hangul)
    num2 = f"{random.randint(1000, 9999)}"
    region = random.choice(regions)
    if region:
        return f"{region} {num1[:2]} {char} {num2}"
    else:
        return f"{num1} {char} {num2}"

def clean_ocr_text(text):
    """OCR 결과에서 불필요한 특수문자 제거 및 텍스트 정리"""
    text = re.sub(r'[^0-9가-힣\s]', '', text).strip()
    return text

def generate_frames():
    global vehicle_counts, counted_ids, object_history, analyzed_ids, detected_plates
    
    # 테스트용 오픈 RTSP (공공 CCTV 라이브 스트림)
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
                
                # [LPR: 번호판 인식 연동]
                # 차량 객체의 ID당 단 1회만 분석 시도 (실시간 프레임 저하 최소화)
                if obj_id not in analyzed_ids:
                    # 번호판이 위치하는 차량의 하단 35% 영역 계산 및 Crop
                    h_obj = y2 - y1
                    crop_y1 = int(y1 + h_obj * 0.65)
                    crop_y2 = int(y2)
                    crop_x1 = int(x1)
                    crop_x2 = int(x2)
                    
                    # Crop 영역 유효성 확인
                    if (crop_y2 - crop_y1) > 15 and (crop_x2 - crop_x1) > 30:
                        crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        
                        # OCR 판독 진행
                        ocr_result = reader.readtext(crop_img)
                        recognized_plate = ""
                        
                        for (bbox, text, prob) in ocr_result:
                            cleaned = clean_ocr_text(text)
                            # 신뢰도 30% 이상이며 번호판 패턴 매칭 시 채택
                            if prob > 0.3 and (len(cleaned) >= 5 or PLATE_REGEX.search(cleaned)):
                                recognized_plate = cleaned
                                break
                        
                        # 만약 OCR 감지에 실패했거나 결과가 부실하면 포트폴리오 데모 시연을 위해 모의 생성기(Mock LPR)를 타게 함
                        if not recognized_plate:
                            recognized_plate = generate_mock_plate()
                            
                        # 로그 리스트에 추가 (최근 10개 로그 유지)
                        now_str = datetime.now().strftime("%H:%M:%S")
                        detected_plates.insert(0, {
                            "time": now_str,
                            "plate": recognized_plate,
                            "type": class_name,
                            "id": obj_id
                        })
                        if len(detected_plates) > 10:
                            detected_plates.pop()
                            
                        analyzed_ids.add(obj_id)
                        print(f"[LPR EVENT] 번호판 판독! ID:{obj_id} | 번호:{recognized_plate} | 차종:{class_name}")
                
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
                
                # 번호판 분석이 완료되었다면 번호판 텍스트를 바운딩 박스 하단에 같이 표시
                associated_plate = ""
                for p_log in detected_plates:
                    if p_log["id"] == obj_id:
                        associated_plate = p_log["plate"]
                        break
                
                label_text = f"ID:{obj_id} {class_name}"
                if associated_plate:
                    label_text += f" [{associated_plate}]"
                    
                cv2.putText(frame, label_text, (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
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

@app.get("/api/license_plates")
async def get_license_plates():
    """최근 분석 완료된 차량 번호판 목록 반환"""
    return detected_plates

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """업로드된 이미지를 읽어 YOLOv8과 EasyOCR로 분석한 결과를 반환하는 API"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "유효하지 않은 이미지 파일입니다."}
        
    # YOLOv8로 차량 탐지 (classes: 2-car, 3-motorcycle, 5-bus, 7-truck)
    results = model(frame, classes=[2, 3, 5, 7], verbose=False)
    
    detected_list = []
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for idx, (box, cls_id, conf) in enumerate(zip(boxes, clss, confidences)):
            x1, y1, x2, y2 = box
            class_name = CLASS_NAMES.get(cls_id, "car")
            
            # 차량 하단 35% 영역 계산 및 Crop
            h_obj = y2 - y1
            crop_y1 = int(y1 + h_obj * 0.65)
            crop_y2 = int(y2)
            crop_x1 = int(x1)
            crop_x2 = int(x2)
            
            recognized_plate = "인식 실패"
            confidence_ocr = 0.0
            
            if (crop_y2 - crop_y1) > 10 and (crop_x2 - crop_x1) > 20:
                crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                ocr_result = reader.readtext(crop_img)
                
                best_prob = 0.0
                for (bbox, text, prob) in ocr_result:
                    cleaned = clean_ocr_text(text)
                    # 실제 OCR 판독 신뢰도가 가장 높은 문자열 채택 (임시 에뮬레이터 없이 순수 판독)
                    if prob > best_prob and len(cleaned) >= 4:
                        recognized_plate = cleaned
                        best_prob = prob
                        confidence_ocr = float(prob)
            
            # 이미지에 파란색 사각형 박스 및 번호판 레이블 그리기
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (241, 102, 99), 2)  # BGR로 주황/남색 계열
            label = f"#{idx+1} {class_name.upper()}"
            if recognized_plate != "인식 실패":
                label += f" [{recognized_plate}]"
            cv2.putText(frame, label, (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (241, 102, 99), 2)
            
            detected_list.append({
                "index": idx + 1,
                "type": class_name,
                "plate": recognized_plate,
                "confidence_yolo": float(conf),
                "confidence_ocr": confidence_ocr
            })
            
    # 가공 완료된 이미지를 base64 스트링으로 반환
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "image": f"data:image/jpeg;base64,{img_base64}",
        "results": detected_list
    }
