import cv2
import numpy as np
import time
import asyncio
from fastapi import FastAPI, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import random
import base64
from datetime import datetime
from ultralytics import YOLO
import easyocr
import re

app = FastAPI(title="CCTV Monitoring Portfolio")

# 템플릿 디렉토리 설정
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# YOLOv8n 가벼운 모델 로드
model = YOLO("yolov8n.pt")

# EasyOCR 리더 초기화 (한국어, 영어 지원)
print("[INFO] EasyOCR 모델을 로드합니다. (초기 1회 약간의 시간이 소요될 수 있습니다.)")
reader = easyocr.Reader(['ko', 'en'], gpu=False)

# 카운팅 및 차량 추적을 위한 전역 변수
vehicle_counts = {
    "car": 0,
    "truck": 0,
    "bus": 0,
    "total": 0
}

# 이미 카운트 처리한 객체 ID 추적용 셋
counted_ids = set()

# 객체의 실시간 이전 좌표 기록 (tracker_id -> 이전 프레임의 center y좌표)
object_history = {}

# COCO 클래스 번호 매핑
CLASS_NAMES = {
    2: "car",
    5: "bus",
    7: "truck"
}

def generate_frames():
    global vehicle_counts, counted_ids, object_history
    
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
        
        # YOLOv8 추적 실행 (차량 관련 클래스 2: car, 5: bus, 7: truck)
        results = model.track(frame, persist=True, classes=[2, 5, 7], verbose=False)
        
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
                crossed = False
                if obj_id in object_history:
                    prev_cy = object_history[obj_id]
                    
                    # 객체가 라인을 위->아래 혹은 아래->위로 가로질렀는지 체크
                    if (prev_cy <= line_y <= cy) or (prev_cy >= line_y >= cy):
                        crossed = True
                
                # 프레임 끊김으로 인해 ID가 바뀌면서 라인을 통과하는 경우를 대비한 안전장치 (오차 30px)
                if not crossed and abs(cy - line_y) < 30:
                    crossed = True
                    
                if crossed and obj_id not in counted_ids:
                    vehicle_counts[class_name] += 1
                    vehicle_counts["total"] += 1
                    counted_ids.add(obj_id)
                    #print(f"[EVENT] 차량 카운트! ID:{obj_id} | {class_name} | Total:{vehicle_counts['total']}")
                
                # 이전 위치 갱신
                object_history[obj_id] = cy
                
                # 영상에 바운딩 박스 및 정보 오버레이 그리기
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                
                # 라벨로 표시
                label_text = f"ID:{obj_id} {class_name}"
                cv2.putText(frame, label_text, (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 화면에 카운팅 가로선 그리기 (주황색)
        cv2.line(frame, (0, line_y), (width, line_y), (0, 165, 255), 2)
        cv2.putText(frame, "COUNTING LINE", (10, line_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # 영상 화면 좌측 상단에 실시간 통계 그리기
        cv2.putText(frame, f"TOTAL: {vehicle_counts['total']}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Car:{vehicle_counts['car']} | Truck:{vehicle_counts['truck']} | Bus:{vehicle_counts['bus']}", 
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

@app.websocket("/ws/count")
async def websocket_count(websocket: WebSocket):
    """웹소켓을 통한 실시간 차량 카운팅 통계 데이터 스트리밍"""
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(vehicle_counts)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print("[INFO] 웹소켓 클라이언트 연결 종료")

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """업로드된 이미지를 읽어 YOLOv8로 분석한 결과를 반환하는 API"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "유효하지 않은 이미지 파일입니다."}
        
    # YOLOv8로 차량 탐지 (classes: 2-car, 5-bus, 7-truck)
    results = model(frame, classes=[2, 5, 7], verbose=False)
    
    detected_list = []
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for idx, (box, cls_id, conf) in enumerate(zip(boxes, clss, confidences)):
            x1, y1, x2, y2 = box
            class_name = CLASS_NAMES.get(cls_id, "car")
            
            # 차량 영역 잘라내기
            vehicle_img = frame[int(y1):int(y2), int(x1):int(x2)]
            plate_text = ""
            
            # EasyOCR로 번호판 인식 (번호판 위치 특성 고려)
            if vehicle_img.size > 0:
                h, w = vehicle_img.shape[:2]
                # 차량 하단 60% 영역에 번호판이 집중되므로 해당 영역만 잘라서 간섭 최소화 (광고, 전화번호 배제)
                bottom_half = vehicle_img[int(h * 0.6):, :]
                
                if bottom_half.size > 0:
                    ocr_results = reader.readtext(bottom_half)
                    
                    # 바운딩 박스 위치를 기준으로 텍스트 정렬 (위->아래, 좌->우)
                    boxes = []
                    for (bbox, text, prob) in ocr_results:
                        center_y = (bbox[0][1] + bbox[2][1]) / 2
                        center_x = (bbox[0][0] + bbox[2][0]) / 2
                        boxes.append((center_y, center_x, text))
                        
                    # y축 좌표를 15px 단위로 그룹화하여 같은 줄에 있는 텍스트는 x축 정렬되도록 함
                    boxes.sort(key=lambda b: (round(b[0] / 15), b[1]))
                    
                    combined_text = "".join([b[2] for b in boxes])
                    
                    # 영어도 남겨둠 (한글을 영어로 오인식하는 경우 대비)
                    cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9]', '', combined_text)
                    
                    # 매우 엄격하면서도 유연한 번호판 정규식: (지역명 0~2자) + (숫자 2~3자) + (필수 문자 1~2자) + (숫자 4자)
                    # 가운데 문자를 필수로 두어 '전화번호'가 번호판으로 오인식되는 것을 차단!
                    # 1->I, l / 0->O, o / 2->Z / 5->S / 8->B 와 같은 OCR 숫자 오인식을 완벽 대비
                    pattern = r'([가-힣]{0,2})([0-9IlOoZzSsBb]{2,3})([가-힣a-zA-Z]{1,2})([0-9IlOoZzSsBb]{4})'
                    matches = list(re.finditer(pattern, cleaned_text))
                    
                    if matches:
                        # 여러 개가 매칭될 경우 점수를 매겨 가장 번호판에 가까운 것을 선택
                        best_match = None
                        best_score = -1
                        
                        for m in matches:
                            m_region = m.group(1)
                            # 숫자 부분 영문 오인식 글자를 실제 숫자로 변환
                            trans_table = str.maketrans('IlOoZzSsBb', '1100225588')
                            m_digits1 = m.group(2).translate(trans_table)
                            m_digits2 = m.group(4).translate(trans_table)
                            m_char = m.group(3)
                            
                            score = 0
                            if m_region: score += 2
                            if m_char and re.search(r'[가-힣]', m_char): score += 2
                            if len(m_digits1) in [2, 3]: score += 1
                            
                            if score >= best_score:
                                best_score = score
                                # match 객체 대신 변환된 문자열 데이터를 저장
                                best_match = {
                                    "region": m_region,
                                    "digits1": m_digits1,
                                    "char": m_char,
                                    "digits2": m_digits2
                                }
                                
                        region = best_match["region"]
                        char_part = best_match["char"]
                        
                        ocr_typos = {
                            "공기": "경기", "경가": "경기", "갱기": "경기", "겸기": "경기",
                            "서율": "서울", "소울": "서울", "세울": "서울",
                            "인전": "인천", "안천": "인천",
                            "대건": "대전", "대잔": "대전",
                            "부신": "부산", "뷰산": "부산",
                            "우산": "울산", "율산": "울산"
                        }
                        if region in ocr_typos:
                            region = ocr_typos[region]
                            
                        # 흔한 영문 오인식 글자 보정
                        char_typos = {"H": "바", "u": "바", "n": "나", "r": "가", "O": "아", "o": "아", "a": "아", "S": "서", "s": "서"}
                        if char_part in char_typos:
                            char_part = char_typos[char_part]
                            
                        plate_text = f"{region}{best_match['digits1']}{char_part}{best_match['digits2']}"
                    
                    if not plate_text:
                        plate_text = f"미인식: {cleaned_text[-10:]}" if len(cleaned_text) > 10 else f"미인식: {cleaned_text}"
            
            # 이미지에 파란색 사각형 박스 및 라벨 그리기
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (241, 102, 99), 2)  # BGR로 주황/남색 계열
            
            # 라벨 텍스트: 차종 + 신뢰도 + (번호판)
            label = f"{class_name.upper()} {conf:.2f}"
            if plate_text:
                label += f" [{plate_text}]"
                
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            detected_list.append({
                "index": idx + 1,
                "type": class_name,
                "confidence_yolo": float(conf),
                "plate": plate_text
            })
            
    # 가공 완료된 이미지를 base64 스트링으로 반환
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "image": f"data:image/jpeg;base64,{img_base64}",
        "results": detected_list
    }
