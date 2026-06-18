import cv2
import numpy as np
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import os

app = FastAPI(title="CCTV Monitoring Portfolio")

# 템플릿 디렉토리 설정
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

def generate_frames():
    """
    웹캠에서 프레임을 읽어와 JPEG로 인코딩한 후 yield하는 제너레이터 함수.
    C310 카메라를 연결할 때는 VideoCapture(0)의 0을 
    적절한 카메라 인덱스 번호나 RTSP 주소로 변경하시면 됩니다.
    """
    # 기본 카메라(0번) 사용 시도
    camera = cv2.VideoCapture(0)
    
    # 카메라가 연결되어 있지 않거나 열리지 않는 경우 더미(Placeholder) 영상 스트리밍
    if not camera.isOpened() or not camera.read()[0]:
        print("⚠️ 카메라를 찾을 수 없습니다. 테스트용 더미 화면을 송출합니다.")
        while True:
            # 검은색 빈 화면 생성 (가로 640, 세로 480)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 안내 문구 삽입
            cv2.putText(frame, "No Camera Detected", (130, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            cv2.putText(frame, "Waiting for C310...", (180, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # 이미지를 JPEG 포맷으로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # 과부하 방지 (1초에 1프레임 송출)
            time.sleep(1)
            
    # 카메라가 정상적으로 연결된 경우 실시간 스트리밍
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    camera.release()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """관제 대시보드 메인 페이지 렌더링"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    """웹캠 영상 실시간 스트리밍 엔드포인트"""
    # 브라우저가 연속된 이미지를 영상으로 인식하도록 미디어 타입 지정
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
