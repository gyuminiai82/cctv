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
    Tapo C310 등의 IP 카메라에서 RTSP 스트림을 읽어와 JPEG로 변환 후 전송합니다.
    현재는 C310이 연결되지 않은 상태이므로, 인터넷상의 오픈된 테스트용 RTSP 주소를 기본값으로 사용합니다.
    """
    # 테스트용 오픈 RTSP (유명한 Big Buck Bunny 애니메이션 테스트 스트림)
    # 실제 C310 사용 시: "rtsp://아이디:비밀번호@192.168.x.x:554/stream1" 로 변경
    RTSP_URL = os.getenv("RTSP_URL", "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4")
    
    print(f"🎥 RTSP 스트림 연결 시도 중: {RTSP_URL}")
    camera = cv2.VideoCapture(RTSP_URL)
    
    # RTSP 스트림 접속 실패 또는 지연 시 더미 영상 송출
    if not camera.isOpened():
        print("⚠️ RTSP 서버 연결에 실패했습니다. 테스트용 더미 화면을 송출합니다.")
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "RTSP Connection Failed", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, "Check Network or Camera", (120, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)
            
    # 정상적으로 RTSP 스트림을 받아오는 경우
    while True:
        success, frame = camera.read()
        if not success:
            # 네트워크 불안정으로 프레임이 끊기면 연결을 재시도해야 할 수 있습니다. (간이 처리)
            print("⚠️ RTSP 프레임 수신 실패. 1초 대기 후 계속 진행...")
            time.sleep(1)
            continue
            
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
    # 브라우저가 연속된 이미지를 영상으로 인식하도록 미디어 타입 지정
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
