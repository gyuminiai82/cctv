import cv2
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
    # 기본 카메라(0번) 사용
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # OpenCV 이미지를 JPEG 포맷으로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # multipart/x-mixed-replace 형식에 맞게 byte stream 구성
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
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
