FROM python:3.10-slim

# 필요한 시스템 패키지 설치 (OpenCV 및 EasyOCR 등 이미지 처리에 필요)
# apt-get 에러(exit code 100) 방지를 위해 미러 및 패키지 간소화
RUN apt-get clean && apt-get update -o Acquire::Retries=3 --fix-missing && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 패키지 목록 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# EasyOCR은 requirements.txt에 누락되어 있을 수 있으므로 명시적으로 설치
RUN pip install --no-cache-dir easyocr

# 전체 소스 코드 복사
COPY . .

# 컨테이너 외부로 노출할 포트
EXPOSE 9000

# 서버 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
