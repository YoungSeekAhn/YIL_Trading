FROM python:3.11-slim

WORKDIR /app

# (선택) 시간대/로케일 필요하면 추가 가능. 일단 최소 구성.

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# 컨테이너 실행 시 SP_YIL.py 실행
CMD ["python", "SP_YIL.py"]
