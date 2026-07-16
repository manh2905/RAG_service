FROM python:3.11-slim

WORKDIR /app

# Cài đặt dependencies hệ thống (nếu cần thiết cho các thư viện NLP/Parsing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port FastAPI
EXPOSE 8000

# Chạy app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
