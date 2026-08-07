FROM python:3.11-slim

WORKDIR /app

RUN useradd -m appuser

# 🔥 Upgrade pip + use faster install
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy only required files
COPY main.py .
COPY best_titanic_model.pkl .
COPY label_encoders.pkl .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]