FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "streamlit run 'Predictor de Voto.py' --server.port=$PORT --server.address=0.0.0.0"]
