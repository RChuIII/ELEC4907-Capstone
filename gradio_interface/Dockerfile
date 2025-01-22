FROM python:3.12.2-slim

WORKDIR /app
COPY /app .
RUN pip install --no-cache-dir gradio
RUN pip install --r requirements.txt
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
