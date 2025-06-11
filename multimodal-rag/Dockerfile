FROM python:3.11-slim

RUN pip install --upgrade pip \
    && pip install awslambdaric

# Install poppler-utils
RUN apt-get update && \
    apt-get install -y poppler-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /var/task

COPY requirements.txt .
COPY pyproject.toml .
COPY src/ ./src/

RUN pip install -r requirements.txt

ENV PYTHONPATH=/var/task/src

CMD ["python", "-m", "awslambdaric", "mm_rag.app.main.handler"]

