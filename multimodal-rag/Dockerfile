FROM python:3.11-slim

RUN pip install --upgrade pip \
    && pip install awslambdaric

# Install poppler-utils
RUN apt-get update && \
    apt-get install -y pandoc curl poppler-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /var/task

RUN curl -LO https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz && \
    tar -xzf tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz && \
    mv tectonic /usr/local/bin/tectonic && \
    chmod +x /usr/local/bin/tectonic && \
    rm tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz

COPY requirements.txt .
COPY pyproject.toml .
COPY src/ ./src/

RUN pip install -r requirements.txt

ENV PYTHONPATH=/var/task/src

CMD ["python", "-m", "awslambdaric", "mm_rag.app.main.handler"]

