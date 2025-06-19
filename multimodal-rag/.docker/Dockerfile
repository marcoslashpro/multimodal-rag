# Building Stage #
FROM python:3.12-bookworm AS builder

# Installing apt packages
RUN apt-get update && apt-get install --no-install-recommends -y \
      pandoc poppler-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /var/task

# Installing Tectonic (LaTeX Engine[pandoc dependency]) and removing the cache
RUN curl -LO https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz && \
    tar -xzf tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz && \
    mv tectonic /usr/local/bin/tectonic && \
    chmod +x /usr/local/bin/tectonic && \
    rm tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz

# Download the latest installer, install it and then remove it
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod -R 655 /install.sh && /install.sh && rm /install.sh

COPY . .

# Set up the UV environment path correctly
ENV PATH="/root/.local/bin:${PATH}"

# Install the dependencies in the pyproject.toml file
RUN uv sync

# Production Stage #
FROM python:3.12-slim AS production

RUN useradd --create-home appuser

WORKDIR /var/task/

# Copy the created venv from the builder stage
COPY  --from=builder /var/task/.venv .venv

COPY --from=builder /var/task/ ./

# Set up environment variables for production
ENV PATH="/src/.venv/bin:$PATH"

# Use the least-priviledged user
RUN chown -R appuser:appuser /var/task
USER appuser

# Set the right path for the system to operate correctly
ENV PYTHONPATH=/src
ENV PATH="/var/task/.venv/bin:$PATH"

CMD ["python3", "-m", "awslambdaric", "mm_rag.app.main.handler"]
