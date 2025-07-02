# Building Stage #
FROM python:3.12-bookworm AS builder

# Installing Tectonic (LaTeX Engine[pandoc dependency]) and removing the cache
# Tectonic is installed here because it's a single binary that can be copied.
WORKDIR /tmp
RUN curl -LO https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz && \
    tar -xzf tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz && \
    mv tectonic /usr/local/bin/tectonic && \
    chmod +x /usr/local/bin/tectonic && \
    rm tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz

# Download the latest installer, install it and then remove it
# uv is used to manage Python dependencies efficiently.
WORKDIR /var/task
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod -R 655 /install.sh && /install.sh && rm /install.sh

COPY . .

# Set up the UV environment path correctly for the builder stage
ENV PATH="/root/.local/bin:${PATH}"

# Install the Python dependencies in a virtual environment
# This virtual environment will be copied to the production stage.
RUN uv sync

# Production Stage #
# Using a slim base image for a smaller final image size.
FROM python:3.12-slim AS production

# Create a non-root user for security best practices.
RUN useradd --create-home appuser

WORKDIR /var/task/

# Install apt packages required by the application in the production stage.
# This ensures all necessary system libraries and dependencies for pandoc and poppler-utils
# are correctly installed and configured within the final image.
# --no-install-recommends keeps the image size down.
# apt-get clean and rm -rf /var/lib/apt/lists/* remove cached package files.
RUN apt-get update && apt-get install --no-install-recommends -y \
      pandoc poppler-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the Tectonic binary from the builder stage.
# Since it's a self-contained executable, direct copying is efficient.
COPY --from=builder /usr/local/bin/tectonic /usr/local/bin/tectonic

# Copy the created virtual environment from the builder stage.
# This contains all your Python dependencies.
COPY --from=builder /var/task/.venv .venv

# Copy the application code from the builder stage.
COPY --from=builder /var/task/ ./

# Set up environment variables for production.
# Ensure the virtual environment's bin directory is in the PATH.
ENV PATH="/var/task/.venv/bin:$PATH"
ENV PYTHONPATH=/var/task

# Change ownership of the application directory to the non-root user.
RUN chown -R appuser:appuser /var/task

# Switch to the least-privileged user.
USER appuser

# Define the command to run the application when the container starts.
# This uses the AWS Lambda Runtime Interface Client (RIC) to handle Lambda invocations.
CMD ["python3", "-m", "awslambdaric", "mm_rag.api.main.handler"]
