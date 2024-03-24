# Base image with NVIDIA CUDA support
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# Install Python, pip, and necessary packages in a single RUN command to reduce layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    dos2unix \
    python3.9 \
    python3-pip \
    && ln -s /usr/bin/python3.9 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --upgrade pip

# Copy the requirements file and install Python dependencies
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Copy source code and scripts into the container
COPY src /opt/src
COPY entry_point.sh fix_line_endings.sh /opt/
RUN chmod +x /opt/entry_point.sh /opt/fix_line_endings.sh \
    && /opt/fix_line_endings.sh "/opt/src" \
    && /opt/fix_line_endings.sh "/opt/entry_point.sh"

# Set working directory
WORKDIR /opt/src

# Set environment variables
ENV MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    PATH="/opt/app:${PATH}"

# Prepare directory for lightning logs with broad permissions
RUN mkdir -p /opt/src/lightning_logs && chmod -R 777 /opt/src/lightning_logs

# Run as a non-root user for better security
USER 1000

# Set the container's entry point
ENTRYPOINT ["/opt/entry_point.sh"]
