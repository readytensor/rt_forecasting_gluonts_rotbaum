# Use a specific version of nvidia/cuda base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 as builder

# Install dependencies, python, pip, and add symbolic link to python3 in a single RUN to reduce layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    dos2unix \
    python3.9 \
    python3-pip \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* 

# Upgrade pip and install requirements in a single layer
COPY ./requirements.txt .
RUN pip3 --no-cache-dir install --upgrade pip && \
    pip3 --no-cache-dir install -r requirements.txt

# Copy source code and entry point script
COPY src /opt/src
COPY ./entry_point.sh /opt/
RUN chmod å+x /opt/entry_point.sh \
    && mkdir -p /opt/src/lightning_logs && chmod -R 777 /opt/src/lightning_logs

# Set working directory
WORKDIR /opt/src

# Set environment variables
ENV MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    PATH="/opt/app:${PATH}"

# Set non-root user and entrypoint
USER 1000
ENTRYPOINT ["/opt/entry_point.sh"]
