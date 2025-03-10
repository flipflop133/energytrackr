FROM ubuntu:24.04

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    autoconf \
    automake \
    autopoint \
    build-essential \
    ccache \
    gcc \
    git \
    libtool \
    linux-tools-common \
    linux-tools-generic \
    linux-tools-6.8.0-40-generic \
    m4 \
    make \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Create a virtual environment
# Create and activate the virtual environment, then install dependencies
RUN python3 -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x measure_energy.sh

# Ensure Python uses the virtual environment
ENV PATH="/app/venv/bin:$PATH"

#CMD ["python3", "main.py", "measure", "conf.json"]
