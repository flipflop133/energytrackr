FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    sudo \
    linux-tools-common \
    linux-tools-generic \
    linux-tools-6.8.0-40-generic \
    build-essential \
    m4 \
    libtool \
    ccache \
    pkg-config \
    autopoint \
    build-essential \
    autoconf \
    automake \
    gcc \
    make

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x measure_energy.sh

#CMD ["python3", "main.py", "measure", "conf.json"]
