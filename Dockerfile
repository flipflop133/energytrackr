FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y \
    python3.13 \
    python3-pip \
    python3-venv \
    git \
    sudo \
    linux-tools-common \
    linux-tools-generic \
    linux-tools-$(uname -r) && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x measure_energy.sh

CMD ["python3", "main.py", "measure", "conf.json"]
