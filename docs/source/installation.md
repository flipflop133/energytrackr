# 🛠 Installation Guide

This document describes how to install and configure the Energy Pipeline for detecting energy regressions in software repositories.

---

## 📦 Requirements

Make sure your system meets the following prerequisites:

### ✅ System Tools

| Tool             | Description                                |
| ---------------- | ------------------------------------------ |
| Python 3.10+     | For running the pipeline                   |
| Git              | Required for repository handling           |
| Bash             | For scripting and system setup             |
| Intel RAPL       | Used to measure energy consumption         |
| `perf`           | Used for performance counters verification |
| Optional: Docker | For containerized execution                |

> 💡 **Note**: The pipeline is designed for Linux systems with Intel CPUs. It uses `/sys/class/powercap/intel-rapl` for energy measurement.

---

## 🐍 Python Environment

It's highly recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install required Python packages:

```bash
pip install -r requirements.txt
```

If you also plan to edit documentation:

```bash
pip install -r docs/requirements.txt
```

> Create this file if it doesn't exist:

```txt
# docs/requirements.txt
sphinx
myst-parser
furo
```

---

## 🧰 System Setup

The script `system_setup.sh` configures your system for stable and accurate energy measurement.

### First Time Setup (once per machine)

```bash
sudo ./system_setup.sh first-setup
```

Then **reboot** your machine.

### Pre-run Setup (every time before running pipeline)

```bash
sudo ./system_setup.sh setup
```

> ⚠️ This puts your system in a minimal state (disables services, sets performance governor, etc.).

### Revert Setup (if needed)

```bash
sudo ./system_setup.sh revert-setup       # Undo setup (no reboot needed)
sudo ./system_setup.sh revert-first-setup # Undo first-time setup (requires reboot)
```

---

## 🔐 Permissions

Make sure you can access performance counters and RAPL:

```bash
sudo sysctl -w kernel.perf_event_paranoid=-1
sudo chmod -R a+rw /sys/class/powercap/intel-rapl
```

These are handled by the setup script, but double-check if you encounter permission errors.

---

## 🐳 Optional: Docker Setup

You can run the pipeline inside a privileged container (useful on remote servers or CI):

### Build the Docker image

```bash
docker buildx build -t pipeline .
```

### Run the container

```bash
docker run -it --privileged pipeline:latest
```

To mount your configuration and code:

```bash
docker run -it --privileged \
  -v $(pwd)/configs:/configs \
  -v $(pwd)/projects:/projects \
  pipeline:latest
```

---

## 💡 Recommended Setup for Accurate Measurement

| Task                              | Status        |
| --------------------------------- | ------------- |
| Run without GUI (in tty)          | ✅ Required    |
| Close unnecessary background apps | ✅ Required    |
| Disable screen auto-brightness    | ✅ Required    |
| Set CPU governor to `performance` | ✅ Required    |
| Disable Wi-Fi and Bluetooth       | ✅ Optional    |
| Use Ethernet                      | ✅ Optional    |
| Ensure system is plugged in       | ✅ Required    |
| Ensure battery is fully charged   | ✅ Recommended |

---

## 🧪 Validate Setup

Run this before starting:

```bash
python main.py stability-test
```

This checks if your system is in a stable state for measurement (e.g., temperature, `perf` access).

---

## ✅ Installation Recap

```bash
# Clone repo
git clone https://github.com/flipflop133/energy_analyzer
cd energy-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Run first-time system setup
sudo ./system_setup.sh first-setup
reboot

# Setup before each run
sudo ./system_setup.sh setup

# Validate system
python main.py stability-test
```

---

Still stuck? Check the [FAQ](faq.md) or open an issue on GitHub.
