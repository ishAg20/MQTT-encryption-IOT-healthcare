# MQTT Encryption for IoT Healthcare (ECDH + AES-GCM)

## Overview

This repo demonstrates **secure telemetry transmission** for IoT healthcare devices using:

- **Elliptic Curve Diffie-Hellman (ECDH):** Secure session key exchange
- **AES-128/256-GCM:** Authenticated encryption of patient data
- **MQTT (Mosquitto):** Real-time publish/subscribe messaging

Publisher and Subscriber negotiate ECDH public keys, exchange, and derive a shared AES key for secure transmission of healthcare data.

## Features

- End-to-end encryption over MQTT using AES-GCM
- Secure ECDH key exchange; Supports session key rotation
- Benchmarking: latency, throughput, payload & resource metrics
- Configurable telemetry dataset for reproducible results
- Python implementation; works with Mosquitto MQTT broker

## Folder Structure

```
├─ aes_handler.py # AES encryption & ECDH key negotiation
├─ ecdh_handler.py # Pure ECDH key exchange handler
├─ mqtt_publisher.py # Publishes encrypted patient telemetry
├─ mqtt_subscriber.py # Receives, decrypts, and logs messages
├─ dataset.py # Telemetry dataset management & cleaning
├─ performance.py # Benchmarking, metrics & plots
├─ config.yaml # Central project configuration
├─ requirements.txt # Python dependencies
├─ yourdataset.csv
├─ README.md # Project documentation
```

## Requirements

Python packages: `pip install -r requirements.txt`

Also install [Mosquitto MQTT broker](https://mosquitto.org/download/)

## Quick Start

### 1. Create & Activate Environment (Anaconda Recommended)

```
conda create -n iot_secure_env python=3.9
conda activate iot_secure_env
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run Mosquitto broker

```
mosquitto -v
```

### 4. Load Dataset & Run Subscriber + Publisher (Three terminals)

Terminal 1 (Broker):
`mosquitto -v`

Terminal 2 (Subscriber):
`python mqtt_subscriber.py`

Terminal 3 (Publisher):
`python mqtt_publisher.py yourdataset.csv`

## Benchmarking

Run enhanced benchmarking and view results:
`python performance.py`

Metrics evaluated:

- Encryption/Decryption Latency (ms)
- Throughput (messages/sec)
- Payload Size Overhead (%)
- CPU/Memory Utilization
- Security-Performance Tradeoffs

Visualizations saved as `aes_performance_analysis.png`.

## Dataset Management (`dataset.py`)

- Load healthcare/IoT dataset from config.
- Filter features, perform sampling.
- Data cleaning, summary stats and export capability.
- Add new strategies for outlier removal and synthetic data.

## Troubleshooting

- Ensure `localhost:1883` is open for MQTT traffic
- Validate dataset format in `config.yaml`
- See logs for ECDH and AES errors during handshake
