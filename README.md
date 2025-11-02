🔐 Securing MQTT Communications in IoT using ECDH and AES-GCM
📘 Overview

This project demonstrates secure IoT data transmission over MQTT by combining:

Elliptic Curve Diffie-Hellman (ECDH) for key exchange

AES-128-GCM for encryption & authentication

The publisher and subscriber securely exchange public keys using ECDH, derive a shared AES key, and then use it to encrypt and decrypt real-time telemetry data.

⚙️ Features

✅ End-to-end encryption using AES-128-GCM
✅ Secure key exchange via ECDH (Elliptic Curve Diffie-Hellman)
✅ Real-time IoT data publishing & subscribing over MQTT
✅ Lightweight, Python-based implementation
✅ Compatible with Mosquitto MQTT broker

🧠 Project Structure
Securing-MQTT-Communications-in-IoT/
│
├── aes_handler.py           # Handles AES and ECDH key exchange logic
├── mqtt_publisher.py        # Publishes encrypted telemetry data
├── mqtt_subscriber.py       # Subscribes and decrypts incoming messages
├── aes_key.bin              # Auto-generated AES key (not shared)
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies

🧩 Dependencies

Install the following Python libraries in your environment:

pip install paho-mqtt cryptography


You must also have a working Mosquitto MQTT broker installed locally.

Download from:
🔗 https://mosquitto.org/download/

🧱 Setting up the Environment
1️⃣ Create & Activate Virtual Environment

If you use Anaconda:

conda create -n sentiment_env python=3.9
conda activate sentiment_env

2️⃣ Navigate to Project Directory
cd C:\Users\<YourUser>\Securing-MQTT-Communications-in-IoT

3️⃣ Install Dependencies
pip install -r requirements.txt


If you don’t have requirements.txt, simply run:

pip install paho-mqtt cryptography

🚀 Running the Project
🖥 Step 1: Start Mosquitto Broker

In one terminal:

mosquitto -v


Keep it running in the background.
You should see:

mosquitto version 2.x starting
Opening ipv4 listen socket on port 1883

🖥 Step 2: Run the Subscriber

In a second terminal:

conda activate sentiment_env
cd C:\Users\<YourUser>\Securing-MQTT-Communications-in-IoT
python mqtt_subscriber.py


You’ll see logs like:

[Subscriber] Connected to broker
[Subscriber] Waiting for ECDH key exchange...
[Subscriber] AES key derived successfully!
[Subscriber] Listening for encrypted messages...

🖥 Step 3: Run the Publisher

In a third terminal:

conda activate sentiment_env
cd C:\Users\<YourUser>\Securing-MQTT-Communications-in-IoT
python mqtt_publisher.py


You’ll see logs like:

[Publisher] Connected to broker
[Publisher] ECDH key exchange completed
[Publisher] Publishing encrypted telemetry data...

🖥 Step 4: Observe Encrypted Data Transmission

The subscriber window will display real-time decrypted messages such as:

[Decrypted] {"patient_id": 101, "heart_rate": 83, "temperature": 36.7, "spo2": 98, "timestamp": "03:43:47"} (Decryption Time: 2.01 ms)

🔐 Encryption Flow Diagram
Publisher (IoT Device)
   │
   ├── Generates ECDH key pair
   │
   ├── Sends public key ───────────▶ Subscriber
   │                               (Receives public key)
   │
   ◀────────────── Receives public key
   │
   ├── Derives shared AES key (ECDH)
   │
   ├── Encrypts telemetry JSON using AES-128-GCM
   │
   └── Publishes encrypted message to MQTT topic
                                   │
Subscriber                         │
   ├── Decrypts message with same AES key
   ├── Validates authentication tag
   └── Displays decrypted JSON data

🧪 Example Output

Publisher:

[Publisher] ECDH exchange done
[Publisher] Sent encrypted telemetry: {"heart_rate": 82, "temperature": 36.7, "spo2": 98}


Subscriber:

[Decrypted] {"patient_id": 101, "heart_rate": 82, "temperature": 36.7, "spo2": 98, "timestamp": "03:42:02"} (Decryption Time: 2.01 ms)

🧰 Technologies Used
Component	Technology
Language	Python 3.9
Messaging Protocol	MQTT
Broker	Eclipse Mosquitto
Encryption	AES-128-GCM
Key Exchange	Elliptic Curve Diffie-Hellman (ECDH)
Libraries	paho-mqtt, cryptography
