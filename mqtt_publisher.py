import json
import time
import paho.mqtt.client as mqtt
from aes_handler import AESHandler
import pandas as pd
import sys

TOPIC_KEY_EXCHANGE = "icu/key_exchange"
TOPIC_DATA = "icu/iot/telemetry"
CSV_FILENAME = sys.argv[1] if len(sys.argv) > 1 else "patientMonitoring.csv"

handler = AESHandler()
client = mqtt.Client()

peer_pubkey_received = False

def on_connect(client, userdata, flags, rc):
    print("[Publisher] Connected to broker.")
    client.subscribe(TOPIC_KEY_EXCHANGE)
    client.publish(TOPIC_KEY_EXCHANGE, json.dumps({
        "role": "publisher",
        "pubkey": handler.get_public_key_bytes()
    }))
    print("[Publisher] Sent ECDH public key.")

def on_message(client, userdata, msg):
    global peer_pubkey_received
    payload = json.loads(msg.payload.decode())
    if payload.get("role") == "subscriber" and not peer_pubkey_received:
        handler.derive_shared_key(payload["pubkey"])
        peer_pubkey_received = True
        print("[Publisher] Shared AES key derived via ECDH handshake ✅")

client.on_connect = on_connect
client.on_message = on_message

client.connect("localhost", 1883)
client.loop_start()

print("[Publisher] Waiting for key exchange...")

# Wait until key exchange is done
while not peer_pubkey_received:
    time.sleep(1)

print("[Publisher] Ready. Publishing encrypted telemetry data...")

# --- Load data from CSV ---
try:
    df = pd.read_csv(CSV_FILENAME)
    print(f"[Publisher] Loaded {len(df)} rows from {CSV_FILENAME}")
except Exception as e:
    print(f"[Publisher] ERROR loading CSV: {e}")
    sys.exit(1)

# --- Publish each row ---
for _, row in df.iterrows():
    data = row.to_dict()
    enc_data = handler.encrypt(json.dumps(data))
    client.publish(TOPIC_DATA, enc_data)
    print("[Publisher] Sent encrypted:", data)
    time.sleep(3)  

print("[Publisher] Done sending all rows.")