import json
import time
import paho.mqtt.client as mqtt
from aes_handler import AESHandler

TOPIC_KEY_EXCHANGE = "icu/key_exchange"
TOPIC_DATA = "icu/iot/telemetry"

handler = AESHandler()
client = mqtt.Client()

key_derived = False

def on_connect(client, userdata, flags, rc):
    print("[Subscriber] Connected to broker.")
    client.subscribe(TOPIC_KEY_EXCHANGE)
    client.subscribe(TOPIC_DATA)

def on_message(client, userdata, msg):
    global key_derived
    topic = msg.topic
    payload = msg.payload.decode()

    if topic == TOPIC_KEY_EXCHANGE:
        data = json.loads(payload)
        if data.get("role") == "publisher" and not key_derived:
            handler.derive_shared_key(data["pubkey"])
            client.publish(TOPIC_KEY_EXCHANGE, json.dumps({
                "role": "subscriber",
                "pubkey": handler.get_public_key_bytes()
            }))
            key_derived = True
            print("[Subscriber] ECDH handshake complete ✅")
            print("[Subscriber] AES key derived securely.")
    elif topic == TOPIC_DATA and key_derived:
        try:
            plaintext = handler.decrypt(payload)
            print("[Decrypted]", plaintext)
        except Exception as e:
            print("[Error decrypting]", str(e))

client.on_connect = on_connect
client.on_message = on_message

client.connect("localhost", 1883)
client.loop_forever()
