import os
import base64
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class AESHandler:
    def __init__(self):
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.shared_key = None
        self.aes_key = None
        self.aesgcm = None

    def get_public_key_bytes(self):
        public_key = self.private_key.public_key()
        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()

    def derive_shared_key(self, peer_public_bytes):
        peer_public_key = serialization.load_pem_public_key(peer_public_bytes.encode())
        self.shared_key = self.private_key.exchange(ec.ECDH(), peer_public_key)
        self.aes_key = HKDF(
            algorithm=hashes.SHA256(),
            length=16,
            salt=None,
            info=b"mqtt-ecdh-key",
        ).derive(self.shared_key)
        self.aesgcm = AESGCM(self.aes_key)
        return self.aes_key

    def encrypt(self, plaintext):
        nonce = os.urandom(12)
        ciphertext = self.aesgcm.encrypt(nonce, plaintext.encode(), None)
        return base64.b64encode(nonce + ciphertext).decode()

    def decrypt(self, b64_ciphertext):
        raw = base64.b64decode(b64_ciphertext)
        nonce, ciphertext = raw[:12], raw[12:]
        plaintext = self.aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode()
