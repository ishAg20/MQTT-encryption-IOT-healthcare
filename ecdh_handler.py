from Crypto.PublicKey import ECC
from Crypto.Hash import SHA256
from Crypto.Protocol.KDF import HKDF

class ECDHHandler:
    def __init__(self, curve_name="P-256"):
        # Generate a private key for this participant
        self.key = ECC.generate(curve=curve_name)
        self.shared_key = None

    def get_public_key(self):
        """Return the public key in exportable PEM format"""
        return self.key.public_key().export_key(format="PEM")

    def generate_shared_key(self, peer_public_pem):
        """Generate a shared secret using the peer's public key"""
        peer_key = ECC.import_key(peer_public_pem)
        shared_secret = (peer_key.pointQ * self.key.d).x.to_bytes(32, byteorder="big")

        # Derive a symmetric key from the shared secret using HKDF
        self.shared_key = HKDF(
            master=shared_secret,
            key_len=32,
            salt=None,
            hashmod=SHA256,
            num_keys=1
        )
        return self.shared_key
