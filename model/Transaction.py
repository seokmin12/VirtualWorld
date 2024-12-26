from hashlib import sha256
import rsa


def apply_sha256(input):
    try:
        digest = sha256()
        digest.update(input.encode('utf-8'))
        hash_bytes = digest.digest()
        hex_string = ''.join(f'{b:02x}' for b in hash_bytes)
        return hex_string
    except Exception as e:
        print(e)
        raise RuntimeError(e)


class Transaction:
    def __init__(self, sender_account_num: str, recipient_account_num: str, amount: int, previous_hash: str):
        self.sender = sender_account_num
        self.recipient = recipient_account_num
        self.amount = amount
        self.hash: str = self.compute_hash()
        self.previous_hash = previous_hash
        self.signature: [bytes] = None

    def compute_hash(self):
        calculatedHash = apply_sha256(
            str(self.sender) + str(self.recipient) + str(self.amount)
        )

        return calculatedHash

    def generateSignature(self, private_key):
        data: str = self.sender + self.recipient + self.hash
        try:
            signature = rsa.sign(data.encode(), private_key, 'SHA-256')
            self.signature = signature
        except Exception as e:
            print(e)

    def verifySignature(self, public_key) -> bool:
        data: str = self.sender + self.recipient + self.hash
        try:
            rsa.verify(data.encode('utf-8'), self.signature, public_key)
            return True
        except (ValueError, TypeError) as e:
            print(e)
        return False

    def getSenderAccountNum(self) -> str:
        return self.sender

    def getRecipientAccountNum(self) -> str:
        return self.recipient

    def getHash(self) -> str:
        return self.hash

    def getPreviousHash(self) -> str:
        return self.previous_hash

    def setHash(self, hash: str):
        self.hash = hash

    def setPreviousHash(self, previous_hash: str):
        self.previous_hash = previous_hash
