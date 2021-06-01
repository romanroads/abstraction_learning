import hashlib

try:
    from Crypto import Random
    from Crypto.Cipher import AES
except:
    raise Exception('Install Crypto! \n pip install pycrypto')


class AESCipher(object):
    def __init__(self, _key):
        self.bs = 16  # bytes, 128 bits
        print('Using block size = %s [bytes]' % self.bs)
        self.key = hashlib.sha256(_key.encode()).digest()
        print('Hash of key="%s" is "%s"' % (_key, self.key))

    def encrypt(self, raw):
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        print('Iv: "%s"' % iv)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return iv + cipher.encrypt(raw)

    def decrypt(self, enc):
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:]))  # .decode('utf-8')

    def _pad(self, s):
        return s + str.encode((self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs))\

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s) - 1:])]
