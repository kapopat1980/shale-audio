"""Check SHALE's Ascon against the official NIST SP 800-232 KAT files."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shale import ascon

D = os.path.dirname(__file__)


def parse(path):
    recs, cur = [], {}
    for line in open(path):
        line = line.strip()
        if not line:
            if cur:
                recs.append(cur); cur = {}
            continue
        k, _, v = line.partition("=")
        if k.strip() != "Count":
            cur[k.strip()] = bytes.fromhex(v.strip())
    if cur:
        recs.append(cur)
    return recs


def main():
    n = bad = 0
    for r in parse(os.path.join(D, "LWC_AEAD_KAT_128_128.txt")):
        ct = ascon.encrypt(r["Key"], r["Nonce"], r["AD"], r["PT"])
        pt = ascon.decrypt(r["Key"], r["Nonce"], r["AD"], r["CT"])
        n += 1
        if ct != r["CT"] or pt != r["PT"]:
            bad += 1
        # tamper: flip one bit of the tag
        t = bytearray(r["CT"]); t[-1] ^= 1
        if ascon.decrypt(r["Key"], r["Nonce"], r["AD"], bytes(t)) is not None:
            bad += 1
    print(f"AEAD KAT: {n} vectors, {bad} failures")
    n2 = bad2 = 0
    for r in parse(os.path.join(D, "LWC_XOF_KAT_128_512.txt")):
        n2 += 1
        if ascon.xof(r["Msg"], len(r["MD"])) != r["MD"]:
            bad2 += 1
    print(f"XOF KAT: {n2} vectors, {bad2} failures")
    n3 = bad3 = 0
    for r in parse(os.path.join(D, "LWC_CXOF_KAT_128_512.txt")):
        n3 += 1
        if ascon.cxof(r["Msg"], len(r["MD"]), r["Z"]) != r["MD"]:
            bad3 += 1
    print(f"CXOF KAT: {n3} vectors, {bad3} failures")
    return bad + bad2 + bad3


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
