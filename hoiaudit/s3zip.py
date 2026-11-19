"""Read the central directory / single entries of a zip on S3 via range requests.

A key may be a list of keys making up one split archive. shift > 0 means bytes are missing from the front (a
truncated archive); entries living there cannot be read."""
import boto3, struct, sys, collections
s3 = boto3.client("s3")
_SIZES = {}


def _size(B, K):
    if (B, K) not in _SIZES:
        _SIZES[(B, K)] = s3.head_object(Bucket=B, Key=K)["ContentLength"]
    return _SIZES[(B, K)]


def rng(B, K, a, b):
    """bytes [a, b) of K. K may be a list of keys forming one split archive (.part00, .part01, ...), read as
    their concatenation."""
    if isinstance(K, str):
        return s3.get_object(Bucket=B, Key=K, Range=f"bytes={a}-{b-1}")["Body"].read()
    out, start = [], 0
    for k in K:
        n = _size(B, k)
        lo, hi = max(a, start), min(b, start + n)
        if lo < hi:
            out.append(rng(B, k, lo - start, hi - start))
        start += n
    return b"".join(out)


def central_dir(B, K):
    size = _size(B, K) if isinstance(K, str) else sum(_size(B, k) for k in K)
    tail = rng(B, K, max(0, size - 65536), size)
    i = tail.rfind(b"PK\x05\x06"); j = tail.rfind(b"PK\x06\x06")
    if j >= 0:
        cd_size, cd_off = struct.unpack("<QQ", tail[j+40:j+56])
        eocd64_pos_in_file = size - len(tail) + j
    else:
        cd_size, cd_off = struct.unpack("<II", tail[i+12:i+20]); eocd64_pos_in_file = size - len(tail) + i
    cd_pos_in_file = eocd64_pos_in_file - cd_size
    shift = cd_off - cd_pos_in_file          # bytes missing from the front (logical offset - file offset)
    cd = rng(B, K, cd_pos_in_file, cd_pos_in_file + cd_size)
    ents, p = [], 0
    while p < len(cd) and cd[p:p+4] == b"PK\x01\x02":
        meth, = struct.unpack("<H", cd[p+10:p+12]); csz, usz = struct.unpack("<II", cd[p+20:p+28])
        nl, el, cl = struct.unpack("<HHH", cd[p+28:p+34]); lho, = struct.unpack("<I", cd[p+42:p+46])
        name = cd[p+46:p+46+nl].decode("utf8", "replace"); ex = cd[p+46+nl:p+46+nl+el]; q = 0
        while q + 4 <= len(ex):
            hid, hl = struct.unpack("<HH", ex[q:q+4]); d = ex[q+4:q+4+hl]; r = 0
            if hid == 1:
                if usz == 0xFFFFFFFF: usz, = struct.unpack("<Q", d[r:r+8]); r += 8
                if csz == 0xFFFFFFFF: csz, = struct.unpack("<Q", d[r:r+8]); r += 8
                if lho == 0xFFFFFFFF: lho, = struct.unpack("<Q", d[r:r+8]); r += 8
            q += 4 + hl
        ents.append(dict(name=name, method=meth, csize=csz, usize=usz, offset=lho))
        p += 46 + nl + el + cl
    return dict(size=size, shift=shift, entries=ents)
if __name__ == "__main__":
    B, K = sys.argv[1], sys.argv[2]
    z = central_dir(B, K); E = z["entries"]
    avail = [e for e in E if e["offset"] - z["shift"] >= 0]
    print(f"{K}: file {z['size']/1e9:.2f} GB, missing-front {z['shift']/1e9:.2f} GB, entries {len(E)}, "
          f"present {len(avail)} ({sum(e['csize'] for e in avail)/1e9:.2f} GB)")
    tops = collections.Counter("/".join(e["name"].split("/")[:2]) for e in E)
    for k, v in list(tops.items())[:8]: print("   ", v, k)
    for e in E[:6] + E[-3:]: print("    ", e["name"], e["usize"])
    exts = collections.Counter(e["name"].rsplit(".", 1)[-1] for e in E if not e["name"].endswith("/"))
    print("    exts:", exts.most_common(8))

import zlib
def read_entry(B, K, e, shift=0):
    a = e["offset"] - shift
    h = rng(B, K, a, a + 30); nl, el = struct.unpack("<HH", h[26:30])
    d = rng(B, K, a + 30 + nl + el, a + 30 + nl + el + e["csize"])
    return d if e["method"] == 0 else zlib.decompressobj(-15).decompress(d)
