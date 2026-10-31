"""Versioned, bounded storage for sorted sparse-selection sets.

IDs are logical positions, never vocabulary IDs or physical KV addresses.
Decoding preserves sets; the original top-k traversal order is not retained.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

MAX_CHUNK_BYTES = 64 * 1024 * 1024
MAX_DECODE_ELEMENTS = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_CHUNKS = 100_000


@dataclass(frozen=True)
class CacheIdentity:
    model_fingerprint: str
    prefix_digest: str
    selector_fingerprint: str
    position_mask_fingerprint: str

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip() or len(value) > 4096:
                raise ValueError(f"{name} must be a nonempty string of at most 4096 characters")


@dataclass(frozen=True)
class EncodedChunk:
    codec: str
    rows: int
    k: int
    universe: int
    dictionary_size: int
    payload: bytes


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"invalid {name}: expected integer in [{minimum}, {maximum}]")
    return value


def _shape(rows, k, universe):
    _integer(universe, "universe", 1, 65536)
    _integer(rows, "rows", 1, MAX_DECODE_ELEMENTS)
    _integer(k, "k", 1, universe)
    if rows * k > MAX_DECODE_ELEMENTS:
        raise ValueError("decoded chunk exceeds element limit")


def _payload_size(codec, rows, k, universe, dictionary_size):
    _shape(rows, k, universe)
    if codec == "raw_u16":
        _integer(dictionary_size, "dictionary_size", 0, 0)
        size = rows * k * 2
    elif codec == "union_bitmap":
        _integer(dictionary_size, "dictionary_size", k, universe)
        size = 2 * dictionary_size + rows * ((dictionary_size + 7) // 8)
    else:
        raise ValueError(f"unknown codec: {codec!r}")
    if size > MAX_CHUNK_BYTES:
        raise ValueError("encoded chunk exceeds byte limit")
    return size


def encode_chunk(indices: np.ndarray, universe=65536, codec="auto") -> EncodedChunk:
    """Sort each row and encode losslessly; auto falls back to raw on expansion."""
    if codec not in ("auto", "raw_u16", "union_bitmap"):
        raise ValueError(f"unknown codec: {codec!r}")
    values = np.asarray(indices)
    if values.ndim != 2 or values.dtype.kind not in "iu":
        raise ValueError("indices must be a two-dimensional integer array")
    rows, k = map(int, values.shape)
    _shape(rows, k, universe)
    if np.any(values < 0) or np.any(values >= universe):
        raise ValueError("selection ID is outside the declared universe")
    ordered = np.sort(values, axis=1)
    if k > 1 and np.any(ordered[:, 1:] == ordered[:, :-1]):
        raise ValueError("duplicate selection IDs in a row")
    ordered = ordered.astype("<u2", copy=False)
    raw = EncodedChunk("raw_u16", rows, k, universe, 0, ordered.tobytes())
    if codec == "raw_u16":
        return raw
    dictionary = np.unique(ordered)
    m = int(dictionary.size)
    size = 2 * m + rows * ((m + 7) // 8)
    if codec == "auto" and (size >= len(raw.payload) or size > MAX_CHUNK_BYTES):
        return raw
    _payload_size("union_bitmap", rows, k, universe, m)
    locations = np.searchsorted(dictionary, ordered)
    bitmap = np.zeros((rows, (m + 7) // 8), dtype=np.uint8)
    np.bitwise_or.at(bitmap, (np.arange(rows)[:, None], locations // 8),
                     (1 << (locations % 8)).astype(np.uint8))
    return EncodedChunk("union_bitmap", rows, k, universe, m,
                        dictionary.astype("<u2", copy=False).tobytes() + bitmap.tobytes())


def decode_chunk(chunk: EncodedChunk) -> np.ndarray:
    """Validate a packet and return its sorted rows as an owned uint16 array."""
    size = _payload_size(chunk.codec, chunk.rows, chunk.k, chunk.universe,
                         chunk.dictionary_size)
    if not isinstance(chunk.payload, bytes) or len(chunk.payload) != size:
        raise ValueError("payload type or length does not match metadata")
    if chunk.codec == "raw_u16":
        out = np.frombuffer(chunk.payload, dtype="<u2").reshape(chunk.rows, chunk.k)
        if np.any(out >= chunk.universe) or np.any(out[:, 1:] <= out[:, :-1]):
            raise ValueError("raw rows must contain sorted unique in-range IDs")
        return out.astype(np.uint16, copy=True)
    m = chunk.dictionary_size
    dictionary = np.frombuffer(chunk.payload, dtype="<u2", count=m)
    if np.any(dictionary >= chunk.universe) or np.any(dictionary[1:] <= dictionary[:-1]):
        raise ValueError("dictionary must contain sorted unique in-range IDs")
    bitmap = np.frombuffer(chunk.payload, dtype=np.uint8, offset=2 * m)
    bitmap = bitmap.reshape(chunk.rows, (m + 7) // 8)
    if m % 8 and np.any(bitmap[:, -1] & (0xFF ^ ((1 << (m % 8)) - 1))):
        raise ValueError("nonzero padding bits")
    out = np.empty((chunk.rows, chunk.k), dtype=np.uint16)
    for i, row in enumerate(bitmap):
        locations = np.flatnonzero(np.unpackbits(row, bitorder="little", count=m))
        if locations.size != chunk.k:
            raise ValueError("bitmap row population does not match k")
        out[i] = dictionary[locations]
    return out


def _identity_matches(identity, expected):
    if not isinstance(identity, CacheIdentity) or not isinstance(expected, CacheIdentity):
        raise ValueError("identity and expected_identity must be CacheIdentity values")
    if identity != expected:
        raise ValueError("cache identity mismatch")


def _entry(chunk, offset):
    return {"offset": offset, "nbytes": len(chunk.payload), "codec": chunk.codec,
            "rows": chunk.rows, "k": chunk.k, "universe": chunk.universe,
            "dictionary_size": chunk.dictionary_size,
            "sha256": hashlib.sha256(chunk.payload).hexdigest()}


def _from_entry(entry, payload):
    return EncodedChunk(*(entry[key] for key in
                          ("codec", "rows", "k", "universe", "dictionary_size")), payload)


def write_store(directory, identity: CacheIdentity, chunks: Iterable[EncodedChunk]) -> None:
    """Create a new directory with data.bin and manifest.json; never overwrite.

    A failed write leaves its partial directory for inspection. A manifest is
    written only after all payloads have been validated and closed.
    """
    _identity_matches(identity, identity)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    entries, offset = [], 0
    with (directory / "data.bin").open("xb") as stream:
        for chunk in chunks:
            if len(entries) >= MAX_CHUNKS:
                raise ValueError("too many chunks")
            decode_chunk(chunk)
            entries.append(_entry(chunk, offset))
            stream.write(chunk.payload)
            offset += len(chunk.payload)
    manifest = {"version": 1, "identity": asdict(identity), "entries": entries}
    encoded = json.dumps(manifest, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise ValueError("manifest exceeds byte limit")
    with (directory / "manifest.json").open("xb") as stream:
        stream.write(encoded)


class DiskStore:
    """Indexed reads from one data file; no full-cache payload preload."""

    def __init__(self, directory, expected_identity: CacheIdentity):
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("manifest exceeds byte limit")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or type(manifest.get("version")) is not int:
                raise ValueError("invalid manifest")
            if manifest["version"] != 1:
                raise ValueError("unsupported manifest version")
            self.identity = CacheIdentity(**manifest["identity"])
            _identity_matches(self.identity, expected_identity)
            self.entries = manifest["entries"]
        except (TypeError, KeyError) as exc:
            raise ValueError("invalid manifest") from exc
        if not isinstance(self.entries, list) or len(self.entries) > MAX_CHUNKS:
            raise ValueError("invalid chunk table")
        self._path = directory / "data.bin"
        file_size, cursor = self._path.stat().st_size, 0
        for entry in self.entries:
            try:
                if not isinstance(entry, dict):
                    raise ValueError("invalid chunk entry")
                size = _payload_size(*(entry[key] for key in
                    ("codec", "rows", "k", "universe", "dictionary_size")))
                _integer(entry["offset"], "offset", 0, file_size)
                _integer(entry["nbytes"], "nbytes", 1, MAX_CHUNK_BYTES)
                digest = entry["sha256"]
                if not isinstance(digest, str) or len(digest) != 64 or any(
                        c not in "0123456789abcdef" for c in digest):
                    raise ValueError("invalid SHA256")
                if entry["offset"] != cursor or entry["nbytes"] != size:
                    raise ValueError("invalid chunk offset or length")
                cursor += size
                if cursor > file_size:
                    raise ValueError("chunk extends past end of data file")
            except (KeyError, TypeError) as exc:
                raise ValueError("invalid chunk entry") from exc
        if cursor != file_size:
            raise ValueError("data file size does not match chunk table")

    def __len__(self):
        return len(self.entries)

    def read(self, index) -> EncodedChunk:
        _integer(index, "chunk index", 0, len(self.entries) - 1)
        entry = self.entries[index]
        with self._path.open("rb") as stream:
            stream.seek(entry["offset"])
            payload = stream.read(entry["nbytes"])
        if len(payload) != entry["nbytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("chunk checksum or length mismatch")
        # A checksum validates the bytes without redundantly decoding on CPU in
        # the GPU prefetch path. write_store validates packets before saving.
        # Cache directories are trusted local artifacts, not authenticated input.
        return _from_entry(entry, payload)


class MemoryStore:
    """DRAM baseline retaining compressed chunks, with the same read interface."""

    def __init__(self, chunks: Iterable[EncodedChunk], identity: CacheIdentity,
                 expected_identity: CacheIdentity):
        _identity_matches(identity, expected_identity)
        self.identity, self.entries, self._chunks = identity, [], []
        offset = 0
        for chunk in chunks:
            if len(self._chunks) >= MAX_CHUNKS:
                raise ValueError("too many chunks")
            decode_chunk(chunk)
            self.entries.append(_entry(chunk, offset))
            self._chunks.append(chunk)
            offset += len(chunk.payload)

    def __len__(self):
        return len(self._chunks)

    def read(self, index) -> EncodedChunk:
        _integer(index, "chunk index", 0, len(self._chunks) - 1)
        return self._chunks[index]
