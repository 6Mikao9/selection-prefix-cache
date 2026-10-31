"""CPU-only evidence checks for experimental native SALA runs."""
import json
import math
from pathlib import Path
import re


def public_provenance(value):
    """Remove private filesystem locations from legacy prepared records."""
    if isinstance(value, dict):
        return {key: public_provenance(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_provenance(item) for item in value]
    if isinstance(value, str) and (value.startswith(("/", "\\\\", "~"))
                                  or re.match(r"^[A-Za-z]:[\\/]", value)):
        return "local-artifact"
    return value


def validate_fraction(fraction):
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("head fractions must be finite and in [0, 1]")
    return fraction


def log_offset(path):
    path = Path(path)
    return path.stat().st_size if path.exists() else 0


def selector_evidence(path, offset, run_id, identity, mode, tokens, fraction,
                      expected_layers=None):
    """Require actual intercepted native calls for this request and exact cut."""
    validate_fraction(fraction)
    path = Path(path)
    if not path.exists() or path.stat().st_size < offset:
        raise RuntimeError("Missing adapter log; native interception is unverified")
    with path.open("rb") as stream:
        stream.seek(offset)
        events = [json.loads(line) for line in stream if line.strip()]
    events = [event for event in events if event.get("run_id") == run_id]
    if not events:
        raise RuntimeError("No selector calls intercepted: hook missing or request used dense attention")
    layers = []
    for event in events:
        if event.get("identity") != identity or event.get("mode") != mode:
            raise RuntimeError("Adapter request identity or mode mismatch")
        block = event.get("block_size")
        if type(block) is not int or block <= 0:
            raise RuntimeError("Adapter did not report a valid block size")
        head = tokens if fraction == 1 else int(tokens * fraction) // block * block
        work = head if mode == "replay" else tokens
        if (event.get("query_tokens") != tokens
                or event.get("effective_head") != head
                or event.get("native_selector_queries") != work
                or event.get("native_selector_called") != bool(work)):
            raise RuntimeError("Adapter token count, effective split, or selector work mismatch")
        layer = event.get("layer")
        if type(layer) is not int or layer in layers:
            raise RuntimeError("Missing or repeated layer; unchunked single-request execution required")
        layers.append(layer)
    if expected_layers is not None and set(layers) != set(expected_layers):
        raise RuntimeError("Capture/replay sparse layer coverage differs")
    return {"layers": sorted(layers), "selector_calls": len(events),
            "effective_head": events[0]["effective_head"],
            "native_selector_queries_per_layer": events[0]["native_selector_queries"],
            "adapter_interception_verified": True}


def logprobs_equal(actual, reference):
    """Missing observations cannot count as output equivalence."""
    actual = actual.get("meta_info", {}).get("output_token_logprobs")
    reference = reference.get("meta_info", {}).get("output_token_logprobs")
    if actual is None or reference is None or not actual or not reference:
        raise RuntimeError("Requested output token logprobs were not returned")
    return actual == reference
