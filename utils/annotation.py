"""Conservative extraction from ComfyUI's submitted graph (never execute nodes)."""
from collections import deque
import json
import math


SAMPLERS = {"KSampler", "KSamplerAdvanced", "D2 KSampler", "D2 KSampler(Advanced)"}


def build_annotation(prompt, unique_id=None, positive=None, negative=None, size=None):
    graph = prompt if isinstance(prompt, dict) else {}

    def node(node_id):
        value = graph.get(str(node_id), {})
        return value if isinstance(value, dict) else {}

    def inputs(node_id):
        value = node(node_id).get("inputs", {})
        return value if isinstance(value, dict) else {}

    def link(value):
        return (isinstance(value, list) and len(value) == 2
                and isinstance(value[0], (str, int))
                and isinstance(value[1], int) and str(value[0]) in graph)

    def literal(value):
        # Linked outputs are not runtime values; do not guess what custom nodes compute.
        return value if isinstance(value, (str, int, float)) and not isinstance(value, bool) else None

    def nearest(starts, predicate):
        queue = deque((str(s), 0) for s in starts)
        seen, found, depth = set(), [], None
        while queue:
            current, distance = queue.popleft()
            if current in seen or (depth is not None and distance > depth):
                continue
            seen.add(current)
            if predicate(current):
                found.append(current)
                depth = distance
                continue
            for value in inputs(current).values():
                if link(value):
                    queue.append((value[0], distance + 1))
        return found[0] if len(found) == 1 else None

    root = inputs(unique_id)
    media = root.get("images", root.get("video"))
    sampler = nearest([media[0]], lambda n: node(n).get("class_type") in SAMPLERS) if link(media) else None
    settings = inputs(sampler)
    # D2 gives its pipe priority. Its runtime contents cannot be inferred safely here.
    if link(settings.get("d2_pipe")):
        settings = {}

    def text(value):
        if isinstance(value, str):
            return value
        if link(value):
            source = node(value[0])
            if source.get("class_type") == "CLIPTextEncode":
                result = inputs(value[0]).get("text")
                return result if isinstance(result, str) else ""
        return ""

    pos = positive if isinstance(positive, str) else text(settings.get("positive"))
    neg = negative if isinstance(negative, str) else text(settings.get("negative"))
    fields = []

    def add(label, value):
        value = literal(value)
        if value is None or value == "" or (isinstance(value, float) and not math.isfinite(value)):
            return
        if isinstance(value, str) and any(c in value for c in ',\n\r"'):
            value = json.dumps(value, ensure_ascii=False)
        fields.append(f"{label}: {value}")

    add("Steps", settings.get("steps"))
    sampler_name = literal(settings.get("sampler_name"))
    scheduler = literal(settings.get("scheduler"))
    if sampler_name:
        add("Sampler", " ".join(str(v) for v in (sampler_name, scheduler) if v is not None and v != ""))
    add("CFG scale", settings.get("cfg"))
    add("Seed", settings.get("seed", settings.get("noise_seed")))
    if isinstance(size, (tuple, list)) and len(size) == 2 and all(isinstance(v, int) and v > 0 for v in size):
        add("Size", f"{size[0]}x{size[1]}")
    model_link = settings.get("model")
    if link(model_link):
        model = nearest([model_link[0]], lambda n: node(n).get("class_type") in {
            "CheckpointLoaderSimple", "CheckpointLoader", "UNETLoader", "D2 Checkpoint Loader", "D2 Load Diffusion Model", "D2 Load Diffusion Model Set"})
        model_settings = inputs(model)
        add("Model", model_settings.get("ckpt_name", model_settings.get("unet_name")))
    result = pos or ""
    if neg:
        result += ("\n\n" if result else "") + "Negative prompt:" + neg
    if fields:
        result += ("\n" if result else "") + ", ".join(fields)
    return result
