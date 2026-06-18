"""Flask routes for workload generation and parsing."""

import logging
import os
import sys
import tempfile

from flask import Blueprint, request, jsonify

from server.auth.auth_service import require_auth
from server.simulation.workload_generator import (
    generate_megatron_workload,
    generate_custom_workload,
    parse_workload,
    MODEL_CONFIGS,
    _LEGACY_MODEL_MAP,
)

logger = logging.getLogger(__name__)

workload_bp = Blueprint("workload", __name__, url_prefix="/api/simulation/workload")


@workload_bp.route("/models", methods=["GET"])
@require_auth
def api_list_models():
    """List available model families and sizes for workload generation."""
    families: dict = {}
    for name, cfg in MODEL_CONFIGS.items():
        family = cfg.get("family", "Other")
        if family not in families:
            families[family] = []
        rec = cfg.get("recommended", {})
        families[family].append({
            "id": name,
            "num_layers": cfg["num_layers"],
            "hidden_size": cfg["hidden_size"],
            "num_experts": cfg.get("num_experts", 0),
            "recommended": {
                "tp": rec.get("tp", 8),
                "pp": rec.get("pp", 1),
                "ep": rec.get("ep", 1),
                "gpus": rec.get("gpus", 8),
            },
        })
    return jsonify({"families": families})


@workload_bp.route("/generate-preset", methods=["POST"])
@require_auth
def api_generate_preset():
    """Generate workload from a preset model configuration."""
    data = request.get_json(silent=True) or {}

    model_size = data.get("model_size", "LLaMA-7B")
    resolved = _LEGACY_MODEL_MAP.get(model_size, model_size)
    if resolved not in MODEL_CONFIGS:
        return jsonify({"error": f"Unknown model '{model_size}'. Valid: {list(MODEL_CONFIGS)}"}), 400

    tp_size = int(data.get("tp_size", 8))
    dp_size = int(data.get("dp_size", 1))
    pp_size = int(data.get("pp_size", 1))
    ep_size = int(data.get("ep_size", 1))
    all_gpus = int(data.get("all_gpus", tp_size * dp_size * pp_size * ep_size))
    ga = int(data.get("ga", 8))
    vpp = int(data.get("vpp", 1))
    _default_comp = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "example", "aiob_data", "gpu_compute_timing.txt")
    comp_filepath = data.get("comp_filepath", _default_comp)
    aiob_enable = bool(data.get("aiob_enable", os.path.exists(comp_filepath)))

    try:
        content, layers = generate_megatron_workload(
            tp_size=tp_size,
            dp_size=dp_size,
            pp_size=pp_size,
            ep_size=ep_size,
            all_gpus=all_gpus,
            model_size=model_size,
            ga=ga,
            vpp=vpp,
            aiob_enable=aiob_enable,
            comp_filepath=comp_filepath,
        )
    except Exception as e:
        logger.exception("Workload generation failed")
        return jsonify({"error": str(e)}), 500

    return jsonify({"content": content, "layers": layers, "model_size": model_size})


@workload_bp.route("/generate-custom", methods=["POST"])
@require_auth
def api_generate_custom():
    """Generate workload from custom layer configurations."""
    data = request.get_json(silent=True) or {}

    required = ["tp_size", "dp_size", "pp_size", "all_gpus", "layers_config"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        content = generate_custom_workload(
            tp_size=int(data["tp_size"]),
            dp_size=int(data["dp_size"]),
            pp_size=int(data["pp_size"]),
            ep_size=int(data.get("ep_size", 1)),
            all_gpus=int(data["all_gpus"]),
            layers_config=data["layers_config"],
        )
    except Exception as e:
        logger.exception("Custom workload generation failed")
        return jsonify({"error": str(e)}), 500

    return jsonify({"content": content})


@workload_bp.route("/parse", methods=["POST"])
@require_auth
def api_parse_workload():
    """Parse workload content and return configuration."""
    data = request.get_json(silent=True) or {}
    content = data.get("content", "")
    if not content:
        return jsonify({"error": "content is required"}), 400

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        config = parse_workload(tmp_path)
        return jsonify({"config": config})
    except Exception as e:
        logger.exception("Workload parsing failed")
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@workload_bp.route("/presets", methods=["GET"])
@require_auth
def api_list_presets():
    """List available model presets."""
    presets = {k: v for k, v in MODEL_CONFIGS.items()}
    return jsonify({"presets": presets})


@workload_bp.route("/timeline-preview", methods=["POST"])
@require_auth
def api_timeline_preview():
    """Generate interactive Gantt timeline HTML from workload content.

    Returns HTML with canvas-rendered Gantt chart showing per-layer
    compute + TP/DP/EP/DP_EP communication timeline.
    """
    data = request.get_json(silent=True) or {}
    content = data.get("content", "")
    max_layers = min(int(data.get("max_layers", 0)), 200)
    rank = int(data.get("rank", 0))

    if not content:
        return jsonify({"error": "content is required"}), 400

    # Write workload to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Import visualize_workload dynamically (adds scripts/ to path)
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from visualize_workload import parse_workload as vw_parse, build_timeline, render_html

        workload = vw_parse(tmp_path)
        events = build_timeline(workload, max_layers=max_layers if max_layers > 0 else 0)
        html_content = render_html(workload, events, max_layers=max_layers if max_layers > 0 else 0, rank=rank)

        return jsonify({"html": html_content})
    except Exception as e:
        logger.exception("Timeline preview generation failed")
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)
