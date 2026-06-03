"""Flask routes for ranktable generation and validation."""

import logging

from flask import Blueprint, request, jsonify

from server.auth.auth_service import require_auth
from server.simulation.ranktable_generator import (
    generate_ranktable,
    generate_ranktable_with_topology,
    validate_ranktable,
)

logger = logging.getLogger(__name__)

ranktable_bp = Blueprint("ranktable", __name__, url_prefix="/api/simulation/ranktable")


@ranktable_bp.route("/generate", methods=["POST"])
@require_auth
def api_generate_ranktable():
    """Generate ranktable from basic parameters."""
    data = request.get_json(silent=True) or {}

    rank_count = data.get("rank_count")
    gpus_per_rack = data.get("gpus_per_rack")

    if not rank_count or not gpus_per_rack:
        return jsonify({"error": "rank_count and gpus_per_rack are required"}), 400

    try:
        ranktable, rank_rack_map = generate_ranktable(
            rank_count=int(rank_count),
            gpus_per_rack=int(gpus_per_rack),
            num_racks=data.get("num_racks"),
            superpod_prefix=data.get("superpod_prefix", "rack"),
        )
    except Exception as e:
        logger.exception("RankTable generation failed")
        return jsonify({"error": str(e)}), 500

    return jsonify({"ranktable": ranktable, "rank_rack_map": rank_rack_map})


@ranktable_bp.route("/generate-custom", methods=["POST"])
@require_auth
def api_generate_custom_ranktable():
    """Generate ranktable with custom topology configuration."""
    data = request.get_json(silent=True) or {}

    rank_count = data.get("rank_count")
    topology = data.get("topology")

    if not rank_count or not topology:
        return jsonify({"error": "rank_count and topology are required"}), 400

    try:
        ranktable, rank_rack_map = generate_ranktable_with_topology(
            rank_count=int(rank_count),
            topology=topology,
        )
    except Exception as e:
        logger.exception("Custom RankTable generation failed")
        return jsonify({"error": str(e)}), 500

    return jsonify({"ranktable": ranktable, "rank_rack_map": rank_rack_map})


@ranktable_bp.route("/validate", methods=["POST"])
@require_auth
def api_validate_ranktable():
    """Validate a ranktable JSON structure."""
    data = request.get_json(silent=True) or {}
    ranktable = data.get("ranktable")

    if not ranktable:
        return jsonify({"error": "ranktable is required"}), 400

    is_valid, errors = validate_ranktable(ranktable)
    return jsonify({"valid": is_valid, "errors": errors})
