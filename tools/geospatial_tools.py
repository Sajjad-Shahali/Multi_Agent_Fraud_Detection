"""Deterministic geospatial fraud tools.

Uses the haversine formula to compute great-circle distances.
No LLM calls inside any function.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from haversine import haversine, Unit
from langchain_core.tools import tool


@tool
def detect_impossible_travel(
    sender_id: str,
    tx_timestamp: str,
    tx_lat: float,
    tx_lng: float,
    locations_json: str,
    max_speed_kmh: float = 900.0,
) -> Dict[str, Any]:
    """Flag if the user's GPS history requires faster-than-possible travel to reach the tx location.

    Args:
        sender_id: Sender ID prefix used to match biotag entries.
        tx_timestamp: ISO-8601 timestamp of the target transaction.
        tx_lat: Latitude of the transaction location.
        tx_lng: Longitude of the transaction location.
        locations_json: JSON-serialised list of GPS ping dicts.
        max_speed_kmh: Maximum plausible speed in km/h (default 900 allows air travel).

    Returns:
        {"flag": bool, "max_speed_kmh": float, "n_pings_checked": int}
    """
    try:
        pings = json.loads(locations_json)
        ts = datetime.fromisoformat(tx_timestamp)
        tx_point = (tx_lat, tx_lng)

        # Match pings to this sender by the prefix portion of the biotag
        sender_prefix = sender_id.split("-")[0] if "-" in sender_id else sender_id
        user_pings = [
            p for p in pings
            if str(p.get("biotag", "")).startswith(sender_prefix)
        ]

        max_obs_speed = 0.0
        for ping in user_pings:
            ping_ts = datetime.fromisoformat(ping["timestamp"])
            dt_hours = abs((ts - ping_ts).total_seconds()) / 3600.0
            if dt_hours < 0.083:  # less than 5 minutes — skip
                continue
            dist_km = haversine(tx_point, (ping["lat"], ping["lng"]), unit=Unit.KILOMETERS)
            speed = dist_km / dt_hours
            max_obs_speed = max(max_obs_speed, speed)

        flag = max_obs_speed > max_speed_kmh
        return {
            "flag": flag,
            "max_speed_kmh": round(max_obs_speed, 1),
            "n_pings_checked": len(user_pings),
        }
    except Exception:
        return {"flag": False, "max_speed_kmh": 0.0, "n_pings_checked": 0}


def detect_impossible_travel_direct(
    sender_id: str,
    tx_timestamp: Any,
    tx_lat: Optional[float],
    tx_lng: Optional[float],
    pings: List[dict],
    max_speed_kmh: float = 900.0,
) -> Dict[str, Any]:
    """Direct version that accepts a list of dicts — used by featurizer."""
    if tx_lat is None or tx_lng is None:
        return {"flag": False, "max_speed_kmh": 0.0, "n_pings_checked": 0}
    try:
        ts = pd.to_datetime(tx_timestamp)
        tx_point = (tx_lat, tx_lng)
        sender_prefix = sender_id.split("-")[0] if "-" in sender_id else sender_id
        user_pings = [p for p in pings if str(p.get("biotag", "")).startswith(sender_prefix)]
        max_obs_speed = 0.0
        for ping in user_pings:
            ping_ts = datetime.fromisoformat(str(ping["timestamp"]))
            dt_hours = abs((ts.to_pydatetime() - ping_ts).total_seconds()) / 3600.0
            if dt_hours < 0.083:
                continue
            dist_km = haversine(tx_point, (float(ping["lat"]), float(ping["lng"])), unit=Unit.KILOMETERS)
            speed = dist_km / dt_hours
            max_obs_speed = max(max_obs_speed, speed)
        return {
            "flag": max_obs_speed > max_speed_kmh,
            "max_speed_kmh": round(max_obs_speed, 1),
            "n_pings_checked": len(user_pings),
        }
    except Exception:
        return {"flag": False, "max_speed_kmh": 0.0, "n_pings_checked": 0}


@tool
def cluster_location_behavior(locations_json: str, sender_id: str) -> Dict[str, Any]:
    """Derive the user's usual activity zone as a centroid and radius.

    Args:
        locations_json: JSON-serialised list of GPS ping dicts.
        sender_id: Sender ID to filter pings by biotag prefix.

    Returns:
        {"centroid_lat": float, "centroid_lng": float, "radius_km": float, "n_pings": int}
    """
    try:
        pings = json.loads(locations_json)
        sender_prefix = sender_id.split("-")[0] if "-" in sender_id else sender_id
        user_pings = [
            p for p in pings
            if str(p.get("biotag", "")).startswith(sender_prefix)
        ]
        if not user_pings:
            return {"centroid_lat": 0.0, "centroid_lng": 0.0, "radius_km": 0.0, "n_pings": 0}

        lats = [p["lat"] for p in user_pings]
        lngs = [p["lng"] for p in user_pings]
        centroid = (float(np.mean(lats)), float(np.mean(lngs)))
        radius = max(
            haversine(centroid, (p["lat"], p["lng"]), unit=Unit.KILOMETERS)
            for p in user_pings
        )
        return {
            "centroid_lat": round(centroid[0], 5),
            "centroid_lng": round(centroid[1], 5),
            "radius_km": round(radius, 2),
            "n_pings": len(user_pings),
        }
    except Exception:
        return {"centroid_lat": 0.0, "centroid_lng": 0.0, "radius_km": 0.0, "n_pings": 0}


@tool
def distance_from_home(
    home_lat: float,
    home_lng: float,
    tx_lat: float,
    tx_lng: float,
) -> Dict[str, Any]:
    """Compute great-circle distance in km between the user's home and the tx location.

    Args:
        home_lat: User's home latitude (from users.json residence).
        home_lng: User's home longitude.
        tx_lat: Transaction location latitude.
        tx_lng: Transaction location longitude.

    Returns:
        {"distance_km": float}
    """
    try:
        dist = haversine((home_lat, home_lng), (tx_lat, tx_lng), unit=Unit.KILOMETERS)
        return {"distance_km": round(dist, 2)}
    except Exception:
        return {"distance_km": 0.0}


# lazy import to avoid circular
import pandas as pd
