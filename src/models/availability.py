"""Optional OpenMP / booster availability helpers (macOS often needs libomp)."""

from __future__ import annotations

XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
XGBOOST_ERROR: str | None = None
LIGHTGBM_ERROR: str | None = None

try:
    import xgboost  # noqa: F401

    XGBOOST_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 — capture native load failures
    XGBOOST_ERROR = str(exc)

try:
    import lightgbm  # noqa: F401

    LIGHTGBM_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    LIGHTGBM_ERROR = str(exc)


def booster_status() -> dict[str, object]:
    return {
        "xgboost": XGBOOST_AVAILABLE,
        "lightgbm": LIGHTGBM_AVAILABLE,
        "xgboost_error": XGBOOST_ERROR,
        "lightgbm_error": LIGHTGBM_ERROR,
        "hint": "On macOS: brew install libomp  (then re-run training)",
    }
