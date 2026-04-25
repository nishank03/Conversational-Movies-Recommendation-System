"""CRS engine implementations.

Each engine implements ``BaseCRS`` so the API and evaluation code treat them
interchangeably.
"""
from crs.crs_engines.base import BaseCRS, EngineContext
from crs.crs_engines.few_shot_crs import FewShotCRS
from crs.crs_engines.rag_crs import RAGCRS

__all__ = ["BaseCRS", "EngineContext", "FewShotCRS", "RAGCRS"]
