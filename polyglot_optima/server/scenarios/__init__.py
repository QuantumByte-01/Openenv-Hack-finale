"""Scenario subsystem: hardware profiles, datasets, generators, curriculum."""

from .hardware_profiles import HARDWARE_PROFILES, HARDWARE_BY_CLASS, profile_by_id
from .trap_library import TRAP_LIBRARY, get_trap_by_id, sample_trap
from .generator import TemplateGenerator, generate_from_template
from .dataset_loader import DatasetLoader, sample_function
from .adaptive_curriculum import AdaptiveCurriculum, MAX_LEVEL

__all__ = [
    "HARDWARE_PROFILES",
    "HARDWARE_BY_CLASS",
    "profile_by_id",
    "TRAP_LIBRARY",
    "get_trap_by_id",
    "sample_trap",
    "TemplateGenerator",
    "generate_from_template",
    "DatasetLoader",
    "sample_function",
    "AdaptiveCurriculum",
    "MAX_LEVEL",
]
