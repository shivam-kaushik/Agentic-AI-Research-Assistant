"""Configuration module for Co-Investigator Agent."""
from .gcp_config import config, tables, GCPConfig, BigQueryTables
from .prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    SYNTHESIZER_USER_PROMPT,
    CONFLICT_DETECTOR_PROMPT,
)

__all__ = [
    "config",
    "tables",
    "GCPConfig",
    "BigQueryTables",
    "PLANNER_SYSTEM_PROMPT",
    "PLANNER_USER_PROMPT",
    "SYNTHESIZER_SYSTEM_PROMPT",
    "SYNTHESIZER_USER_PROMPT",
    "CONFLICT_DETECTOR_PROMPT",
]
