"""
Abstract Base Parser Interface
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from architectai_dataset_builder.models.canonical import ArchitectAISample
from architectai_dataset_builder.models.evaluation import (
    ArchitectureGenerationEvalSample,
    DiagramEvalSample,
    FreeResponseEvalSample,
    MultipleChoiceEvalSample,
)

ParsedSampleType = (
    ArchitectAISample
    | MultipleChoiceEvalSample
    | FreeResponseEvalSample
    | ArchitectureGenerationEvalSample
    | DiagramEvalSample
    | dict[str, Any]
)


class BaseParser(ABC):
    def __init__(self, source_id: str) -> None:
        self.source_id = source_id

    @abstractmethod
    def parse_directory(self, raw_dir: Path) -> list[Any]:
        """Parses all raw files in raw_dir into parsed data objects."""
