"""
Abstract Base Parser Interface
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Union
from architectai_dataset_builder.models.canonical import ArchitectAISample
from architectai_dataset_builder.models.evaluation import (
    MultipleChoiceEvalSample,
    FreeResponseEvalSample,
    ArchitectureGenerationEvalSample,
    DiagramEvalSample,
)

ParsedSampleType = Union[
    ArchitectAISample,
    MultipleChoiceEvalSample,
    FreeResponseEvalSample,
    ArchitectureGenerationEvalSample,
    DiagramEvalSample,
]


class BaseParser(ABC):
    def __init__(self, source_id: str):
        self.source_id = source_id

    @abstractmethod
    def parse_directory(self, raw_dir: Path) -> List[ParsedSampleType]:
        """Parses all raw files in raw_dir into parsed data objects."""
        pass
