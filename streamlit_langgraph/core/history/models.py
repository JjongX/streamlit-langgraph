# History models for executor conversation tracking.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HistoryBlock:
    """Pure data block for conversation history."""
    category: str
    content: Any
    filename: Optional[str] = None
    file_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize block to dictionary format."""
        return {
            "category": self.category,
            "content": self.content,
            "filename": self.filename,
            "file_id": self.file_id,
        }


@dataclass
class HistorySection:
    """Pure data section for conversation history."""
    role: str
    blocks: List[HistoryBlock] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return len(self.blocks) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize section to dictionary format."""
        return {
            "role": self.role,
            "blocks": [block.to_dict() for block in self.blocks],
        }
