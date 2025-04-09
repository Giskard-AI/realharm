from typing import Literal

from pydantic import BaseModel, computed_field


class ModerationOutput(BaseModel):
    safe: bool
    categories: list[str]

    @computed_field
    @property
    def label(self) -> Literal["safe", "unsafe"]:
        return "safe" if self.safe else "unsafe"
