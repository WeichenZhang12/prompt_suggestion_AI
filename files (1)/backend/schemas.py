from pydantic import BaseModel, Field
from typing import Literal, List

class CompleteRequest(BaseModel):
    prefix: str = Field(..., description="Code prefix typed by the user")
    max_new_tokens: int = Field(default=64, ge=1, le=256)

    model_config = {
        "json_schema_extra": {
            "example": {
                "prefix": "def sort_list(arr):\n    ",
                "max_new_tokens": 64,
            }
        }
    }

class TokenInfo(BaseModel):
    text: str = Field(..., description="Decoded text of this token")
    logprob: float = Field(..., description="Log-probability of this token (≤ 0)")

class CompleteResponse(BaseModel):
    completion: str = Field(..., description="Generated code completion text")
    confidence: float = Field(..., description="Mean log-prob confidence score in [0, 1]")
    ui_mode: Literal["inline", "collapsed", "hidden"] = Field(
        ...,
        description=(
            "inline    → show as ghost text (confidence ≥ 0.80)\n"
            "collapsed → show in expandable panel (0.40 ≤ confidence < 0.80)\n"
            "hidden    → suppress entirely (confidence < 0.40)"
        ),
    )
    tokens: List[TokenInfo] = Field(
        default_factory=list,
        description="Per-token text and log-prob, used for the confidence heatmap",
    )
