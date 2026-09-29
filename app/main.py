"""HTTP API for the chime-bell partial-peak consolidation service."""

from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .solver import PeakResult, SolveRequest, SolveResult, solve

app = FastAPI(
    title="编钟残音联合基频分析",
    description=(
        "Jointly select harmonic numbers and rejectable noise peaks so that "
        "one fundamental frequency lies within every adopted peak's tolerance."
    ),
    version="1.0.0",
)


class PeakIn(BaseModel):
    frequency: float = Field(..., description="峰值频率（Hz，十进制，按频率升序录入）")
    tolerance: float = Field(..., description="该峰允许的频率绝对偏差（Hz，非负）")

    @field_validator("frequency")
    @classmethod
    def _freq_positive(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")) or v <= 0:
            raise ValueError("峰值频率必须为正的有限数值")
        return v

    @field_validator("tolerance")
    @classmethod
    def _tol_nonneg(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")) or v < 0:
            raise ValueError("允许偏差必须为非负的有限数值")
        return v


class AnalysisRequest(BaseModel):
    peaks: List[PeakIn] = Field(..., description="6 至 18 条按频率严格升序排列的峰值")
    f0_min: float = Field(..., description="候选基频闭区间下界（Hz）")
    f0_max: float = Field(..., description="候选基频闭区间上界（Hz）")
    max_harmonic: int = Field(..., description="最高泛音序号（≥2）")
    max_rejected: int = Field(..., description="最多可剔除的杂峰数")

    @field_validator("peaks")
    @classmethod
    def _validate_peaks(cls, peaks: List[PeakIn]) -> List[PeakIn]:
        if not 6 <= len(peaks) <= 18:
            raise ValueError("请录入 6 至 18 条峰值")
        prev: Optional[float] = None
        for p in peaks:
            if prev is not None and p.frequency <= prev:
                raise ValueError("峰值频率必须按严格升序排列且不得重复")
            prev = p.frequency
        return peaks

    @field_validator("f0_min", "f0_max")
    @classmethod
    def _positive_finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")) or v <= 0:
            raise ValueError("基频区间端点必须为正的有限数值")
        return v

    @field_validator("f0_max")
    @classmethod
    def _ordered_interval(cls, v: float, info) -> float:
        lo = info.data.get("f0_min")
        if lo is not None and v < lo:
            raise ValueError("候选基频区间上界不得小于下界")
        return v

    @field_validator("max_harmonic")
    @classmethod
    def _h_range(cls, v: int) -> int:
        if not 2 <= v <= 128:
            raise ValueError("最高泛音序号必须在 2 至 128 之间")
        return v

    @field_validator("max_rejected")
    @classmethod
    def _r_range(cls, v: int) -> int:
        if v < 0:
            raise ValueError("最多可剔除杂峰数不得为负")
        return v


class PeakOut(BaseModel):
    index: int
    frequency: float
    tolerance: float
    adopted: bool
    harmonic: Optional[int]
    expected: Optional[float]
    abs_error: Optional[float]
    rel_error: Optional[float]

    @classmethod
    def from_domain(cls, p: PeakResult) -> "PeakOut":
        return cls(
            index=p.index,
            frequency=p.frequency,
            tolerance=p.tolerance,
            adopted=p.adopted,
            harmonic=p.harmonic,
            expected=p.expected,
            abs_error=p.abs_error,
            rel_error=p.rel_error,
        )


class AnalysisResponse(BaseModel):
    feasible: bool
    f0: Optional[float]
    peaks: List[PeakOut]
    adopted_count: int
    rejected_count: int
    max_relative_error: Optional[float]
    sum_squared_relative_error: Optional[float]
    first_unexplainable_index: Optional[int]
    reason: Optional[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze(payload: AnalysisRequest) -> AnalysisResponse:
    n_peaks = len(payload.peaks)
    if payload.max_rejected > n_peaks - 1:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_request",
                "message": (
                    f"最多可剔除 {n_peaks - 1} 条杂峰"
                    "（至少保留 1 条采用峰才能确定基频）"
                ),
            },
        )

    req = SolveRequest(
        frequencies=tuple(p.frequency for p in payload.peaks),
        tolerances=tuple(p.tolerance for p in payload.peaks),
        f0_min=payload.f0_min,
        f0_max=payload.f0_max,
        max_harmonic=payload.max_harmonic,
        max_rejected=payload.max_rejected,
    )
    result: SolveResult = solve(req)

    by_index = {p.index: p for p in result.adopted}
    by_index.update({p.index: p for p in result.rejected})
    peaks_out = [PeakOut.from_domain(by_index[i]) for i in range(n_peaks)]

    return AnalysisResponse(
        feasible=result.feasible,
        f0=result.f0,
        peaks=peaks_out,
        adopted_count=len(result.adopted),
        rejected_count=len(result.rejected),
        max_relative_error=result.max_relative_error,
        sum_squared_relative_error=result.sum_squared_relative_error,
        first_unexplainable_index=result.first_unexplainable_index,
        reason=result.reason,
    )


# Frontend (static single-page UI) is served last so /health and /api win.
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
