"""FastAPI backend.  Run from business_entity_resolution/:

    uvicorn api.main:app --port 8000
"""
import io
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from service import Matcher  # noqa: E402

app = FastAPI(title="Business Entity Resolution API", version="1.0")
_matcher = None


def matcher():
    global _matcher
    if _matcher is None:
        _matcher = Matcher()
    return _matcher


class Record(BaseModel):
    business_name: str
    business_address: str = ""
    country: str = ""


class ComparePair(BaseModel):
    a: Record
    b: Record


@app.on_event("startup")
def _load():
    matcher()


@app.get("/health")
def health():
    m = matcher()
    return {"status": "ok", "pool_records": int(m.index.n), "model": "lightgbm+isotonic",
            "decision_params": m.params}


@app.post("/match")
def match(rec: Record, top_k: int = 10):
    return matcher().match([rec.model_dump()], top_k=top_k)[0]


@app.post("/compare")
def compare(pair: ComparePair):
    return matcher().compare(pair.a.model_dump(), pair.b.model_dump())


@app.post("/batch")
async def batch(file: UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(await file.read()), sep="\t", dtype=str, keep_default_na=False)
    need = {"entity_id", "business_name", "business_address", "country"}
    if not need <= set(df.columns):
        raise HTTPException(400, f"TSV must have columns {sorted(need)}")
    if len(df) > 5000:
        raise HTTPException(400, "max 5000 rows per request")
    out = matcher().batch(df).to_csv(sep="\t", index=False)
    return StreamingResponse(io.StringIO(out), media_type="text/tab-separated-values",
                             headers={"Content-Disposition": "attachment; filename=matching_results.tsv"})


@app.get("/stats")
def stats():
    return matcher().metrics
