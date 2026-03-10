from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path

# Optional: FAISS for fast search (recommended)
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

APP_DIR = Path(__file__).parent
NOTES_PATH = APP_DIR / "notes.json"
EMB_PATH = APP_DIR / "embeddings.npy"
FAISS_PATH = APP_DIR / "faiss.index"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model once
from pathlib import Path
MODEL_PATH = Path(__file__).parent.parent / "training" / "finetuned_model"
model = SentenceTransformer(str(MODEL_PATH))

def load_notes():
    if NOTES_PATH.exists():
        try:
            return json.loads(NOTES_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_notes(notes_list):
    NOTES_PATH.write_text(json.dumps(notes_list, indent=2), encoding="utf-8")

notes = load_notes()

# In-memory index
note_embeddings: np.ndarray | None = None
faiss_index = None

def rebuild_index():
    """Recompute embeddings + rebuild FAISS, then cache to disk."""
    global note_embeddings, faiss_index

    if len(notes) == 0:
        note_embeddings = None
        faiss_index = None
        if EMB_PATH.exists():
            EMB_PATH.unlink()
        if FAISS_PATH.exists():
            FAISS_PATH.unlink()
        return

    embs = model.encode([n["text"] for n in notes], normalize_embeddings=True)
    note_embeddings = np.array(embs, dtype=np.float32)

    # Cache embeddings to disk
    np.save(EMB_PATH, note_embeddings)

    # Build FAISS index (inner product == cosine if normalized)
    if HAS_FAISS:
        d = note_embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(d)
        faiss_index.add(note_embeddings)
        faiss.write_index(faiss_index, str(FAISS_PATH))
    else:
        faiss_index = None

def load_cached_index() -> bool:
    """Load cached embeddings + FAISS index if present."""
    global note_embeddings, faiss_index

    if len(notes) == 0:
        return False

    if not EMB_PATH.exists():
        return False

    try:
        note_embeddings = np.load(EMB_PATH).astype(np.float32)
    except Exception:
        return False

    if HAS_FAISS and FAISS_PATH.exists():
        try:
            faiss_index = faiss.read_index(str(FAISS_PATH))
        except Exception:
            faiss_index = None
    else:
        faiss_index = None

    # sanity check: cache matches notes length
    if note_embeddings.shape[0] != len(notes):
        note_embeddings = None
        faiss_index = None
        return False

    return True

# Build/load index on startup
if not load_cached_index():
    rebuild_index()

class NoteIn(BaseModel):
    text: str

class SearchIn(BaseModel):
    query: str
    top_k: int = 5

@app.get("/health")
def health():
    return {
        "status": "ok",
        "notes": len(notes),
        "faiss": HAS_FAISS and (faiss_index is not None),
        "cached_embeddings": EMB_PATH.exists(),
    }

@app.get("/notes")
def list_notes():
    return {"count": len(notes), "notes": notes}

@app.post("/notes")
def add_note(payload: NoteIn):
    text = payload.text.strip()
    if not text:
        return {"ok": False, "error": "Empty note"}

    notes.append({"id": len(notes) + 1, "text": text})
    save_notes(notes)
    rebuild_index()
    return {"ok": True, "note": notes[-1], "count": len(notes)}

@app.delete("/notes/{note_id}")
def delete_note(note_id: int):
    global notes
    idx = next((i for i, n in enumerate(notes) if n["id"] == note_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Note not found")

    deleted = notes.pop(idx)

    # Reassign IDs to stay clean (optional)
    notes = [{"id": i + 1, "text": n["text"]} for i, n in enumerate(notes)]

    save_notes(notes)
    rebuild_index()
    return {"ok": True, "deleted": deleted, "count": len(notes)}

@app.post("/search")
def search(payload: SearchIn):
    if len(notes) == 0:
        return {"query": payload.query, "results": []}

    q = payload.query.strip()
    if not q:
        return {"query": payload.query, "results": []}

    top_k = max(1, min(payload.top_k, len(notes)))

    # Query embedding
    q_emb = model.encode([q], normalize_embeddings=True).astype(np.float32)

    # Use FAISS if available, else fallback to numpy dot product
    if faiss_index is not None:
        scores, idxs = faiss_index.search(q_emb, top_k)  # (1,k)
        idxs = idxs[0]
        scores = scores[0]
    else:
        if note_embeddings is None:
            rebuild_index()
        sims = note_embeddings @ q_emb[0]
        idxs = np.argsort(-sims)[:top_k]
        scores = sims[idxs]

    results = []
    for i, s in zip(idxs, scores):
        i = int(i)
        results.append({"id": notes[i]["id"], "text": notes[i]["text"], "score": float(s)})

    return {"query": q, "results": results}