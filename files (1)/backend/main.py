from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import CompleteRequest, CompleteResponse
from model import CodeCompletionModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Code Completion API",
    description="Confidence-based code completion with 3-tier UI strategy",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
model = CodeCompletionModel()

@app.on_event("startup")
async def startup_event():
    logger.info("Loading model...")
    model.load()
    logger.info("Model loaded successfully.")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model.is_loaded}

@app.post("/api/complete", response_model=CompleteResponse)
def complete(request: CompleteRequest):
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
        
    try:
        completion, confidence, tokens = model.generate(
            prefix=request.prefix,
            max_new_tokens=request.max_new_tokens,
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Determine UI tier based on confidence
    if confidence >= 0.80:
        ui_mode = "inline"       # ghost text
    elif confidence >= 0.40:
        ui_mode = "collapsed"    # expandable panel
    else:
        ui_mode = "hidden"       # suppressed

    return CompleteResponse(
        completion=completion,
        confidence=round(confidence, 4),
        ui_mode=ui_mode,
        tokens=tokens,
    )
