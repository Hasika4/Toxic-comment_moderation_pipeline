import os, re, json, time, logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NLTK setup 
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()

def clean(text: str) -> str:
    text = re.sub(r'http\S+|www\.\S+', ' ', str(text).lower())
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    return " ".join(stemmer.stem(w) for w in tokens
                    if w not in stop_words and len(w) > 2)

# Load artifacts
artifacts = joblib.load("model.pkl")
xgb   = artifacts['xgb']
tfidf = artifacts['tfidf']
THRESHOLD = 0.630    # optimal threshold from training

# Gemini setup 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=GEMINI_API_KEY)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"toxic_flag": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["toxic_flag"]
}

SYSTEM_PROMPT = """You are a content moderation classifier for online news comments.

Given a comment, decide whether a human moderator would consider it toxic.

Mark "yes" if the comment contains insults, personal attacks, obscenity,
threats, identity-based attacks, or hostile language likely to make someone
leave the discussion. Mark "yes" even for mild name-calling, dismissive
insults, or condescending remarks aimed at people rather than ideas.

Mark "no" for ordinary opinions, disagreement, criticism of ideas or
policies, and strongly worded but civil argument.

Return JSON matching the required schema."""

GEN_CONFIG = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "max_output_tokens": 256,
    "response_mime_type": "application/json",
    "response_schema": RESPONSE_SCHEMA,
}

gem = genai.GenerativeModel("gemini-3.1-flash-lite",
                             system_instruction=SYSTEM_PROMPT)

# Gemini call with retry 
def gemini_classify(text: str) -> str:
    for attempt in range(3):
        try:
            r = gem.generate_content(text[:1000], generation_config=GEN_CONFIG)
            return json.loads(r.text).get("toxic_flag", "no")
        except Exception as e:
            if "429" in str(e):
                logger.warning(f"Rate limit hit, retrying ({attempt+1}/3)")
                time.sleep(20)
                continue
            logger.error(f"Gemini error: {e}")
            break
    return "fallback"   # triggers XGBoost prediction on error

# FastAPI app 
app = FastAPI(title="Toxic Comment Moderation API")

class CommentRequest(BaseModel):
    comment: str

class PredictionResponse(BaseModel):
    comment:    str
    prediction: int          # 1 = toxic, 0 = not toxic
    confidence: float
    routed_to:  str          # "xgboost" or "gemini"
    label:      str          # "toxic" or "not toxic"

@app.get("/")
def root():
    return {"status": "ok", "service": "Toxic Comment Moderation API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: CommentRequest):
    if not req.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    # preprocess and vectorize
    cleaned  = clean(req.comment)
    features = tfidf.transform([cleaned])
    prob     = float(xgb.predict_proba(features)[0][1])

    # routing decision
    if 0.4 <= prob <= 0.8:
        logger.info(f"Routing to Gemini (prob={prob:.3f})")
        flag = gemini_classify(req.comment)

        if flag == "fallback":
            # API error — fall back to classifier
            prediction = int(prob >= THRESHOLD)
            routed_to  = "xgboost_fallback"
        else:
            prediction = 1 if flag == "yes" else 0
            routed_to  = "gemini"
    else:
        prediction = int(prob >= THRESHOLD)
        routed_to  = "xgboost"

    return PredictionResponse(
        comment    = req.comment,
        prediction = prediction,
        confidence = round(prob, 4),
        routed_to  = routed_to,
        label      = "toxic" if prediction == 1 else "not toxic"
    )