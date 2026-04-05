import os
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

if not os.environ["GROQ_API_KEY"].strip():
    raise ValueError("GROQ_API_KEY is not set. Run: export GROQ_API_KEY=your_key_here")

BASE_MODEL = "llama-3.1-8b-instant"
JUDGE_MODEL = "llama-3.3-70b-versatile"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

client = Groq(api_key=GROQ_API_KEY)

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Loading NLI model...")
nli_model = CrossEncoder(NLI_MODEL_NAME)
print("Ready!")
