import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

if not OPENROUTER_API_KEY.strip():
    raise ValueError("OPENROUTER_API_KEY is not set. Add it to .env or export it.")

# GPT models via OpenRouter
BASE_MODEL  = "openai/gpt-4o-mini"
JUDGE_MODEL = "openai/gpt-4o"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
NLI_MODEL_NAME   = "cross-encoder/nli-deberta-v3-small"

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Loading NLI model...")
nli_model = CrossEncoder(NLI_MODEL_NAME)
print("Ready!")
