from tiers.llm_utils import ask_llm, format_retrieved_context


def tier1_basic_rag(question, retrieved, client, BASE_MODEL):
    context = format_retrieved_context(retrieved)

    prompt = f"""
You are answering a question using ONLY the provided evidence.

Rules:
1. Use only facts explicitly present in the evidence.
2. If the question asks what something consists of / includes / contains, list the supported components explicitly.
3. If the evidence is partial, give the supported partial answer rather than saying "Insufficient information."
4. Be concise and factual.
5. Do not mention document IDs or retrieval process.

Evidence:
{context}

Question:
{question}

Answer:
""".strip()

    out = ask_llm(prompt, client=client, model=BASE_MODEL, temperature=0.0, max_tokens=220)

    return {
        "answer": out["text"],
        "latency": out["latency"],
        "tokens": out["tokens"],
    }
