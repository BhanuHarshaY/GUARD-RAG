import re
import time


def ask_llm(prompt, client, model, temperature=0.0, max_tokens=300, max_retries=6):
    start = time.time()
    delay = 5.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            break
        except Exception as e:
            if "rate_limit_exceeded" in str(e) or "429" in str(e):
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                    continue
            raise
    latency = round(time.time() - start, 3)

    text = response.choices[0].message.content.strip()
    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    return {
        "text": text,
        "latency": latency,
        "tokens": total_tokens if total_tokens is not None else 0
    }


def format_retrieved_context(retrieved, max_chars=3500):
    lines = []
    total = 0
    for r in retrieved:
        line = f"[{r['chunk_id']}] {r['text']}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def strip_refinement_prefix(answer):
    if answer is None:
        return ""
    answer = str(answer).strip()

    m = re.search(r"REVISED:\s*(.*)", answer, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    m = re.search(r"FINAL:\s*(.*)", answer, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    return answer.strip()
