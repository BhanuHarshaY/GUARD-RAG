import re
import time


def ask_llm(prompt, client, model, temperature=0.0, max_tokens=300):
    import time as _time
    for attempt in range(12):
        try:
            start = _time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency = round(_time.time() - start, 3)
            text = response.choices[0].message.content.strip()
            usage = getattr(response, "usage", None)
            total_tokens = getattr(usage, "total_tokens", None) if usage else None
            return {
                "text": text,
                "latency": latency,
                "tokens": total_tokens if total_tokens is not None else 0,
            }
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                # parse "try again in Xm Y.Zs" if present
                import re
                match = re.search(r"try again in (?:(\d+)m)?(\d+\.?\d*)s", err)
                if match:
                    mins = int(match.group(1) or 0)
                    secs = float(match.group(2))
                    wait = mins * 60 + secs + 2
                else:
                    wait = min(2 ** attempt * 5, 600)
                print(f"  Rate limited, waiting {wait:.0f}s...")
                _time.sleep(wait)
            else:
                raise
    raise RuntimeError("Failed after 12 retries")


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
