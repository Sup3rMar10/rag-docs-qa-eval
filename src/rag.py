"""RAG pipeline: retrieve -> build a grounded prompt -> generate with a local open model (no API keys)."""
from src.chunking import Chunk

ABSTAIN_TEXT = "I don't know based on the provided documentation."

# v1 was the first prompt; with a 1.5B model it often replied with only a citation such as "[2]".
# v2 requires an answer sentence before the citation. Both are kept so reported results stay reproducible.
RAG_SYSTEMS = {
    "v1": (
        "You answer questions about Azure Container Apps using ONLY the numbered passages provided. "
        "Be concise (one to three sentences) and cite the passages you used like [1] or [2]. "
        f"If the passages do not contain the answer, reply exactly: {ABSTAIN_TEXT}"
    ),
    "v2": (
        "You answer questions about Azure Container Apps using ONLY the numbered passages provided. "
        "Write a direct answer in one to three complete sentences that states the specific facts "
        "(numbers, names, limits, commands) found in the passages, then add the number of the passage you "
        "used in brackets, like [2]. Never reply with only a citation. "
        f"If the passages do not contain the answer, reply exactly: {ABSTAIN_TEXT}"
    ),
}
DEFAULT_PROMPT = "v2"
RAG_SYSTEM = RAG_SYSTEMS[DEFAULT_PROMPT]
CLOSED_BOOK_SYSTEM = (
    "You answer questions about Azure Container Apps. Be concise (one to three sentences). "
    f"If you are not sure, reply exactly: {ABSTAIN_TEXT}"
)


def format_context(passages: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] ({p.title} > {p.heading})\n{p.text}" if p.heading else f"[{i}] ({p.title})\n{p.text}"
                       for i, p in enumerate(passages, start=1))


def rag_messages(question: str, passages: list[Chunk], prompt: str = DEFAULT_PROMPT) -> list[dict]:
    user = f"Passages:\n{format_context(passages)}\n\nQuestion: {question}"
    return [{"role": "system", "content": RAG_SYSTEMS[prompt]}, {"role": "user", "content": user}]


def closed_book_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": CLOSED_BOOK_SYSTEM}, {"role": "user", "content": question}]


class LocalGenerator:
    """Greedy decoding with a local chat model (CPU by default)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        except TypeError:  # older transformers: torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device).eval()
        self.model_name = model_name

    def generate(self, messages: list[dict], max_new_tokens: int = 140) -> str:
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                      top_p=None, top_k=None,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def answer_question(question: str, retriever, chunks: list[Chunk], generator, k: int = 4) -> dict:
    hits = retriever.search(question, k)
    passages = [chunks[i] for i, _ in hits]
    text = generator.generate(rag_messages(question, passages))
    return {"answer": text,
            "sources": [{"n": n, "doc": p.doc, "title": p.title, "heading": p.heading, "score": s,
                         "text": p.text} for n, (p, (_, s)) in enumerate(zip(passages, hits, strict=True), 1)]}
