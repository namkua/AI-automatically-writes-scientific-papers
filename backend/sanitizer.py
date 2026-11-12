from __future__ import annotations

import re
import unicodedata

from backend.schemas import WriterOutput

_VIETNAMESE_REPHRASES = {
    "tối ưu hóa": "optimization",
    "tối ưu": "optimization",
    "trên": "on",
    "thiết bị nhúng": "embedded devices",
    "edge": "edge",
    "mô hình": "model",
    "phát hiện": "detection",
    "gian lận": "fraud",
    "tài chính": "financial",
    "dùng": "using",
    "sử dụng": "using",
    "học sâu": "deep learning",
    "mạng nơ-ron đồ thị": "graph neural networks",
    "đồ thị": "graph",
    "năng lượng": "energy",
    "hệ thống": "systems",
    "cho": "for",
}

_SHORT_TOPIC_EXPANSIONS = {
    "pruning": "Dynamic Pruning Methods for Resource-Constrained Inference",
    "quantization": "Adaptive Quantization Methods for Efficient Neural Inference",
    "transformer": "Efficient Transformer Architectures for Resource-Constrained Inference",
    "attention": "Adaptive Attention Mechanisms for Efficient Neural Inference",
    "classification": "Robust Neural Classification under Resource Constraints",
    "fraud": "Deep Learning Methods for Financial Fraud Detection",
}

_CODE_SPAN_RE = re.compile(r"(```[\s\S]*?```|`[^`\n]+`)")
_HEADING_RE = re.compile(
    r"^(\s*)(?:###\s+|####\s+|\*\*)?(\d+\.\d+(?:\.\d+)?\s+[A-Za-z][^\n*]+|\bAppendix\s+[A-Z]:\s+[^\n*]+)(?:\*\*)?\s*$",
    re.MULTILINE,
)


def sanitize_academic_topic(topic: str) -> str:
    """Return a safe, readable topic while preserving meaning and Unicode."""
    value = unicodedata.normalize("NFKC", str(topic or ""))
    value = re.sub(r"[\x00-\x1f\x7f]", " ", value)
    value = value.replace("<", " ").replace(">", " ")
    value = re.sub(r"[`$#%&_{}\\]", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" .,:;|")
    lowered = value.casefold()
    for source, target in sorted(_VIETNAMESE_REPHRASES.items(), key=lambda item: -len(item[0])):
        value = re.sub(re.escape(source), target, value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value.split()) < 3:
        value = _SHORT_TOPIC_EXPANSIONS.get(
            lowered,
            f"{value.title()} Methods for Resource-Constrained Research and Deployment",
        )
    return value[:240].strip(" .,:;")


def sanitize_math_placeholders(text: str, replacement: str = r"\text{[công thức toán chưa được sinh]}") -> str:
    """Replace raw placeholder strings so broken dummy labels never appear in manuscripts."""
    patterns = (
        r"\[mathematical\s+formulation\]",
        r"\[mathematical\s+expression\]",
        r"\[insert\s+equation\]",
        r"\[equation\s+here\]",
        r"\[formula\]",
    )
    value = text
    for pattern in patterns:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    return value


def strip_math_delimiters(formula: str) -> str:
    """Strip outer math delimiters ($...$, $$...$$, \\[...\\], \\(...\\)) for LaTeX blocks."""
    s = str(formula).strip()
    changed = True
    while changed and s:
        changed = False
        for left, right in (("$$", "$$"), ("$", "$"), (r"\[", r"\]"), (r"\(", r"\)")):
            if len(s) > len(left) + len(right) and s.startswith(left) and s.endswith(right):
                s = s[len(left) : -len(right)].strip()
                changed = True
                break
    return s


def unwrap_math_code_blocks(text: str) -> str:
    """Unwrap accidental code fences and de-indent lines containing LaTeX math for KaTeX rendering."""
    def repl_fence(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        has_math = any(tok in content for tok in ("$", r"\(", r"\[", r"\cdot", r"\frac", r"\sum", r"\mathbf"))
        is_code = bool(
            re.search(
                r"(\bdef\b|\bclass\b|\bimport\b|\breturn\b|\bwhile\b|\bfor\b|\bassert\b|====|----|Input\s*:|Output\s*:|torch\.|nn\.)",
                content,
            )
        )
        if has_math and not is_code:
            return f"\n\n{content}\n\n"
        return match.group(0)

    fenced_re = re.compile(r"```(?:[a-zA-Z0-9_-]+[ \t]*\r?\n)?([\s\S]*?)```")
    text = fenced_re.sub(repl_fence, text)
    text = re.sub(r"`(\$[^`\n]+\$)`", r"\1", text)

    in_code_block = False
    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            cleaned_lines.append(line)
            continue
        if not in_code_block:
            match_indent = re.match(r"^(\s{4,}|\t+)(.*)", line)
            if match_indent:
                rest = match_indent.group(2)
                has_math = any(tok in rest for tok in ("$", r"\(", r"\[", r"\cdot", r"\frac", r"\sum", r"\mathbf"))
                is_code = bool(re.search(r"(\bdef\b|\bclass\b|\bimport\b|\breturn\b|\bwhile\b|\bfor\b)", rest))
                if has_math and not is_code:
                    cleaned_lines.append(rest)
                    continue
        cleaned_lines.append(line)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines))


def normalize_markdown_headings(text: str) -> str:
    """Ensure standalone subsection headings like 3.1, 3.2 use Markdown ### prefix."""
    def repl(match: re.Match[str]) -> str:
        return f"\n\n### {match.group(2).strip()}\n\n"

    parts = _CODE_SPAN_RE.split(str(text))
    for idx in range(0, len(parts), 2):
        parts[idx] = _HEADING_RE.sub(repl, parts[idx])
    return re.sub(r"\n{3,}", "\n\n", "".join(parts))


def normalize_markdown_math(text: str) -> str:
    """Convert \\(...\\) and \\[...\\] outside code blocks to $...$ and $$...$$ and normalize headings."""
    text = normalize_markdown_headings(unwrap_math_code_blocks(str(text)))
    parts = _CODE_SPAN_RE.split(text)
    for idx in range(0, len(parts), 2):
        seg = parts[idx]
        seg = re.sub(r"\\\[\s*(.*?)\s*\\\]", lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n", seg, flags=re.DOTALL)
        seg = re.sub(r"\\\(\s*(.*?)\s*\\\)", lambda m: f"${m.group(1).strip()}$", seg, flags=re.DOTALL)
        parts[idx] = seg
    return "".join(parts)


def sanitize_writer_output(writer: WriterOutput) -> WriterOutput:
    """Sanitize placeholders and normalize Markdown math across all prose fields of WriterOutput."""
    text_fields = (
        "abstract",
        "introduction",
        "related_work",
        "methodology",
        "experimental_results",
        "discussion",
        "conclusion",
    )
    data = writer.model_dump()
    for field in text_fields:
        value = data.get(field, "")
        if isinstance(value, str):
            data[field] = normalize_markdown_math(sanitize_math_placeholders(value))
    formulas = data.get("mathematical_formulas") or []
    data["mathematical_formulas"] = [strip_math_delimiters(f) for f in formulas if strip_math_delimiters(f)] or ["y = f(x)"]
    return WriterOutput(**data)

