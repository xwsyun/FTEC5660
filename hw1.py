#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    instructions = (
        "You are reading a supermarket receipt (prices in HKD). "
        "Work through the receipt line by line from top to bottom and extract "
        "exactly three things. Read every amount digit by digit.\n\n"
        '1. "final_payment": the amount the customer actually paid, i.e. the total '
        "AFTER the rounding adjustment. It is usually on the payment line "
        "(e.g. OCTOPUS, CASH, VISA, EPS) or labelled as the total after ROUNDING. "
        "Example: if the receipt shows ROUNDING -$0.01 and OCTOPUS $102.30, "
        'then "final_payment" is 102.30.\n\n'
        '2. "subtotal": the SUBTOTAL line amount (after all discounts, before rounding).\n\n'
        '3. "discounts": a JSON array with the amount of EVERY discount as a POSITIVE '
        "number (strip any minus sign). Collect EVERY negative amount on the receipt, "
        "EXCEPT the ROUNDING line. Discounts often have keywords (OFF, DISCOUNT, PROMO, "
        'COUPON, MEMBER, SAVE - e.g. "Buy 2 Save $36" gives -36.00, "5% OFF" gives '
        "-5.39), but some are just a bare negative amount next to an item with no "
        "keyword at all (e.g. a damaged-packaging markdown like -11.00) - count those "
        "too. Go line by line and do not skip any negative amount except ROUNDING. "
        "If there is no discount line, use an empty array [].\n\n"
        "Rules:\n"
        "- Output ONLY one valid JSON object and nothing else: no markdown fences, "
        "no explanation, no extra text.\n"
        '- Exact format: {{"final_payment": 102.30, "subtotal": 102.31, "discounts": [5.39]}}\n'
        "- Numbers only: no currency symbols, no commas.\n"
        "- Double-check each number against the receipt image before answering."
    )

    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                [
                    {"type": "image_url", "image_url": {"url": "{image}"}},
                    {"type": "text", "text": instructions},
                ],
            )
        ]
    )
    return prompt | model


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    import json
    import re
    from decimal import Decimal, InvalidOperation

    def to_decimal(value: Any) -> Decimal | None:
        try:
            return Decimal(str(value).replace(",", "").strip())
        except (InvalidOperation, ValueError, TypeError):
            return None

    def extract_json(text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def content_text(result: Any) -> str:
        content = getattr(result, "content", result)
        return content if isinstance(content, str) else str(content)

    def receipt_values(data: dict[str, Any]) -> tuple[Decimal, Decimal]:
        """Return (final_payment, amount_without_discount) for one extraction."""
        final_payment = to_decimal(data.get("final_payment")) or Decimal("0")
        subtotal = to_decimal(data.get("subtotal")) or Decimal("0")
        raw_discounts = data.get("discounts") or []
        if not isinstance(raw_discounts, list):
            raw_discounts = [raw_discounts]
        discount_total = sum(
            (abs(amount) for d in raw_discounts if (amount := to_decimal(d)) is not None),
            Decimal("0"),
        )
        return final_payment, subtotal + discount_total

    def median(values: list[Decimal]) -> Decimal:
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    # Sample each receipt multiple times and take the median: a single
    # misread line then cannot swing the total.
    SAMPLES_PER_RECEIPT = 5
    inputs: list[dict[str, str]] = []
    for path in images:
        data_url = image_data_url(path)
        inputs.extend([{"image": data_url} for _ in range(SAMPLES_PER_RECEIPT)])
    results = chain.batch(inputs)

    total_paid = Decimal("0")
    total_without_discount = Decimal("0")
    for i in range(len(images)):
        samples = results[i * SAMPLES_PER_RECEIPT : (i + 1) * SAMPLES_PER_RECEIPT]
        paids: list[Decimal] = []
        without_discounts: list[Decimal] = []
        for result in samples:
            paid, without_discount = receipt_values(extract_json(content_text(result)))
            paids.append(paid)
            without_discounts.append(without_discount)
        total_paid += median(paids)
        total_without_discount += median(without_discounts)

    def format_hkd(amount: Decimal) -> str:
        return f"HK${amount.quantize(Decimal('0.01')):.2f}"

    return {QUERY_1: format_hkd(total_paid), QUERY_2: format_hkd(total_without_discount)}


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
