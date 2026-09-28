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
    ### YOUR CODE HERE
    import os
    try:
        from langchain_deepseek import ChatDeepSeek
        llm = ChatDeepSeek(
            model="deepseek-v4-flash-vision-exp",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            temperature=0.0,
        )
    except (ImportError, Exception):
        # 兼容备用方案：如果未安装 langchain-deepseek，走标准的 ChatOpenAI 接口调用 DeepSeek
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="deepseek-v4-flash-vision-exp",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0.0,
        )
    return llm


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    from langchain_core.messages import HumanMessage

    system_instruction = (
        "You are an expert financial assistant analyzing a supermarket receipt image.\n"
        "Extract two numbers from this receipt:\n"
        "1. paid: The final net amount actually paid/charged (OCTOPUS, VISA, CASH, etc., AFTER ROUNDING/抹零).\n"
        "2. undiscounted: The original amount before discounts. Calculated as: SUBTOTAL plus all discount/promotion/coupon lines added back as positive values. Do NOT add back ROUNDING.\n\n"
        "Respond ONLY with a valid JSON object in this exact format, with no extra text or markdown:\n"
        "{\"paid\": 102.30, \"undiscounted\": 107.70}"
    )

    batch_messages = []
    for image_path in images:
        data_url = image_data_url(image_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": system_instruction},
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
            ]
        )
        batch_messages.append([message])

    # 使用 batch 并行/批量处理所有小票图片
    results = chain.batch(batch_messages)

    total_paid = Decimal("0.00")
    total_undiscounted = Decimal("0.00")

    for i, res in enumerate(results):
        content = getattr(res, "content", str(res)).strip()
        # 清理可能存在的 markdown 代码块标记 ```json ... ```
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        try:
            parsed = json.loads(content)
            paid_val = Decimal(str(parsed.get("paid", 0))).quantize(Decimal("0.01"))
            undisc_val = Decimal(str(parsed.get("undiscounted", 0))).quantize(Decimal("0.01"))
        except Exception:
            # 如果 json 解析失败，尝试正则提取数字容错
            amounts = re.findall(r"\d+(?:\.\d+)?", content)
            paid_val = Decimal(amounts[0]) if amounts else Decimal("0.00")
            undisc_val = Decimal(amounts[1]) if len(amounts) > 1 else paid_val

        total_paid += paid_val
        total_undiscounted += undisc_val

    # 必须保证最终返回的字符串只包含一个合法的金额数字，如 "HK$1974.30"
    return {
        QUERY_1: f"HK${total_paid:.2f}",
        QUERY_2: f"HK${total_undiscounted:.2f}",
    }


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
