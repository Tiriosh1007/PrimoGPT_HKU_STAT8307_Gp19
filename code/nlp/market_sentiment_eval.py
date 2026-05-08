from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TARGET_COLUMNS = {
    "news_relevance": "News Relevance",
    "sentiment": "Sentiment",
    "price_impact_potential": "Price Impact Potential",
    "trend_direction": "Trend Direction",
    "earnings_impact": "Earnings Impact",
    "investor_confidence": "Investor Confidence",
    "risk_profile_change": "Risk Profile Change",
}


@dataclass(slots=True)
class Record:
    row_id: int
    date: str
    prompt: str
    targets: dict[str, int]


class BatchRequestError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate market-sentiment prompts against labeled targets via OpenRouter."
    )
    parser.add_argument("--csv-path", default="AAPL_data.csv", help="Path to the source CSV file.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for predictions and metrics.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of rows per LLM request.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=2,
        help="Maximum number of concurrent OpenRouter requests.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quicker runs.")
    parser.add_argument(
        "--model",
        default=None,
        help="OpenRouter model slug. Defaults to OPENROUTER_MODEL from the environment.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and batch the CSV without making API calls.",
    )
    return parser.parse_args()


def load_records(csv_path: Path, max_rows: int | None = None) -> list[Record]:
    records: list[Record] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_id, row in enumerate(reader):
            targets = {
                prediction_key: int(row[column_name])
                for prediction_key, column_name in TARGET_COLUMNS.items()
            }
            records.append(
                Record(
                    row_id=row_id,
                    date=row["Date"],
                    prompt=row["Prompt"].strip(),
                    targets=targets,
                )
            )
            if max_rows is not None and len(records) >= max_rows:
                break
    return records


def chunk_records(records: list[Record], batch_size: int) -> list[list[Record]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


def build_messages(batch: list[Record]) -> list[dict[str, str]]:
    schema = {
        "predictions": [
            {
                "row_id": 0,
                "news_relevance": 0,
                "sentiment": 0,
                "price_impact_potential": 0,
                "trend_direction": 0,
                "earnings_impact": 0,
                "investor_confidence": 0,
                "risk_profile_change": 0,
            }
        ]
    }
    system_message = (
        "You are a financial sentiment evaluator. "
        "Analyze each item independently and return only valid JSON. "
        "Each prediction field must be an integer matching the allowed label ranges from the provided prompt."
    )
    batch_payload = []
    for record in batch:
        batch_payload.append(
            {
                "row_id": record.row_id,
                "date": record.date,
                "prompt": record.prompt,
            }
        )
    user_message = (
        "Predict labels for every item below. "
        "Return one JSON object with a top-level 'predictions' array. "
        "Each array entry must include row_id and all seven prediction fields as integers. "
        f"Expected JSON shape: {json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Items:\n{json.dumps(batch_payload, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def extract_json_payload(raw_text: str) -> dict[str, Any]:
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, flags=re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("No JSON object found in model response")
    return json.loads(raw_text[start : end + 1])


def normalize_prediction(entry: dict[str, Any]) -> dict[str, int]:
    normalized: dict[str, int] = {"row_id": int(entry["row_id"])}
    for prediction_key in TARGET_COLUMNS:
        if prediction_key not in entry:
            raise KeyError(f"Missing field '{prediction_key}' in prediction entry")
        normalized[prediction_key] = int(str(entry[prediction_key]).strip())
    return normalized


def coerce_content(message_content: Any) -> str:
    if message_content is None:
        return ""
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, dict):
        text = message_content.get("text")
        return str(text) if text is not None else ""
    if isinstance(message_content, list):
        parts: list[str] = []
        for item in message_content:
            item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            item_text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if item_type == "text" and item_text is not None:
                parts.append(str(item_text))
        return "\n".join(part for part in parts if part)
    raise TypeError("Unsupported message content format returned by OpenRouter")


def log_batch_event(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


async def request_batch(
    client: AsyncOpenAI,
    model: str,
    batch: list[Record],
    temperature: float,
) -> list[dict[str, int]]:
    row_ids = [record.row_id for record in batch]
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = await client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=build_messages(batch),
                response_format={"type": "json_object"},
            )
            raw_content = coerce_content(response.choices[0].message.content)
            if not raw_content.strip():
                raise ValueError("Empty model response")
            parsed = extract_json_payload(raw_content)
            predictions = [normalize_prediction(item) for item in parsed["predictions"]]
            if not predictions:
                raise ValueError("Model response did not contain any predictions")
            return predictions
        except Exception as error:
            last_error = error
            log_batch_event(
                "WARN",
                f"Batch {row_ids} attempt {attempt}/3 failed: {error}",
            )
            if attempt == 3:
                break
            await asyncio.sleep(attempt)
    raise BatchRequestError(f"Batch {row_ids} failed after 3 attempts: {last_error}")


async def collect_predictions(
    records: list[Record],
    model: str,
    batch_size: int,
    max_concurrency: int,
    temperature: float,
) -> dict[int, dict[str, int]]:
    predictions: dict[int, dict[str, int]] = {}
    semaphore = asyncio.Semaphore(max_concurrency)
    batches = chunk_records(records, batch_size)
    default_headers: dict[str, str] = {}
    if http_referer := os.getenv("OPENROUTER_HTTP_REFERER"):
        default_headers["HTTP-Referer"] = http_referer
    if x_title := os.getenv("OPENROUTER_X_TITLE"):
        default_headers["X-Title"] = x_title

    client = AsyncOpenAI(
        api_key=get_api_key(),
        base_url=OPENROUTER_BASE_URL,
        default_headers=default_headers or None,
        timeout=120.0,
    )
    try:
        async def process(batch: list[Record]) -> None:
            async with semaphore:
                try:
                    batch_predictions = await request_batch(client, model, batch, temperature)
                except Exception as error:
                    log_batch_event("ERROR", str(error))
                    return
                for prediction in batch_predictions:
                    predictions[prediction["row_id"]] = prediction

        await asyncio.gather(*(process(batch) for batch in batches))
    finally:
        await client.close()
    return predictions


def get_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY to your OpenRouter key"
        )
    return api_key


def get_model(cli_model: str | None) -> str:
    model = cli_model or os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL")
    if not model:
        raise EnvironmentError(
            "Set OPENROUTER_MODEL or OPENAI_MODEL, or pass --model with your OpenRouter model slug"
        )
    return model


def build_row_output(record: Record, prediction: dict[str, int] | None) -> dict[str, Any]:
    matches = {
        key: prediction is not None and prediction.get(key) == record.targets[key]
        for key in TARGET_COLUMNS
    }
    return {
        "row_id": record.row_id,
        "date": record.date,
        "targets": record.targets,
        "prediction": prediction,
        "matches": matches,
        "exact_match": all(matches.values()) if prediction is not None else False,
    }


def compute_metrics(records: list[Record], predictions: dict[int, dict[str, int]]) -> dict[str, Any]:
    total_rows = len(records)
    per_field_correct = {key: 0 for key in TARGET_COLUMNS}
    predicted_rows = 0
    exact_matches = 0

    for record in records:
        prediction = predictions.get(record.row_id)
        if prediction is None:
            continue
        predicted_rows += 1
        row_exact = True
        for key in TARGET_COLUMNS:
            is_match = prediction.get(key) == record.targets[key]
            if is_match:
                per_field_correct[key] += 1
            else:
                row_exact = False
        if row_exact:
            exact_matches += 1

    field_accuracy = {
        key: (per_field_correct[key] / total_rows if total_rows else 0.0)
        for key in TARGET_COLUMNS
    }
    return {
        "total_rows": total_rows,
        "predicted_rows": predicted_rows,
        "missing_rows": total_rows - predicted_rows,
        "exact_match_accuracy": exact_matches / total_rows if total_rows else 0.0,
        "field_accuracy": field_accuracy,
    }


def write_outputs(
    output_dir: Path,
    records: list[Record],
    predictions: dict[int, dict[str, int]],
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(build_row_output(record, predictions.get(record.row_id))) + "\n")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def print_summary(metrics: dict[str, Any]) -> None:
    print(f"Processed rows: {metrics['total_rows']}")
    print(f"Predicted rows: {metrics['predicted_rows']}")
    print(f"Missing rows: {metrics['missing_rows']}")
    print(f"Exact-match accuracy: {metrics['exact_match_accuracy']:.4f}")
    for key, value in metrics["field_accuracy"].items():
        print(f"{key}: {value:.4f}")


async def async_main(args: argparse.Namespace) -> int:
    load_dotenv()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    records = load_records(csv_path, args.max_rows)
    if not records:
        raise ValueError("No records found in the CSV file")

    if args.dry_run:
        batches = chunk_records(records, args.batch_size)
        print(f"Dry run loaded {len(records)} rows into {len(batches)} batches.")
        print(f"First batch row_ids: {[record.row_id for record in batches[0]]}")
        return 0

    model = get_model(args.model)

    predictions = await collect_predictions(
        records=records,
        model=model,
        batch_size=args.batch_size,
        max_concurrency=args.max_concurrency,
        temperature=args.temperature,
    )
    metrics = compute_metrics(records, predictions)
    write_outputs(Path(args.output_dir), records, predictions, metrics)
    print_summary(metrics)
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
