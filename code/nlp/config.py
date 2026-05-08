import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, "train_data")
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "benchmarks_output")

# Benchmarking Settings (Exact as per Proposal)
MODELS = {
    "gemini": "models/gemini-3.1-pro", # Baseline: Gemini 3.1 Pro
    "qwen_base": "Qwen/Qwen3.5-27B", # Baseline: Qwen3.5-27B (includes instruct tuning)
    "qwen_qlora": "qwen_qlora_adapter", # Fine-tuned: 4-bit QLoRA
    "qwen_qalora": "qwen_qalora_adapter" # Fine-tuned: 4-bit QA-LoRA
}

# The 7 Structured Financial Signals
SIGNAL_SCHEMA = [
    "news_relevance", "sentiment", "price_impact_potential", 
    "trend_direction", "earnings_impact", 
    "investor_confidence", "risk_profile_change"
]
