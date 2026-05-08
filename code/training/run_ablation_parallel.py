#!/usr/bin/env python3
"""
Parallel Ablation Launcher
Spawns 6 independent Python processes for NLP-only ablation training.
"""
import subprocess, sys, os, time
from pathlib import Path

OUR_CODE = Path("/Users/BryanMak/Documents/STAT8307 Natural Language Processing and Text Analytics/project/our_code")
SCRIPT = OUR_CODE / "ppo_training" / "run_ablation_nlp_only.py"
OUTPUT_DIR = OUR_CODE / "ablation_nlp_only_results"

LLM_CONFIGS = ["gemini","qwen_base","qwen_qlora","qwen_qalora"]
TICKERS = ["AAPL","AMZN","CRM","MSFT","NFLX"]
SEEDS = [42,123,456,789,2024,314,271,1618,999,7,
         100,200,300,400,500,600,700,800,900,1000,
         1111,2222,3333,4444,5555,6666,7777,8888,9999,12345]

MAX_WORKERS = 6

def main():
    # Build task list
    tasks = []
    for llm_config in LLM_CONFIGS:
        for ticker in TICKERS:
            for seed in SEEDS:
                run_name = f"nlponly_{llm_config}_{ticker}_seed{seed}"
                if not (OUTPUT_DIR / "models" / run_name / "metrics.json").exists():
                    tasks.append((llm_config, ticker, seed))
    
    total = len(LLM_CONFIGS) * len(TICKERS) * len(SEEDS)
    done = total - len(tasks)
    print(f"Total: {total}, Already done: {done}, Remaining: {len(tasks)}")
    print(f"Running with {MAX_WORKERS} parallel workers")
    print(f"Estimated time: ~{len(tasks) * 60 / MAX_WORKERS / 60:.0f} minutes")
    
    running = []  # List of (subprocess, task, start_time)
    task_idx = 0
    n_ok = 0
    n_err = 0
    
    while task_idx < len(tasks) or running:
        # Fill up to MAX_WORKERS
        while len(running) < MAX_WORKERS and task_idx < len(tasks):
            llm_config, ticker, seed = tasks[task_idx]
            cmd = [
                sys.executable, str(SCRIPT),
                "--mode", "train",
                "--llm", llm_config,
                "--ticker", ticker,
                "--seed", str(seed),
            ]
            t0 = time.time()
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=str(OUR_CODE / "ppo_training"),
            )
            running.append((proc, tasks[task_idx], t0))
            task_idx += 1
        
        # Check for completed processes
        still_running = []
        for proc, task, t0 in running:
            ret = proc.poll()
            if ret is not None:
                elapsed = time.time() - t0
                llm_config, ticker, seed = task
                run_name = f"nlponly_{llm_config}_{ticker}_seed{seed}"
                if ret == 0:
                    n_ok += 1
                    # Read result from metrics
                    mp = OUTPUT_DIR / "models" / run_name / "metrics.json"
                    if mp.exists():
                        import json
                        with open(mp) as f:
                            m = json.load(f)
                        ret_str = f"{m.get('total_return',0)*100:+.1f}%"
                        print(f"  OK {run_name}: {ret_str} ({elapsed:.0f}s)")
                    else:
                        print(f"  OK {run_name} ({elapsed:.0f}s)")
                else:
                    n_err += 1
                    stderr = proc.stderr.read().decode()[-300:] if proc.stderr else ""
                    print(f"  ERR {run_name}: {stderr}")
            else:
                # Check timeout (5 min)
                if time.time() - t0 > 300:
                    proc.kill()
                    n_err += 1
                    print(f"  TIMEOUT {run_name}")
                else:
                    still_running.append((proc, task, t0))
        
        running = still_running
        if running or task_idx < len(tasks):
            time.sleep(2)
    
    print(f"\nCOMPLETE: {n_ok} ok, {n_err} err, {done} skipped = {n_ok + n_err + done} total")

if __name__ == "__main__":
    main()
