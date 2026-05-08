import os, csv, re
import numpy as np

models_dir = 'models'
pattern = re.compile(r'deep_(\w+)_(\w+)_seed(\d+)')

results = {}  # (llm, ticker) -> list of returns

for d in sorted(os.listdir(models_dir)):
    m = pattern.match(d)
    if not m:
        continue
    llm, ticker, seed = m.group(1), m.group(2), int(m.group(3))
    
    av_path = os.path.join(models_dir, d, 'account_value.csv')
    if not os.path.exists(av_path):
        continue
    
    with open(av_path) as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if len(rows) < 2:
        continue
    
    header = rows[0]
    val_idx = None
    for i, h in enumerate(header):
        if 'account_value' in h.lower() or 'portfolio' in h.lower():
            val_idx = i
            break
    if val_idx is None:
        val_idx = len(header) - 1
    
    try:
        first_val = float(rows[1][val_idx])
        last_val = float(rows[-1][val_idx])
        ret = (last_val - first_val) / first_val
    except:
        continue
    
    key = (llm, ticker)
    if key not in results:
        results[key] = []
    results[key].append({'ret': ret, 'seed': seed})

configs = ['gemini', 'qwen_base', 'qwen_qlora', 'qwen_qalora', 'no_nlp']
tickers = ['AAPL', 'AMZN', 'CRM', 'MSFT', 'NFLX']

print("=" * 90)
print("INDIVIDUAL AGENT MEAN RETURNS (active agents: |return| > 0.1%)")
print("=" * 90)
print(f"{'Config':<15} {'AAPL':>8} {'AMZN':>8} {'CRM':>8} {'MSFT':>8} {'NFLX':>8} {'Mean':>8}")
indiv_means = {}
for cfg in configs:
    row = []
    for t in tickers:
        rets = results.get((cfg, t), [])
        active = [r['ret'] for r in rets if abs(r['ret']) > 0.001]
        if active:
            mean_val = np.mean(active)
        else:
            mean_val = 0.0
        indiv_means[(cfg, t)] = mean_val
        row.append(mean_val)
    mean_all = np.mean(row)
    print(f"{cfg:<15} {row[0]*100:>+8.1f}% {row[1]*100:>+8.1f}% {row[2]*100:>+8.1f}% {row[3]*100:>+8.1f}% {row[4]*100:>+8.1f}% {mean_all*100:>+8.1f}%")

print()
print("=" * 90)
print("ENSEMBLE RETURNS (from ensemble_results.csv)")
print("=" * 90)
ens = {}
with open('ensemble_results.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['llm_config'], row['ticker'])
        ens[key] = float(row['total_return'])

print(f"{'Config':<15} {'AAPL':>8} {'AMZN':>8} {'CRM':>8} {'MSFT':>8} {'NFLX':>8} {'Mean':>8}")
ens_means = {}
for cfg in configs:
    row = [ens.get((cfg, t), 0) for t in tickers]
    mean_all = np.mean(row)
    ens_means[cfg] = mean_all
    for i, t in enumerate(tickers):
        ens_means[(cfg, t)] = row[i]
    print(f"{cfg:<15} {row[0]*100:>+8.1f}% {row[1]*100:>+8.1f}% {row[2]*100:>+8.1f}% {row[3]*100:>+8.1f}% {row[4]*100:>+8.1f}% {mean_all*100:>+8.1f}%")

print()
print("=" * 90)
print("GAP: Ensemble return - Individual agent mean return (pp)")
print("=" * 90)
print(f"{'Config':<15} {'AAPL':>8} {'AMZN':>8} {'CRM':>8} {'MSFT':>8} {'NFLX':>8} {'Mean':>8}")
for cfg in configs:
    gaps = []
    for t in tickers:
        gap = (ens_means[(cfg, t)] - indiv_means[(cfg, t)]) * 100
        gaps.append(gap)
    mean_gap = np.mean(gaps)
    print(f"{cfg:<15} {gaps[0]:>+8.1f}pp {gaps[1]:>+8.1f}pp {gaps[2]:>+8.1f}pp {gaps[3]:>+8.1f}pp {gaps[4]:>+8.1f}pp {mean_gap:>+8.1f}pp")

print()
print("=" * 90)
print("KEY COMPARISON: LLM vs No-NLP")
print("=" * 90)

# Individual level: LLM mean - No-NLP mean per ticker
print("\nIndividual Agent Mean: LLM - No-NLP (pp):")
print(f"{'Config':<15} {'AAPL':>8} {'AMZN':>8} {'CRM':>8} {'MSFT':>8} {'NFLX':>8} {'Mean':>8}")
for cfg in configs[:-1]:  # skip no_nlp
    diffs = []
    for t in tickers:
        diff = (indiv_means[(cfg, t)] - indiv_means[('no_nlp', t)]) * 100
        diffs.append(diff)
    mean_diff = np.mean(diffs)
    print(f"{cfg:<15} {diffs[0]:>+8.1f}pp {diffs[1]:>+8.1f}pp {diffs[2]:>+8.1f}pp {diffs[3]:>+8.1f}pp {diffs[4]:>+8.1f}pp {mean_diff:>+8.1f}pp")

print("\nEnsemble: LLM - No-NLP (pp):")
print(f"{'Config':<15} {'AAPL':>8} {'AMZN':>8} {'CRM':>8} {'MSFT':>8} {'NFLX':>8} {'Mean':>8}")
for cfg in configs[:-1]:
    diffs = []
    for t in tickers:
        diff = (ens_means[(cfg, t)] - ens_means[('no_nlp', t)]) * 100
        diffs.append(diff)
    mean_diff = np.mean(diffs)
    print(f"{cfg:<15} {diffs[0]:>+8.1f}pp {diffs[1]:>+8.1f}pp {diffs[2]:>+8.1f}pp {diffs[3]:>+8.1f}pp {diffs[4]:>+8.1f}pp {mean_diff:>+8.1f}pp")

print()
print("=" * 90)
print("AMPLIFICATION: Ensemble gap / Individual gap")
print("=" * 90)
print(f"{'Config':<15} {'AAPL':>8} {'AMZN':>8} {'CRM':>8} {'MSFT':>8} {'NFLX':>8}")
for cfg in configs[:-1]:
    amps = []
    for t in tickers:
        indiv_diff = (indiv_means[(cfg, t)] - indiv_means[('no_nlp', t)]) * 100
        ens_diff = (ens_means[(cfg, t)] - ens_means[('no_nlp', t)]) * 100
        if abs(indiv_diff) > 0.1:
            amps.append(ens_diff / indiv_diff)
        else:
            amps.append(float('nan'))
    print(f"{cfg:<15} {amps[0]:>8.1f}x {amps[1]:>8.1f}x {amps[2]:>8.1f}x {amps[3]:>8.1f}x {amps[4]:>8.1f}x")

# Also count active agents per config
print()
print("=" * 90)
print("ACTIVE AGENT COUNTS per (config, ticker)")
print("=" * 90)
print(f"{'Config':<15} {'AAPL':>8} {'AMZN':>8} {'CRM':>8} {'MSFT':>8} {'NFLX':>8} {'Total':>8}")
for cfg in configs:
    row = []
    for t in tickers:
        rets = results.get((cfg, t), [])
        active = [r for r in rets if abs(r['ret']) > 0.001]
        row.append(len(active))
    print(f"{cfg:<15} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>8} {sum(row):>8}")
