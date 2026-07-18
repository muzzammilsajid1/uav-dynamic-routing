import re

log_path = r'C:\Users\Fast Computer\.gemini\antigravity\brain\c09ef384-b85e-4fad-864c-9336cf198985\.system_generated\tasks\task-207.log'
with open(log_path, 'r', encoding='utf-8') as f:
    data = f.read()

blocks = re.split(r'----------------------------------------', data)
steps = []
for b in blocks:
    m_ts = re.search(r'total_timesteps\s+\|\s+(\d+)', b)
    m_rew = re.search(r'mean_reward\s+\|\s+([-\d\.]+)', b)
    if m_ts and m_rew:
        steps.append({'ts': int(m_ts.group(1)), 'rew': float(m_rew.group(1))})

if not steps:
    print("No data parsed!")
else:
    steps.sort(key=lambda x: x['ts'])
    targets = [0, 100000, 200000, 300000, 400000, 500000]
    res = [min(steps, key=lambda x: abs(x['ts'] - t)) for t in targets]
    for r in res:
        print(f"TS: {r['ts']:<8} | Mean Reward: {r['rew']}")
