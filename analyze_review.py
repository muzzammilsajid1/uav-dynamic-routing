import csv
from collections import Counter

with open('docs/failure_trajectory_manual_review.csv') as f:
    rows = list(csv.DictReader(f))

print('=== Disagreements by pilot ===')
print(Counter(r['pilot'] for r in rows if r['agree'] != 'True'))
print()
print('=== Disagreements by family ===')
print(Counter(r['family'] for r in rows if r['agree'] != 'True'))
print()
print('=== Disagreements by grid_size ===')
print(Counter(r['grid_size'] for r in rows if r['agree'] != 'True'))
print()
print('=== stuck_pattern breakdown (all 30) ===')
print(Counter(r['stuck_pattern'] for r in rows))
print()
print('=== Most common manual_label given ===')
print(Counter(r['manual_label'] for r in rows))
print()
print('=== Most common automated_label ===')
print(Counter(r['automated_label'] for r in rows))
print()
print('=== Full disagreement list ===')
for r in rows:
    if r['agree'] != 'True':
        print(f"{r['pilot']:3} {r['scenario_id']:32} auto={r['automated_label']:20} -> manual={r['manual_label']:20} ({r['stuck_pattern']})")
        