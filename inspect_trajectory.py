import json
import sys
import csv
from pathlib import Path

LOG_PATH = "docs/failure_trajectory_manual_review.csv"

def inspect(pilot, scenario_id):
    path = f"raw_temp/{pilot}/step_100000/trajectories/{scenario_id}.json"
    with open(path) as f:
        t = json.load(f)
    s = t['summary']
    traj = t['trajectory']
    print(f"=== {pilot} / {scenario_id} ===")
    print(f"Automated label: {s['failure_label']}")
    print(f"Grid: {s['grid_size']} | Family: {s['scenario_family']} | Route: {s['route_bucket']}")
    print(f"Success: {s['success']} | Timeout: {s['timeout']} | Collision: {s['collision']}")
    print(f"Decisions: {s['decisions']} | A* cost: {s['fresh_astar_cost']:.2f}")
    print(f"First 10: {traj[:10]}")
    print(f"Last 10: {traj[-10:]}")
    unique_cells = len(set(tuple(p) for p in traj))
    print(f"Unique cells visited: {unique_cells} out of {len(traj)} steps")
    print()
    print("Now classify it. Enter your manual label (or press Enter to accept the automated one):")
    print("  1=two_cell_oscillation 2=longer_loop 3=aimless_movement 4=goal_then_failure")
    print("  5=blocked_corridor 6=poor_route_choice 7=timeout_despite_progress 8=obs_action_anomaly")
    choice = input("Your label (1-8, or Enter to agree): ").strip()
    labels = {
        "1": "two_cell_oscillation", "2": "longer_loop", "3": "aimless_movement",
        "4": "goal_then_failure", "5": "blocked_corridor", "6": "poor_route_choice",
        "7": "timeout_despite_progress", "8": "obs_action_anomaly"
    }
    manual_label = labels.get(choice, s['failure_label'])
    stuck_from_start = input("Stuck from step 1, or progress-then-stuck? (start/progress): ").strip()
    notes = input("Any other notes (optional): ").strip()

    row = {
        "pilot": pilot, "scenario_id": scenario_id, "automated_label": s['failure_label'],
        "manual_label": manual_label, "agree": manual_label == s['failure_label'],
        "stuck_pattern": stuck_from_start, "grid_size": s['grid_size'],
        "family": s['scenario_family'], "route_bucket": s['route_bucket'],
        "notes": notes
    }
    file_exists = Path(LOG_PATH).exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Logged to {LOG_PATH}")

if __name__ == "__main__":
    inspect(sys.argv[1], sys.argv[2])
    