"""
verify_embedding.py
Extracts UAV_ENV_CODE from train_her_colab_v2.py, writes it to a temp file,
and diffs it against the real rl_agent/uav_env.py.
"""
import sys
import os
import difflib

# --- Step 1: Extract UAV_ENV_CODE from the Colab script ---
# We exec() just the assignment statement so we don't need to parse manually.
colab_script = open("train_her_colab_v2.py", encoding="utf-8").read()

# Find just the UAV_ENV_CODE = r'''...''' block
start = colab_script.index("UAV_ENV_CODE = r'''")
end   = colab_script.index("'''\n\nwith open", start) + 3
assignment = colab_script[start:end]

local_ns = {}
exec(assignment, local_ns)
embedded_code = local_ns["UAV_ENV_CODE"]

# --- Step 2: Write embedded code to temp file ---
temp_path = "temp_embedded_uav_env.py"
with open(temp_path, "w", encoding="utf-8") as f:
    f.write(embedded_code)

# --- Step 3: Read the real uav_env.py ---
real_path = os.path.join("rl_agent", "uav_env.py")
real_code  = open(real_path, encoding="utf-8").read()

# --- Step 4: Unified diff ---
embedded_lines = embedded_code.splitlines(keepends=True)
real_lines     = real_code.splitlines(keepends=True)

diff = list(difflib.unified_diff(
    real_lines,
    embedded_lines,
    fromfile="rl_agent/uav_env.py  (real)",
    tofile="temp_embedded_uav_env.py  (from Colab string)",
))

if diff:
    print(f"DIFFERENCES FOUND ({len(diff)} diff lines):\n")
sys.stdout.reconfigure(encoding="utf-8")
print("".join(diff))
else:
    print("ZERO DIFFERENCES. Embedded code is byte-for-byte identical to real uav_env.py.")

# Cleanup
os.remove(temp_path)
