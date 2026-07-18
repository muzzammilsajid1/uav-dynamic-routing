"""
rebuild_colab_cell2.py
Replaces Cell 2 of train_her_colab_v2.py with a %%writefile magic cell
that embeds the exact literal content of rl_agent/uav_env.py, avoiding
all string-escaping and encoding issues.
"""
import sys
import os
import difflib

sys.stdout.reconfigure(encoding="utf-8")

# Read real uav_env.py
with open("rl_agent/uav_env.py", encoding="utf-8") as f:
    real_uav_env = f.read()

# Read current Colab script
with open("train_her_colab_v2.py", encoding="utf-8") as f:
    colab = f.read()

# Locate the exact boundaries of Cell 2 and Cell 3
c2_marker = "# %% ---------------------------------------------------------------------------\n# CELL 2:"
c3_marker = "# %% ---------------------------------------------------------------------------\n# CELL 3:"

cell2_start = colab.index(c2_marker)
cell2_end   = colab.index(c3_marker)

# Build replacement Cell 2 using %%writefile magic.
# The %%writefile line is the first line of the Colab cell.
# Colab writes everything after that line verbatim to disk,
# so no string quoting, no escaping, and no Unicode corruption.
cell2_new = "\n".join([
    "# %% ---------------------------------------------------------------------------",
    "# CELL 2: Write uav_env.py to /content/uav_env.py",
    "#",
    "# HOW TO USE IN COLAB:",
    "#   Paste this entire block as a single Colab code cell.",
    "#   The %%writefile magic on the FIRST LINE causes Colab to write",
    "#   the rest of the cell verbatim to disk with zero string-escaping.",
    "# ------------------------------------------------------------------------------",
    "",
    "# %%writefile /content/uav_env.py",
]) + "\n" + real_uav_env + "\n\n"

new_colab = colab[:cell2_start] + cell2_new + colab[cell2_end:]

with open("train_her_colab_v2.py", "w", encoding="utf-8") as f:
    f.write(new_colab)

print("Rebuilt Cell 2 using %%writefile magic.")
print(f"train_her_colab_v2.py is now {len(new_colab)} bytes.")

# -----------------------------------------------------------------------
# Now verify: simulate what Colab's %%writefile would write to disk,
# and diff it against the real uav_env.py
# -----------------------------------------------------------------------
print()
print("Verifying embedded content against real uav_env.py ...")

# After %%writefile, Colab writes everything AFTER the first line
# (i.e., after "# %%writefile /content/uav_env.py\n")
# Locate that section in the rebuilt file
with open("train_her_colab_v2.py", encoding="utf-8") as f:
    rebuilt = f.read()

writefile_marker = "# %%writefile /content/uav_env.py\n"
wf_start = rebuilt.index(writefile_marker) + len(writefile_marker)
# The embedded content ends at the next cell marker
next_cell = rebuilt.index("\n# %% ---", wf_start)
embedded_content = rebuilt[wf_start:next_cell].rstrip("\n") + "\n"

real_lines     = real_uav_env.splitlines(keepends=True)
embedded_lines = embedded_content.splitlines(keepends=True)

diff = list(difflib.unified_diff(
    real_lines,
    embedded_lines,
    fromfile="rl_agent/uav_env.py  (real)",
    tofile="train_her_colab_v2.py  (%%writefile content)",
))

if diff:
    print(f"DIFFERENCES FOUND ({len(diff)} lines):\n")
    print("".join(diff))
else:
    print("ZERO DIFFERENCES. %%writefile content is byte-for-byte identical to real uav_env.py.")
