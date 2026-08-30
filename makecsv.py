import json
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "v5_output/phrases_output.json"
OUTPUT_FILE = "heatmap_data.csv"


# ============================================================
# LOAD JSON
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


# ============================================================
# FIND ALL STUDENTS AND WEEKS
# ============================================================

students = {}
weeks = set()

for reflection in data:

    sid = reflection.get("sid")
    week = reflection.get("week")

    if sid is None or week is None:
        continue

    weeks.add(week)

    # Store student information
    if sid not in students:
        students[sid] = {
            "sid": sid,
            "name": None,
        }

    # Try to get the student's name from the filename
    # or reflection text if needed later.
    students[sid]["name"] = reflection.get("filename", sid)


# Sort weeks numerically:
# Week 1, Week 2, ..., Week 10
def week_number(week):
    import re

    match = re.search(r"(\d+)", week)

    if match:
        return int(match.group(1))

    return 999


weeks = sorted(weeks, key=week_number)


# ============================================================
# CREATE STUDENT × WEEK MATRIX
# ============================================================

rows = []

for sid in students:

    row = {}

    for week in weeks:
        row[week] = "No submission"

    row["SID"] = sid

    rows.append(row)


df = pd.DataFrame(rows)

df = df.set_index("SID")


# ============================================================
# INSERT EMOTIONAL RESPONSE SCORES
# ============================================================

for reflection in data:

    sid = reflection.get("sid")
    week = reflection.get("week")

    if sid is None or week is None:
        continue

    # Get emotional score
    scores = reflection.get("scores", {})

    emotional = scores.get("emotional")

    if emotional is None:
        emotional = "Missing"

    # Make sure the student exists
    if sid not in df.index:
        continue

    # Store emotional response
    df.loc[sid, week] = emotional


# ============================================================
# ORDER COLUMNS
# ============================================================

df = df[weeks]


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(OUTPUT_FILE)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("Done!")
print(f"Created: {OUTPUT_FILE}")
print()

print(f"Students: {len(df)}")
print(f"Weeks: {len(weeks)}")
print()

print("Weeks:")
print(weeks)

print()
print("Emotional response counts:")
print()

for week in weeks:

    print(f"--- {week} ---")

    print(
        df[week]
        .value_counts()
        .to_string()
    )

    print()


# ============================================================
# DISPLAY FIRST FEW ROWS
# ============================================================

print("First few rows:")
print()

print(df.head())