import json
import pandas as pd
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "v5_output/phrases_output.json"

OUTPUT_EXCEL = "reflection_word_counts.xlsx"
OUTPUT_CSV = "reflection_word_counts.csv"


# ============================================================
# LOAD JSON
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read().strip()


# Your uploaded JSON appears to have a trailing comma /
# incomplete closing bracket, so fix that if necessary.
if text.endswith(","):
    text = text[:-1]

if not text.endswith("]"):
    text += "\n]"


data = json.loads(text)


# ============================================================
# EXTRACT WORD COUNTS
# ============================================================

rows = []

for reflection in data:

    week = reflection.get("week")
    word_count = reflection.get("word_count")

    # Ignore entries that don't contain the required fields
    if week is None or word_count is None:
        continue

    rows.append({
        "Week": week,
        "Word Count": word_count
    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(rows)


# Extract the numerical week so that Week 1, Week 2, ...,
# Week 10 are sorted correctly.
df["Week Number"] = (
    df["Week"]
    .str.extract(r"(\d+)")
    .astype(int)
)

df = df.sort_values("Week Number")


# ============================================================
# CREATE BOX-PLOT-FRIENDLY TABLE
# ============================================================

# Each week becomes a column:
#
# Week 1 | Week 2 | Week 3 | ...
# 288    | 293    | 285    | ...
# 219    |  ...   |  ...   |
#
# This format works nicely when selecting the data
# for a box plot in Excel/Word.

boxplot_data = (
    df.groupby("Week Number")["Word Count"]
    .apply(list)
)

max_count = boxplot_data.apply(len).max()

boxplot_table = pd.DataFrame({
    f"Week {week}": values + [None] * (max_count - len(values))
    for week, values in boxplot_data.items()
})


# ============================================================
# SAVE TO EXCEL
# ============================================================

with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:

    # All reflections in normal table format
    df[["Week", "Word Count"]].to_excel(
        writer,
        sheet_name="Word Counts",
        index=False
    )

    # Data arranged for box plot
    boxplot_table.to_excel(
        writer,
        sheet_name="Box Plot Data",
        index=False
    )


# ============================================================
# SAVE CSV
# ============================================================

df[["Week", "Word Count"]].to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("Done!")
print()

print(f"Excel file created: {OUTPUT_EXCEL}")
print(f"CSV file created:   {OUTPUT_CSV}")
print()

print("Number of reflections:", len(df))
print()

print("Reflections per week:")
print(df.groupby("Week")["Word Count"].count())

print()
print("Word counts:")
print(df[["Week", "Word Count"]].to_string(index=False))