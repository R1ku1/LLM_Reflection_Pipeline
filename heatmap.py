import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv('heatmap_data.csv', index_col=0)

score_map = {
    'Present': 2,
    'Weak': 1,
    'Missing': 0,
    'No submission': -1
}

color_map = {
    2: '#16A34A',
    1: '#F59E0B',
    0: '#DC2626',
    -1: '#E5E7EB'
}

numeric = df.apply(
    lambda col: col.map(lambda x: score_map.get(x, -1))
)


# ============================================================
# SORT BY TOTAL WEEKS ABSENT
# ============================================================

# Treat both "Missing" and "No submission" as absent.
absent_count = ((numeric == 0) | (numeric == -1)).sum(axis=1)

# Secondary sorting:
#   1. Most absent weeks first
#   2. Then most weak weeks
#   3. Then most present weeks
weak_count = (numeric == 1).sum(axis=1)
present_count = (numeric == 2).sum(axis=1)

sort_key = pd.DataFrame({
    'absent': absent_count,
    'weak': weak_count,
    'present': present_count
})

numeric_sorted = numeric.loc[
    sort_key.sort_values(
        by=['absent', 'weak', 'present'],
        ascending=[False, False, True]
    ).index
]


n_students = len(numeric_sorted)
n_weeks = len(numeric_sorted.columns)


# ============================================================
# FIGURE
# ============================================================

# Wider and considerably shorter than the original 9 x 14.
fig, ax = plt.subplots(figsize=(14, 6))

fig.patch.set_facecolor('white')
ax.set_facecolor('white')


# ============================================================
# DRAW HEATMAP
# ============================================================

for row_idx in range(n_students):
    for col_idx in range(n_weeks):

        val = numeric_sorted.iloc[row_idx, col_idx]

        ax.add_patch(
            plt.Rectangle(
                (col_idx, row_idx),
                1,
                1,
                facecolor=color_map[val],
                linewidth=0
            )
        )


# ============================================================
# AXES
# ============================================================

ax.set_xlim(0, n_weeks)
ax.set_ylim(0, n_students)

# Put week labels in the centre of each cell.
ax.set_xticks(
    [i + 0.5 for i in range(n_weeks)]
)

ax.set_xticklabels(
    [f'Week {i + 1}' for i in range(n_weeks)],
    fontsize=12,
    fontweight='bold'
)

# No individual student names/IDs.
# This keeps the figure compact even with many students.
ax.set_yticks([])





# ============================================================
# WEEK DIVIDERS
# ============================================================

for x in range(1, n_weeks):
    ax.axvline(
        x,
        color='white',
        linewidth=1.2
    )


# ============================================================
# GROUP COUNTS
# ============================================================

# Consistently absent:
# absent in every week
always_absent = int(
    (absent_count == n_weeks).sum()
)

# Consistently present:
# present in at least 6 weeks
always_present = int(
    (present_count >= 6).sum()
)


# ============================================================
# RIGHT-SIDE GROUP ANNOTATIONS
# ============================================================

bracket_x = n_weeks + 0.12

# ---- Consistently absent ----
if always_absent > 0:

    ax.annotate(
        '',
        xy=(bracket_x, 0),
        xytext=(bracket_x, always_absent),
        arrowprops=dict(
            arrowstyle='-',
            color='#DC2626',
            lw=2
        )
    )

    ax.text(
        bracket_x + 0.12,
        always_absent / 2,
        f'Consistently\nabsent\n(n={always_absent})',
        va='center',
        fontsize=10,
        color='#DC2626',
        fontweight='bold'
    )


# ---- Consistently present ----
if always_present > 0:

    ax.annotate(
        '',
        xy=(bracket_x, n_students - always_present),
        xytext=(bracket_x, n_students),
        arrowprops=dict(
            arrowstyle='-',
            color='#16A34A',
            lw=2
        )
    )

    ax.text(
        bracket_x + 0.12,
        n_students - always_present / 2,
        f'Consistently\npresent\n(n={always_present})',
        va='center',
        fontsize=10,
        color='#16A34A',
        fontweight='bold'
    )


# ============================================================
# LEGEND
# ============================================================

legend = [
    mpatches.Patch(
        color='#16A34A',
        label='Present (2+ phrases)'
    ),
    mpatches.Patch(
        color='#F59E0B',
        label='Partially present (1 phrase)'
    ),
    mpatches.Patch(
        color='#DC2626',
        label='Absent (0 phrases)'
    ),
    mpatches.Patch(
        color='#E5E7EB',
        label='No submission'
    ),
]

ax.legend(
    handles=legend,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    ncol=1,
    fontsize=10,
    frameon=False
)


# ============================================================
# CLEAN UP SPINES
# ============================================================

for spine in ax.spines.values():
    spine.set_visible(False)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout()

plt.savefig(
    'figure6_heatmap_sorted.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()