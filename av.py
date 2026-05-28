import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(page_title="Daily Note Allocation", layout="wide")
st.title("Daily Note Allocation Count")

st.markdown(
    """
Upload a CSV with at least these columns:

- **Open**: trade open date
- **Close**: trade close date
- **Note**: classification label
"""
)


@st.cache_data(show_spinner=False)
def compute_daily_counts(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a cleaned DataFrame with columns:
      - Open (datetime, normalized to date)
      - Close (datetime, normalized to date)
      - Note (string)

    Compute daily active counts per Note using a difference-array approach.
    """
    # Global date range over all trades
    start_date = clean_df["Open"].min()
    end_date = clean_df["Close"].max()

    date_index = pd.date_range(start=start_date, end=end_date, freq="D")

    notes = sorted(clean_df["Note"].astype(str).unique())

    # Difference-array table: rows = dates, cols = notes
    diff = pd.DataFrame(0, index=date_index, columns=notes, dtype=np.int64)

    # Faster than iterrows
    for row in clean_df.itertuples(index=False):
        note = str(row.Note)
        open_d = row.Open
        close_d = row.Close

        # Clip to global range (mostly redundant, but safe)
        if close_d < start_date or open_d > end_date:
            continue

        if open_d in diff.index:
            diff.at[open_d, note] += 1

        if close_d in diff.index:
        diff.at[close_d, note] -= 1

    # Cumulative sum over dates to get active counts
    daily_counts = diff.cumsum()
    daily_counts.index.name = "date"
    return daily_counts


uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

# Read CSV
df = pd.read_csv(uploaded_file)

# Basic checks
required_cols = {"Open", "Close", "Note"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing required columns: {', '.join(sorted(missing))}")
    st.stop()

# Parse dates
df["Open"] = pd.to_datetime(df["Open"], errors="coerce")
df["Close"] = pd.to_datetime(df["Close"], errors="coerce")

# Drop rows with invalid dates or missing Note
df = df.dropna(subset=["Open", "Close", "Note"])

if df.empty:
    st.warning("No valid rows after cleaning (check Open/Close/Note columns).")
    st.stop()

# Normalize to date (no time); ensure Note is string
df["Open"] = df["Open"].dt.normalize()
df["Close"] = df["Close"].dt.normalize()
df["Note"] = df["Note"].astype(str)

# Compute full-range daily counts once (cached)
daily_counts_full = compute_daily_counts(df)

if daily_counts_full.empty:
    st.warning("No daily counts could be computed from the data.")
    st.stop()

full_start = daily_counts_full.index.min()
full_end = daily_counts_full.index.max()

# Sidebar controls
st.sidebar.header("Controls")

# Date range limit within full range
user_start = st.sidebar.date_input("Start date", full_start.date(), min_value=full_start.date(), max_value=full_end.date())
user_end = st.sidebar.date_input("End date", full_end.date(), min_value=full_start.date(), max_value=full_end.date())

if user_start > user_end:
    st.sidebar.error("Start date must be on or before end date.")
    st.stop()

start_date = pd.to_datetime(user_start)
end_date = pd.to_datetime(user_end)

# Slice the precomputed daily counts
daily_counts = daily_counts_full.loc[start_date:end_date]

if daily_counts.empty:
    st.warning("No data in the selected date range.")
    st.stop()

# Granularity: daily vs monthly snapshots
granularity = st.sidebar.selectbox("Time granularity", ["Daily", "Monthly snapshots"])

if granularity == "Daily":
    plot_data = daily_counts
else:
    # Monthly snapshot: last available day in each month
    plot_data = daily_counts.resample("M").last()
    plot_data.index.name = "date"

if plot_data.empty:
    st.warning("No data to plot after applying granularity.")
    st.stop()

# Limit number of categories/Notes plotted
num_notes = plot_data.shape[1]
max_default = min(10, num_notes)
max_upper = min(50, num_notes)  # hard cap to keep things manageable

max_categories = st.sidebar.slider(
    "Max categories (Notes) to plot",
    min_value=1,
    max_value=max_upper,
    value=max_default,
)

# Rank notes by overall activity (sum over time) and select top-N
note_totals = plot_data.sum(axis=0)
top_notes = note_totals.sort_values(ascending=False).head(max_categories).index.tolist()

plot_subset = plot_data[top_notes]

# Optional: show data table (can be heavy for large ranges)
show_table = st.checkbox("Show data table for plotted series (can be slow for large ranges)", value=False)

if show_table:
    st.subheader("Allocation counts (table) for plotted Notes")
    st.dataframe(plot_subset.reset_index(), use_container_width=True)

# Static matplotlib plot
st.subheader(f"{granularity} allocation counts by Note (Top {len(top_notes)} categories)")

fig, ax = plt.subplots(figsize=(12, 6))

for note in top_notes:
    ax.plot(plot_subset.index, plot_subset[note], label=note, linewidth=1.3)

ax.set_xlabel("Date")
ax.set_ylabel("Active count")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# Legend on the right, vertically stacked
ax.legend(
    title="Note",
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
    borderaxespad=0.0,
    fontsize="small",
)

fig.tight_layout()
st.pyplot(fig)
