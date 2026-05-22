import streamlit as st
import pandas as pd
import numpy as np

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

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
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

    # Normalize to date (no time)
    df["Open"] = df["Open"].dt.normalize()
    df["Close"] = df["Close"].dt.normalize()

    # Determine date range
    start_date = df["Open"].min()
    end_date = df["Close"].max()

    # Optional: allow user to restrict date range
    st.sidebar.header("Date range")
    user_start = st.sidebar.date_input("Start date", start_date.date())
    user_end = st.sidebar.date_input("End date", end_date.date())

    # Ensure valid order
    if user_start > user_end:
        st.sidebar.error("Start date must be on or before end date.")
        st.stop()

    start_date = pd.to_datetime(user_start)
    end_date = pd.to_datetime(user_end)

    date_index = pd.date_range(start=start_date, end=end_date, freq="D")

    # Unique Note classes
    notes = sorted(df["Note"].dropna().astype(str).unique())

    # Difference-array table: rows = dates, cols = notes
    diff = pd.DataFrame(0, index=date_index, columns=notes, dtype=np.int64)

    # Populate +1 at Open, -1 at Close+1 for each trade
    for _, row in df.iterrows():
        note = str(row["Note"])
        open_d = max(row["Open"], start_date)
        close_d = min(row["Close"], end_date)

        if close_d < start_date or open_d > end_date:
            continue

        # +1 at open
        if open_d in diff.index:
            diff.at[open_d, note] += 1

        # -1 at day after close, if within range
        next_day = close_d + pd.Timedelta(days=1)
        if next_day <= end_date and next_day in diff.index:
            diff.at[next_day, note] -= 1

    # Cumulative sum over dates to get active counts
    daily_counts = diff.cumsum()
    daily_counts.index.name = "date"
    result_df = daily_counts.reset_index()

    st.subheader("Daily allocation counts (table)")
    st.dataframe(result_df, use_container_width=True)

    st.subheader("Daily allocation counts by Note (line chart)")
    # Plot with date as index, one line per Note
    result_plot = result_df.set_index("date")
    st.line_chart(result_plot)

else:
    st.info("Please upload a CSV file to begin.")
