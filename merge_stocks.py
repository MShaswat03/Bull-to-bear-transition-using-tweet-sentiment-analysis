import pandas as pd


def standardize_columns(df):
    """Standardize column names to lowercase, remove spaces, and slashes."""
    df.columns = (
        df.columns.str.strip().str.replace(" ", "").str.replace("/", "").str.lower()
    )
    return df


def clean_numeric_columns(df, columns):
    """Convert columns with strings (like $ or ,) to numeric values."""
    for col in columns:
        df[col] = (
            df[col]
            .replace("[\$,]", "", regex=True)  # ✅ Remove $ and ,
            .astype(float)  # ✅ Convert to numeric
        )
    return df


def add_company_column(df, company_name):
    """Add the company column to identify the data source."""
    df.loc[:, "company"] = company_name  # ✅ Avoids SettingWithCopyWarning
    return df


def merge_stock_data(files, yahoo_file):
    """Merge multiple stock data files into one and average duplicate dates."""
    dfs = []

    # --- 1) Load Yahoo Finance Data ---
    yahoo_df = pd.read_csv(yahoo_file)

    # ✅ Check for known column names and rename if needed
    yahoo_df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "Close": "close",
            "Close/Last": "close",  # Yahoo sometimes uses this
        },
        inplace=True,
    )

    # ✅ Standardize column names
    yahoo_df = standardize_columns(yahoo_df)

    # ✅ Print column names to debug
    print("Yahoo columns:", yahoo_df.columns.tolist())

    # ✅ Clean and convert numeric columns
    yahoo_df = clean_numeric_columns(yahoo_df, ["open", "close"])

    # ✅ Check if 'close' column is present
    if "close" not in yahoo_df.columns:
        raise KeyError(
            f"Column 'close' not found in {yahoo_file}. Available columns: {yahoo_df.columns.tolist()}"
        )

    # ✅ Select necessary columns and add 'company'
    yahoo_df = add_company_column(yahoo_df[["date", "open", "close"]], "Tesla")
    dfs.append(yahoo_df)

    # --- 2) Load all stock files and add 'company' ---
    company_mapping = {
        "data/tesla_stocks.csv": "Tesla",
        "data/AAPL.csv": "Apple",
        "data/GOOGL.csv": "Google",
        "data/META.csv": "Meta",
    }

    for file, company_name in company_mapping.items():
        df = pd.read_csv(file)

        # ✅ Standardize column names
        df = standardize_columns(df)

        # ✅ Rename columns if needed
        if "date" not in df.columns and "Date" in df.columns:
            df.rename(columns={"Date": "date"}, inplace=True)
        if "open" not in df.columns and "Open" in df.columns:
            df.rename(columns={"Open": "open"}, inplace=True)
        if "close" not in df.columns and "Close" in df.columns:
            df.rename(columns={"Close": "close"}, inplace=True)

        # ✅ Check if required columns are present
        if not {"date", "open", "close"}.issubset(df.columns):
            raise KeyError(
                f"Missing required columns in {file}. Columns found: {df.columns.tolist()}"
            )

        # ✅ Clean and convert numeric columns
        df = clean_numeric_columns(df, ["open", "close"])

        # ✅ Add company name and select required columns
        df = add_company_column(df[["date", "open", "close"]], company_name)
        dfs.append(df)

    # --- 3) Merge all data and average duplicate dates ---
    merged = (
        pd.concat(dfs)
        .groupby(["date", "company"])
        .agg({"open": "mean", "close": "mean"})
        .reset_index()
    )
    merged = merged.sort_values(["date", "company"])

    return merged


if __name__ == "__main__":
    merged = merge_stock_data(
        [
            "data/tesla_stocks.csv",
            "data/AAPL.csv",
            "data/GOOGL.csv",
            "data/META.csv",
        ],
        "data/Yahoo_Tesla.csv",
    )
    merged.to_csv("results/merged_stocks.csv", index=False)
    print("✅ Merged stock data saved to results/merged_stocks.csv")
