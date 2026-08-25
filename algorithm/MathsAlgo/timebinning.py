# ============================================================
# 2. TIME BINNING 
# ============================================================
def convert_to_days(df):
    df["day"] = (df["ts"] / (24 * 60 * 60)).astype(int)

def convert_to_months(df):
    df["month"] = (df["day"] // 30).astype(int)

for df in [main, a2q, c2a, c2q]:
    convert_to_days(df)
    convert_to_months(df)
