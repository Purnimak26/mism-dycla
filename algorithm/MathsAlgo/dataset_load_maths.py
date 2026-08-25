# ============================================================
# 1. LOAD DATA 
# ============================================================
header = ["src", "dst", "ts"]

main = pd.read_csv("mathsoverflow(main).csv", sep=" ", names=header)
a2q = pd.read_csv("sx-mathoverflow-a2q.csv", sep=" ", names=header)
c2a = pd.read_csv("sx-mathoverflow-c2a.csv", sep=" ", names=header)
c2q = pd.read_csv("sx-mathoverflow-c2q.csv", sep=" ", names=header)

a2q["layer"] = "A2Q"
c2a["layer"] = "C2A"
c2q["layer"] = "C2Q"
