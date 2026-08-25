# ============================================================
# 1. LOAD DATA
# ============================================================

header = ["src", "dst", "ts"]

main = pd.read_csv("email-Eu-core-temporal (2).txt", sep=" ", names=header)

dept1 = pd.read_csv("email-Eu-core-temporal-Dept1 (2).txt", sep=" ", names=header)

dept2 = pd.read_csv("email-Eu-core-temporal-Dept2 (1).txt", sep=" ", names=header)

dept3 = pd.read_csv("email-Eu-core-temporal-Dept3 (1).txt", sep=" ", names=header)

dept4 = pd.read_csv("email-Eu-core-temporal-Dept4 (1).txt", sep=" ", names=header)
