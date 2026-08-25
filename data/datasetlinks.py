from pathlib import Path

content = """References

[1] J. Leskovec and A. Krevl, “SNAP Datasets: Stanford Network Analysis Project — Email-Eu-core temporal network,” Stanford University. [Online]. Available: https://snap.stanford.edu/data/email-Eu-core-temporal.html

[2] J. Leskovec and A. Krevl, “SNAP Datasets: Stanford Network Analysis Project — Math Overflow temporal network,” Stanford University. [Online]. Available: https://snap.stanford.edu/data/sx-mathoverflow.html
"""

path = Path("/mnt/data/references.txt")
path.write_text(content, encoding="utf-8")
print(path)
