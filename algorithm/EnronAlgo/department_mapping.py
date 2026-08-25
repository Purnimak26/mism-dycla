dept1["dept"] = "Dept1"
dept2["dept"] = "Dept2"
dept3["dept"] = "Dept3"
dept4["dept"] = "Dept4"


# ============================================================
# 2. DEPARTMENT MAPPING
# ============================================================

DepartmentMapping = defaultdict(set)


def dept_map(df):
    dept = df["dept"].iloc[0]

    for u, v in zip(df["src"], df["dst"]):
        DepartmentMapping[u].add(dept)
        DepartmentMapping[v].add(dept)


dept_map(dept1)
dept_map(dept2)
dept_map(dept3)
dept_map(dept4)


print("Department Mapping")

for node, dept in list(DepartmentMapping.items())[:20]:
    print("Node", node, "->", dept)

all_nodes = len(DepartmentMapping)

multi_dept_nodes = sum(
    1 for d in DepartmentMapping.values()
    if len(d) > 1
)

print("Total nodes:", all_nodes)
print("Nodes in multiple departments:", multi_dept_nodes)


# ============================================================
# 3. FIND MISSING NODES
# ============================================================

main_nodes = set(main["src"]).union(set(main["dst"]))
dept_nodes = set(DepartmentMapping.keys())

missing_nodes = main_nodes - dept_nodes

print("Missing nodes:", len(missing_nodes))

for node in missing_nodes:
    DepartmentMapping[node] = {"Unknown"}
