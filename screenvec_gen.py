res = ""
# for y in range(28):
#     for x in range(28):
#         # print(f"{x=},{y=}")
#         res += f"pxl-Test({y+1},{x+1}),"
# res = f"{res[:-1]}->L1"
res += "{"
for y in range(28//2):
    for x in range(28):
        # print(f"{x=},{y=}")
        res += f"pxl-Test({y+1},{x+1}),"
res = f"{res[:-1]}->L1\n"
res += "Archive L1\n"
res += "{"
for y in range(28//2, 28):
    for x in range(28):
        # print(f"{x=},{y=}")
        res += f"pxl-Test({y+1},{x+1}),"
res = f"{res[:-1]}->L2\n"
# res += "Archive L2\n"
print(res)