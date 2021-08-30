res = "16*{"
for y in range(8):
    for x in range(8):
        # print(f"{x=},{y=}")
        res += f"pxl-Test({y+1},{x+1}),"
res = f"{res[:-1]}->L1"
print(res)