with open("thrml/block_sampling.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "interaction_slices[i, :m] = inds" in line:
        new_lines.append("                        interaction_slices[i, :m] = inds\n")
    elif "active[i, :m] = 1" in line:
        new_lines.append("                        active[i, :m] = True\n")
    else:
        new_lines.append(line)

with open("thrml/block_sampling.py", "w") as f:
    f.writelines(new_lines)
