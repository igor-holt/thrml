with open("thrml/block_sampling.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if "                    active = np.zeros((n_nodes, n_interactions), dtype=bool)" in line:
        new_lines.append(line)
        new_lines.append("                    for i, inds in enumerate(interact_inds):\n")
        new_lines.append("                        if not inds:\n")
        new_lines.append("                            continue\n")
        new_lines.append("                        m = len(inds)\n")
        new_lines.append("                        interaction_slices[i, :m] = inds\n")
        new_lines.append("                        active[i, :m] = 1\n")
        new_lines.append("                        \n")
        new_lines.append("                        for k, tail_block in enumerate(interaction_group.tail_nodes):\n")
        new_lines.append("                            global_slices[k][i, :m] = [\n")
        new_lines.append("                                gibbs_spec.node_global_location_map[tail_block.nodes[ind]][1]\n")
        new_lines.append("                                for ind in inds\n")
        new_lines.append("                            ]\n")
        skip = True
    elif skip and "interaction_slices = jnp.array(interaction_slices)" in line:
        skip = False
        new_lines.append("\n                    interaction_slices = jnp.array(interaction_slices)\n")
    elif not skip:
        new_lines.append(line)

with open("thrml/block_sampling.py", "w") as f:
    f.writelines(new_lines)
