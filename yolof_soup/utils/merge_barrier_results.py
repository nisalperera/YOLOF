import os
import json

from glob import glob

merged_lmc_results = {}

for file_path in glob("/home/nisalperera/YOLOF/results/lmc_barriers/*_pairpair_*.json"):
    with open(file_path, "r") as f:
        data = json.load(f)

        base_name = os.path.basename(file_path)
        base_num = base_name[4]
        pair_num = base_name.split("_")[-1].split(".")[0]
        if base_num not in merged_lmc_results:
            merged_lmc_results[base_num] = {}
        merged_lmc_results[base_num][f"pair_{pair_num}"] = data[base_num][f"pair_{pair_num}"]

sorted_merged_lmc_results = dict(sorted(merged_lmc_results.items(), key=lambda x: int(x[0])))
with open("/home/nisalperera/YOLOF/results/phase4_barrier_results.json", "w") as f:
    json.dump(sorted_merged_lmc_results, f, indent=4)