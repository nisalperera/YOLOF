"""
RQ1 / H1 – Wilcoxon signed-rank test + Cohen's d.
Compares head-specific vs global soup mAP across 3 soup types.
Run AFTER Phase 3.

Run: python -m experiments.rq1_final_test
"""

from __future__ import annotations

import json
import logging

from yolof_soup.config.experiment_config import RESULTS_DIR
from yolof_soup.utils import wilcoxon_one_tailed, cohens_d

logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("RQ1 / H1: Statistical Test")
    logger.info("=" * 60)

    with open(f"{RESULTS_DIR}/phase3_soup_results.json") as f:
        p3 = json.load(f)

    head_maps = [
        p3["uniform_head"]["AP"],
        p3["greedy_head"]["AP"],
        p3["learned_head"]["AP"],
    ]
    global_maps = [
        p3["uniform_global"]["AP"],
        p3.get("greedy_global",  {}).get("AP", p3["uniform_global"]["AP"]),
        p3.get("learned_global", {}).get("AP", p3["uniform_global"]["AP"]),
    ]
    baseline = p3["baseline"]["AP"]

    logger.info("Head-specific mAP : %s", head_maps)
    logger.info("Global mAP        : %s", global_maps)
    logger.info("Baseline mAP      : %.4f", baseline)

    wsr = wilcoxon_one_tailed(head_maps, global_maps)
    logger.info("\n--- Wilcoxon Signed-Rank (H1) ---")
    logger.info(json.dumps(wsr, indent=2))
    logger.info("H1 supported: %s", wsr["h1_supported"])

    best_head = max(head_maps)
    d         = cohens_d([best_head], [baseline])
    logger.info("\n--- Cohen's d (best head soup vs. baseline) ---")
    logger.info("d = %.4f", d)

    results = dict(head_maps=head_maps, global_maps=global_maps,
                   baseline=baseline, wilcoxon=wsr, cohens_d=d)
    out = f"{RESULTS_DIR}/rq1_test_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("RQ1 results saved → %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()