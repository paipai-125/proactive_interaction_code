# probe_kvcache

This directory is independent from `probe/`.

`build_probe_dataset_kvcache.py` is the stats-stage sliding visual-prefix
KV-cache experiment. It reuses Response-G1 scene graphs/retrieval and MMDuet2
message labels. It is a bounded-KV approximation after frame eviction, so first
compare it with the no-KV builder on a fixed small subset.
