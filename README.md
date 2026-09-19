# diskann-vamana-viz

A small Rust crate for visualizing a sequential debug Vamana-style build on 2D random points.

It produces:
- 8 graph-evolution snapshots
- a 4x2 overview figure
- a final-graph query visualization with visited nodes and discovery tree
- only the **top-1** final neighbor highlighted in green

## Run

```bash
cargo run --release
```

Or with parameters:

```bash
cargo run --release -- \
  --n 200 \
  --max-degree 8 \
  --beam 16 \
  --alpha 1.2 \
  --extra-seeds 2 \
  --seed 7 \
  --out-dir output
```

## Output files

- `frame_01.svg` ... `frame_08.svg`
- `figure1_overview.svg`
- `final_graph_query_trace.svg`

## Notes

This is a sequential visualization crate. It uses the same progressive RobustPrune semantics as `rust-diskann`: alpha relaxation starts at 1.0 inside each prune and the candidate pool is capped at 750. It is not intended to reproduce the production builder's parallel scheduling.
