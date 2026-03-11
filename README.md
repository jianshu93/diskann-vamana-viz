# diskann-vamana-viz-top1

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
  --passes 2 \
  --extra-seeds 2 \
  --seed 7 \
  --out-dir output
```

## Output files

- `frame_01.svg` ... `frame_08.svg`
- `figure1_overview.svg`
- `final_graph_query_trace_top1.svg`

## Notes

This is a sequential visualization crate. It is Vamana-style, but it is not intended to be a byte-for-byte reproduction of the production `rust-diskann` builder, which is fully parallelized and optimized.
