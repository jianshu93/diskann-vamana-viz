diskann-vamana-viz
=================

This folder contains 8 snapshot SVG files, one combined overview,
and one final-graph query-trace figure.

NOTE
----
This visualization intentionally uses a single-threaded incremental
build for clarity. It matches the current rust-diskann pruning logic:
- greedy candidate collection
- robust alpha-pruning
- nearest-neighbor backfill
- reverse insertion with slack-triggered local reprune

Parameters
----------
n_points         = 200
max_degree       = 8
build_beam_width = 16
alpha            = 1.200
passes           = 2
extra_seeds      = 2
seed             = 117
medoid           = 6

Files
-----
frame_01.svg ... frame_08.svg
figure1_overview.svg
final_graph_query_trace.svg
