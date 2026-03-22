use plotters::prelude::*;
use rand::prelude::*;
use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

const GRAPH_SLACK_FACTOR: f64 = 1.3;

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Debug)]
struct Snapshot {
    step: usize,
    total_steps: usize,
    title: String,
    graph: Vec<Vec<usize>>, // directed graph
}

#[derive(Clone, Debug)]
struct SearchTrace {
    query: Point2,
    visited_order: Vec<usize>,
    visited_set: HashSet<usize>,
    parent: HashMap<usize, usize>, // child -> parent when first discovered
    top1: Option<(usize, f64)>,
}

#[derive(Clone, Debug)]
struct Config {
    n_points: usize,
    max_degree: usize,
    build_beam_width: usize,
    alpha: f64,
    passes: usize,
    extra_seeds: usize,
    seed: u64,
    out_dir: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_points: 200,
            max_degree: 8,
            build_beam_width: 16,
            alpha: 1.2,
            passes: 2,
            extra_seeds: 2,
            seed: 7,
            out_dir: PathBuf::from("output"),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_args();
    fs::create_dir_all(&cfg.out_dir)?;

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let points = generate_points(cfg.n_points, &mut rng);
    let medoid = calculate_medoid(&points);

    println!("Generating Vamana visualization with:");
    println!("  n_points         = {}", cfg.n_points);
    println!("  max_degree       = {}", cfg.max_degree);
    println!("  build_beam_width = {}", cfg.build_beam_width);
    println!("  alpha            = {:.3}", cfg.alpha);
    println!("  passes           = {}", cfg.passes);
    println!("  extra_seeds      = {}", cfg.extra_seeds);
    println!("  seed             = {}", cfg.seed);
    println!("  medoid           = {}", medoid);
    println!("  out_dir          = {}", cfg.out_dir.display());

    let snapshots = build_vamana_debug_snapshots(&points, medoid, &cfg);

    for (idx, snap) in snapshots.iter().enumerate() {
        let path = cfg.out_dir.join(format!("frame_{:02}.svg", idx + 1));
        render_single_snapshot(&path, &points, medoid, snap)?;
    }

    let overview_path = cfg.out_dir.join("figure1_overview.svg");
    render_overview_grid(&overview_path, &points, medoid, &snapshots)?;

    let final_snap = snapshots
        .last()
        .expect("snapshots should not be empty")
        .clone();

    let query = generate_query_point(&mut rng);
    let trace = search_with_trace(
        &points,
        &final_snap.graph,
        query,
        medoid,
        cfg.build_beam_width,
    );

    let query_fig_path = cfg.out_dir.join("final_graph_query_trace.svg");
    render_single_snapshot_with_trace(
        &query_fig_path,
        &points,
        medoid,
        &final_snap,
        &trace,
    )?;

    let summary_path = cfg.out_dir.join("README.txt");
    fs::write(
        &summary_path,
        format!(
            "diskann-vamana-viz\n\
             =================\n\n\
             This folder contains 8 snapshot SVG files, one combined overview,\n\
             and one final-graph query-trace figure.\n\n\
             NOTE\n\
             ----\n\
             This visualization intentionally uses a single-threaded incremental\n\
             build for clarity. It matches the current rust-diskann pruning logic:\n\
             - greedy candidate collection\n\
             - robust alpha-pruning\n\
             - nearest-neighbor backfill\n\
             - reverse insertion with slack-triggered local reprune\n\n\
             Parameters\n\
             ----------\n\
             n_points         = {}\n\
             max_degree       = {}\n\
             build_beam_width = {}\n\
             alpha            = {:.3}\n\
             passes           = {}\n\
             extra_seeds      = {}\n\
             seed             = {}\n\
             medoid           = {}\n\n\
             Files\n\
             -----\n\
             frame_01.svg ... frame_08.svg\n\
             figure1_overview.svg\n\
             final_graph_query_trace.svg\n",
            cfg.n_points,
            cfg.max_degree,
            cfg.build_beam_width,
            cfg.alpha,
            cfg.passes,
            cfg.extra_seeds,
            cfg.seed,
            medoid,
        ),
    )?;

    println!("Done.");
    println!("  Combined figure : {}", overview_path.display());
    println!("  Query overlay   : {}", query_fig_path.display());
    println!("  Per-frame SVGs  : {}/frame_XX.svg", cfg.out_dir.display());

    Ok(())
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--n" => {
                cfg.n_points = args
                    .next()
                    .expect("missing value after --n")
                    .parse()
                    .expect("invalid --n")
            }
            "--max-degree" => {
                cfg.max_degree = args
                    .next()
                    .expect("missing value after --max-degree")
                    .parse()
                    .expect("invalid --max-degree")
            }
            "--beam" => {
                cfg.build_beam_width = args
                    .next()
                    .expect("missing value after --beam")
                    .parse()
                    .expect("invalid --beam")
            }
            "--alpha" => {
                cfg.alpha = args
                    .next()
                    .expect("missing value after --alpha")
                    .parse()
                    .expect("invalid --alpha")
            }
            "--passes" => {
                cfg.passes = args
                    .next()
                    .expect("missing value after --passes")
                    .parse()
                    .expect("invalid --passes")
            }
            "--extra-seeds" => {
                cfg.extra_seeds = args
                    .next()
                    .expect("missing value after --extra-seeds")
                    .parse()
                    .expect("invalid --extra-seeds")
            }
            "--seed" => {
                cfg.seed = args
                    .next()
                    .expect("missing value after --seed")
                    .parse()
                    .expect("invalid --seed")
            }
            "--out-dir" => {
                cfg.out_dir = PathBuf::from(args.next().expect("missing value after --out-dir"))
            }
            "-h" | "--help" => {
                print_help_and_exit();
            }
            other => {
                eprintln!("Unknown argument: {other}");
                print_help_and_exit();
            }
        }
    }

    cfg
}

fn print_help_and_exit() -> ! {
    eprintln!(
        "Usage:\n\
         cargo run --release -- [options]\n\n\
         Options:\n\
           --n <int>              Number of 2D points (default: 200)\n\
           --max-degree <int>     Out-degree cap M (default: 8)\n\
           --beam <int>           Build beam width L (default: 16)\n\
           --alpha <float>        Alpha for pruning (default: 1.2)\n\
           --passes <int>         Number of refinement passes (default: 2)\n\
           --extra-seeds <int>    Extra random greedy-search starts (default: 2)\n\
           --seed <int>           RNG seed (default: 7)\n\
           --out-dir <path>       Output directory (default: output)\n"
    );
    std::process::exit(1)
}

fn generate_points(n: usize, rng: &mut StdRng) -> Vec<Point2> {
    (0..n)
        .map(|_| Point2 {
            x: rng.gen_range(0.0..1.0),
            y: rng.gen_range(0.0..1.0),
        })
        .collect()
}

fn generate_query_point(rng: &mut StdRng) -> Point2 {
    Point2 {
        x: rng.gen_range(0.0..1.0),
        y: rng.gen_range(0.0..1.0),
    }
}

fn l2(a: Point2, b: Point2) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

fn calculate_medoid(points: &[Point2]) -> usize {
    let n = points.len();
    let pivot_count = n.min(8);
    let mut rng = StdRng::seed_from_u64(123456789);
    let pivots: Vec<usize> = (0..pivot_count).map(|_| rng.gen_range(0..n)).collect();

    let mut best_i = 0usize;
    let mut best_score = f64::INFINITY;
    for i in 0..n {
        let score: f64 = pivots.iter().map(|&p| l2(points[i], points[p])).sum();
        if score < best_score {
            best_score = score;
            best_i = i;
        }
    }
    best_i
}

fn bootstrap_random_graph(n: usize, max_degree: usize, rng: &mut StdRng) -> Vec<Vec<usize>> {
    let target = max_degree.min(n.saturating_sub(1));
    let mut graph = vec![Vec::<usize>::new(); n];

    for u in 0..n {
        let mut set = HashSet::with_capacity(target);
        while set.len() < target {
            let v = rng.gen_range(0..n);
            if v != u {
                set.insert(v);
            }
        }
        graph[u] = set.into_iter().collect();
        graph[u].sort_unstable();
    }

    graph
}

fn snapshot_targets(total_steps: usize, count: usize) -> Vec<usize> {
    (1..=count)
        .map(|i| {
            let raw = ((i as f64) * (total_steps as f64) / (count as f64)).ceil() as usize;
            raw.clamp(1, total_steps)
        })
        .collect()
}

/// Single-threaded incremental visualization:
/// - random bootstrap graph
/// - per-node greedy candidate collection
/// - medoid + extra random seeds
/// - robust alpha-pruning with nearest-neighbor backfill
/// - reverse insertion with slack-triggered local reprune
///
/// This is intentionally slower and simpler than the real chunked build,
/// but it now matches the current pruning logic in rust-diskann.
fn build_vamana_debug_snapshots(points: &[Point2], medoid: usize, cfg: &Config) -> Vec<Snapshot> {
    let n = points.len();
    let passes = cfg.passes.max(1);
    let total_refinements = n * passes;
    let target_steps = snapshot_targets(total_refinements, 8);

    let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0xDEADBEEFCAFEBABE);
    let mut graph = bootstrap_random_graph(n, cfg.max_degree, &mut rng);

    let mut snapshots = Vec::<Snapshot>::new();
    let mut refined_count = 0usize;
    let mut next_target_idx = 0usize;

    for pass_idx in 0..passes {
        let pass_alpha = if passes == 1 {
            cfg.alpha
        } else if pass_idx == 0 {
            1.0
        } else {
            cfg.alpha
        };

        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);

        for &u in &order {
            let snapshot = graph.clone();

            let mut candidates = Vec::<(usize, f64)>::new();

            // Start from current adjacency.
            for &nb in &snapshot[u] {
                candidates.push((nb, l2(points[u], points[nb])));
            }

            // Seed list: medoid + distinct random starts.
            let mut seeds = vec![medoid];
            while seeds.len() < 1 + cfg.extra_seeds {
                let s = rng.gen_range(0..n);
                if !seeds.contains(&s) {
                    seeds.push(s);
                }
            }

            for &start in &seeds {
                let visited = greedy_search_visited_collect(
                    points,
                    &snapshot,
                    points[u],
                    start,
                    cfg.build_beam_width,
                );
                candidates.extend(visited);
            }

            dedup_keep_best_by_id_in_place_vis(&mut candidates);

            let pruned = prune_neighbors(u, &candidates, points, cfg.max_degree, pass_alpha);

            // Set u outgoing.
            graph[u] = pruned.clone();

            // Incremental reverse insertion with slack-triggered local reprune.
            inter_insert_with_slack(
                &mut graph,
                u,
                &pruned,
                points,
                cfg.max_degree,
                pass_alpha,
            );

            refined_count += 1;

            while next_target_idx < target_steps.len()
                && refined_count >= target_steps[next_target_idx]
            {
                snapshots.push(Snapshot {
                    step: refined_count,
                    total_steps: total_refinements,
                    title: format!(
                        "{:.1}% refined (pass {}, step {}/{})",
                        100.0 * refined_count as f64 / total_refinements as f64,
                        pass_idx + 1,
                        refined_count,
                        total_refinements
                    ),
                    graph: graph.clone(),
                });
                next_target_idx += 1;
            }
        }
    }

    while snapshots.len() < 8 {
        snapshots.push(Snapshot {
            step: total_refinements,
            total_steps: total_refinements,
            title: format!(
                "100.0% refined (pass {}, step {}/{})",
                passes, total_refinements, total_refinements
            ),
            graph: graph.clone(),
        });
    }

    snapshots.truncate(8);
    snapshots
}

fn greedy_search_visited_collect(
    points: &[Point2],
    graph: &[Vec<usize>],
    query: Point2,
    start_id: usize,
    beam_width: usize,
) -> Vec<(usize, f64)> {
    let start_dist = l2(query, points[start_id]);
    let mut visited_set = HashSet::<usize>::new();
    let mut frontier = vec![(start_id, start_dist)];
    let mut work = vec![(start_id, start_dist)];
    let mut visited = vec![(start_id, start_dist)];
    visited_set.insert(start_id);

    while !frontier.is_empty() {
        frontier.sort_by(|a, b| a.1.total_cmp(&b.1));
        work.sort_by(|a, b| a.1.total_cmp(&b.1));

        let best = frontier[0].1;
        let worst_work = work.last().map(|x| x.1).unwrap_or(f64::INFINITY);
        if work.len() >= beam_width && best >= worst_work {
            break;
        }

        let (cur, _) = frontier.remove(0);
        for &nb in &graph[cur] {
            if visited_set.contains(&nb) {
                continue;
            }
            let d = l2(query, points[nb]);
            visited_set.insert(nb);
            visited.push((nb, d));

            if work.len() < beam_width {
                work.push((nb, d));
                frontier.push((nb, d));
            } else {
                let current_worst = work
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.1.total_cmp(&b.1.1))
                    .map(|(idx, item)| (idx, item.1))
                    .unwrap();
                if d < current_worst.1 {
                    work.remove(current_worst.0);
                    work.push((nb, d));
                    frontier.push((nb, d));
                }
            }
        }
    }

    visited
}

fn search_with_trace(
    points: &[Point2],
    graph: &[Vec<usize>],
    query: Point2,
    medoid: usize,
    beam_width: usize,
) -> SearchTrace {
    let start_dist = l2(query, points[medoid]);

    let mut visited_set = HashSet::<usize>::new();
    let mut visited_order = Vec::<usize>::new();
    let mut parent = HashMap::<usize, usize>::new();

    let mut frontier = vec![(medoid, start_dist)];
    let mut work = vec![(medoid, start_dist)];

    visited_set.insert(medoid);
    visited_order.push(medoid);

    while !frontier.is_empty() {
        frontier.sort_by(|a, b| a.1.total_cmp(&b.1));
        work.sort_by(|a, b| a.1.total_cmp(&b.1));

        let best = frontier[0].1;
        let worst_work = work.last().map(|x| x.1).unwrap_or(f64::INFINITY);

        if work.len() >= beam_width && best >= worst_work {
            break;
        }

        let (cur, _) = frontier.remove(0);

        for &nb in &graph[cur] {
            if visited_set.contains(&nb) {
                continue;
            }

            let d = l2(query, points[nb]);
            visited_set.insert(nb);
            visited_order.push(nb);
            parent.insert(nb, cur);

            if work.len() < beam_width {
                work.push((nb, d));
                frontier.push((nb, d));
            } else {
                let current_worst = work
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.1.total_cmp(&b.1.1))
                    .map(|(idx, item)| (idx, item.1))
                    .unwrap();

                if d < current_worst.1 {
                    work.remove(current_worst.0);
                    work.push((nb, d));
                    frontier.push((nb, d));
                }
            }
        }
    }

    work.sort_by(|a, b| a.1.total_cmp(&b.1));
    let top1 = work.first().copied();

    SearchTrace {
        query,
        visited_order,
        visited_set,
        parent,
        top1,
    }
}

fn dedup_keep_best_by_id_in_place_vis(cands: &mut Vec<(usize, f64)>) {
    if cands.is_empty() {
        return;
    }

    cands.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.total_cmp(&b.1))
    });

    let mut write = 0usize;
    for read in 0..cands.len() {
        if write == 0 || cands[read].0 != cands[write - 1].0 {
            cands[write] = cands[read];
            write += 1;
        }
    }
    cands.truncate(write);
}

/// alpha-pruning with nearest-neighbor backfill.
/// This matches your current rust-diskann pruning logic.
fn prune_neighbors(
    node_id: usize,
    candidates: &[(usize, f64)],
    points: &[Point2],
    max_degree: usize,
    alpha: f64,
) -> Vec<usize> {
    if candidates.is_empty() || max_degree == 0 {
        return Vec::new();
    }

    let mut sorted = candidates.to_vec();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut uniq = Vec::<(usize, f64)>::with_capacity(sorted.len());
    let mut last_id: Option<usize> = None;

    for &(cand_id, cand_dist) in &sorted {
        if cand_id == node_id {
            continue;
        }
        if last_id == Some(cand_id) {
            continue;
        }
        uniq.push((cand_id, cand_dist));
        last_id = Some(cand_id);
    }

    if uniq.is_empty() {
        return Vec::new();
    }

    let mut pruned = Vec::<usize>::with_capacity(max_degree);

    // Phase 1: robust alpha-pruning
    for &(cand_id, cand_dist_to_node) in &uniq {
        let mut occluded = false;

        for &sel_id in &pruned {
            let d_cand_sel = l2(points[cand_id], points[sel_id]);
            if alpha * d_cand_sel <= cand_dist_to_node {
                occluded = true;
                break;
            }
        }

        if !occluded {
            pruned.push(cand_id);
            if pruned.len() >= max_degree {
                return pruned;
            }
        }
    }

    // Phase 2: nearest-neighbor backfill
    if pruned.len() < max_degree {
        for &(cand_id, _) in &uniq {
            if pruned.contains(&cand_id) {
                continue;
            }
            pruned.push(cand_id);
            if pruned.len() >= max_degree {
                break;
            }
        }
    }

    pruned
}

fn inter_insert_with_slack(
    graph: &mut [Vec<usize>],
    src: usize,
    pruned_list: &[usize],
    points: &[Point2],
    max_degree: usize,
    alpha: f64,
) {
    let slack_limit =
        ((GRAPH_SLACK_FACTOR * max_degree as f64).ceil() as usize).max(max_degree);

    for &dst in pruned_list {
        if dst == src {
            continue;
        }

        if graph[dst].contains(&src) {
            continue;
        }

        if graph[dst].len() < slack_limit {
            graph[dst].push(src);
            continue;
        }

        let mut ids = graph[dst].clone();
        ids.push(src);
        ids.sort_unstable();
        ids.dedup();

        let pool: Vec<(usize, f64)> = ids
            .into_iter()
            .filter(|&id| id != dst)
            .map(|id| (id, l2(points[dst], points[id])))
            .collect();

        graph[dst] = prune_neighbors(dst, &pool, points, max_degree, alpha);
    }
}

fn undirected_edges(graph: &[Vec<usize>]) -> Vec<(usize, usize)> {
    let mut set = HashSet::<(usize, usize)>::new();
    for (u, nbrs) in graph.iter().enumerate() {
        for &v in nbrs {
            if u == v {
                continue;
            }
            let e = if u < v { (u, v) } else { (v, u) };
            set.insert(e);
        }
    }
    let mut edges: Vec<_> = set.into_iter().collect();
    edges.sort_unstable();
    edges
}

fn bounds(points: &[Point2], query: Option<Point2>) -> ((f64, f64), (f64, f64)) {
    let (mut xmin, mut xmax) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);

    for p in points {
        xmin = xmin.min(p.x);
        xmax = xmax.max(p.x);
        ymin = ymin.min(p.y);
        ymax = ymax.max(p.y);
    }

    if let Some(q) = query {
        xmin = xmin.min(q.x);
        xmax = xmax.max(q.x);
        ymin = ymin.min(q.y);
        ymax = ymax.max(q.y);
    }

    let xpad = (xmax - xmin).max(1e-6) * 0.05;
    let ypad = (ymax - ymin).max(1e-6) * 0.05;
    ((xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad))
}

fn backtrack_path_to_top1(
    medoid: usize,
    parent: &HashMap<usize, usize>,
    top1: Option<(usize, f64)>,
) -> Vec<usize> {
    let mut path = Vec::<usize>::new();

    if let Some((best_id, _)) = top1 {
        let mut cur = best_id;
        path.push(cur);

        while cur != medoid {
            if let Some(&par) = parent.get(&cur) {
                cur = par;
                path.push(cur);
            } else {
                break;
            }
        }

        path.reverse();
    }

    path
}

fn draw_path_to_top1<DB: DrawingBackend>(
    chart: &mut ChartContext<
        '_,
        DB,
        Cartesian2d<
            plotters::coord::types::RangedCoordf64,
            plotters::coord::types::RangedCoordf64,
        >,
    >,
    points: &[Point2],
    path_to_top1: &[usize],
) -> Result<(), Box<dyn std::error::Error>>
where
    DB::ErrorType: 'static,
{
    if path_to_top1.len() >= 2 {
        for w in path_to_top1.windows(2) {
            let a = points[w[0]];
            let b = points[w[1]];
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(a.x, a.y), (b.x, b.y)],
                ShapeStyle::from(&RGBColor(255, 140, 0)).stroke_width(4),
            )))?;
        }
    }

    chart.draw_series(path_to_top1.iter().map(|&i| {
        let p = points[i];
        Circle::new(
            (p.x, p.y),
            6,
            ShapeStyle::from(&RGBColor(255, 140, 0)).filled(),
        )
    }))?;

    Ok(())
}

fn render_single_snapshot(
    path: &Path,
    points: &[Point2],
    medoid: usize,
    snap: &Snapshot,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(path, (900, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    draw_snapshot_area(&root, points, medoid, snap, None)?;
    root.present()?;
    Ok(())
}

fn render_single_snapshot_with_trace(
    path: &Path,
    points: &[Point2],
    medoid: usize,
    snap: &Snapshot,
    trace: &SearchTrace,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(path, (900, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    draw_snapshot_area(&root, points, medoid, snap, Some(trace))?;
    root.present()?;
    Ok(())
}

fn render_overview_grid(
    path: &Path,
    points: &[Point2],
    medoid: usize,
    snapshots: &[Snapshot],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(path, (3600, 1800)).into_drawing_area();
    root.fill(&WHITE)?;

    let panels = root.split_evenly((2, 4));
    for (panel, snap) in panels.iter().zip(snapshots.iter()) {
        draw_snapshot_area(panel, points, medoid, snap, None)?;
    }

    root.present()?;
    Ok(())
}

fn draw_snapshot_area<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    points: &[Point2],
    medoid: usize,
    snap: &Snapshot,
    trace: Option<&SearchTrace>,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB::ErrorType: 'static,
{
    let ((xmin, xmax), (ymin, ymax)) = bounds(points, trace.map(|t| t.query));
    let edges = undirected_edges(&snap.graph);
    let edge_count = edges.len();

    let caption = match trace {
        Some(t) => format!(
            "{} | undirected edges = {} | visited = {} | top1 = {}",
            snap.title,
            edge_count,
            t.visited_set.len(),
            usize::from(t.top1.is_some())
        ),
        None => format!(
            "{} | undirected edges = {} | step {}/{}",
            snap.title, edge_count, snap.step, snap.total_steps
        ),
    };

    let mut chart = ChartBuilder::on(area)
        .margin(20)
        .caption(caption, ("Helvetica", 20).into_font())
        .x_label_area_size(0)
        .y_label_area_size(0)
        .build_cartesian_2d(xmin..xmax, ymin..ymax)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_labels(0)
        .y_labels(0)
        .axis_style(BLACK)
        .light_line_style(WHITE.mix(0.0))
        .bold_line_style(WHITE.mix(0.0))
        .draw()?;

    chart.draw_series(edges.iter().map(|&(u, v)| {
        let p = points[u];
        let q = points[v];
        PathElement::new(vec![(p.x, p.y), (q.x, q.y)], BLACK.mix(0.36))
    }))?;

    chart.draw_series(points.iter().enumerate().map(|(i, p)| {
        let radius = if i == medoid { 7 } else { 5 };
        let style = if i == medoid {
            ShapeStyle::from(&RED).filled()
        } else {
            ShapeStyle::from(&BLUE).filled()
        };
        Circle::new((p.x, p.y), radius, style)
    }))?;

    if let Some(trace) = trace {
        let path_to_top1 = backtrack_path_to_top1(medoid, &trace.parent, trace.top1);

        draw_path_to_top1(&mut chart, points, &path_to_top1)?;

        if let Some((best_id, _)) = trace.top1 {
            let bp = points[best_id];
            chart.draw_series(std::iter::once(Circle::new(
                (bp.x, bp.y),
                8,
                ShapeStyle::from(&GREEN).filled(),
            )))?;

            chart.draw_series(std::iter::once(PathElement::new(
                vec![(trace.query.x, trace.query.y), (bp.x, bp.y)],
                ShapeStyle::from(&MAGENTA.mix(0.55)).stroke_width(2),
            )))?;
        }

        let mp = points[medoid];
        chart.draw_series(std::iter::once(Circle::new(
            (mp.x, mp.y),
            7,
            ShapeStyle::from(&RED).filled(),
        )))?;

        chart.draw_series(std::iter::once(Cross::new(
            (trace.query.x, trace.query.y),
            10,
            ShapeStyle::from(&MAGENTA).stroke_width(3),
        )))?;
    }

    chart.plotting_area().draw(&Rectangle::new(
        [(xmin, ymin), (xmax, ymax)],
        ShapeStyle::from(&BLACK).stroke_width(1),
    ))?;

    Ok(())
}