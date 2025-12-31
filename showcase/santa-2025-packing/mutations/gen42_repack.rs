//! Evolved Packing Algorithm - Generation 42 REPACK
//!
//! MUTATION STRATEGY: Periodically repack subsets of trees.
//!
//! Changes from Gen28:
//! - After SA, try removing and re-adding boundary trees
//! - This can find better arrangements that SA alone can't discover
//! - Keeps configuration if repack improves score

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub enum PlacementStrategy { ClockwiseSpiral, CounterclockwiseSpiral, Grid, Random, BoundaryFirst }

pub struct EvolvedConfig {
    pub search_attempts: usize,
    pub direction_samples: usize,
    pub sa_iterations: usize,
    pub sa_initial_temp: f64,
    pub sa_cooling_rate: f64,
    pub sa_min_temp: f64,
    pub translation_scale: f64,
    pub rotation_granularity: f64,
    pub center_pull_strength: f64,
    pub sa_passes: usize,
    pub early_exit_threshold: usize,
    pub boundary_focus_prob: f64,
    pub num_strategies: usize,
    pub density_grid_resolution: usize,
    pub gap_penalty_weight: f64,
    pub local_density_radius: f64,
    pub fill_move_prob: f64,
    pub hot_restart_interval: usize,
    pub hot_restart_temp: f64,
    pub elite_pool_size: usize,
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        Self {
            search_attempts: 200,
            direction_samples: 64,
            sa_iterations: 28000,
            sa_initial_temp: 0.45,
            sa_cooling_rate: 0.99993,
            sa_min_temp: 0.00001,
            translation_scale: 0.055,
            rotation_granularity: 45.0,
            center_pull_strength: 0.07,
            sa_passes: 2,
            early_exit_threshold: 2500,
            boundary_focus_prob: 0.85,
            num_strategies: 5,
            density_grid_resolution: 20,
            gap_penalty_weight: 0.15,
            local_density_radius: 0.5,
            fill_move_prob: 0.15,
            hot_restart_interval: 800,
            hot_restart_temp: 0.35,
            elite_pool_size: 3,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BoundaryEdge { Left, Right, Top, Bottom, Corner, None }

pub struct EvolvedPacker { pub config: EvolvedConfig }

impl Default for EvolvedPacker {
    fn default() -> Self { Self { config: EvolvedConfig::default() } }
}

impl EvolvedPacker {
    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        let mut rng = rand::thread_rng();
        let mut packings: Vec<Packing> = Vec::with_capacity(max_n);

        let strategies = [
            PlacementStrategy::ClockwiseSpiral,
            PlacementStrategy::CounterclockwiseSpiral,
            PlacementStrategy::Grid,
            PlacementStrategy::Random,
            PlacementStrategy::BoundaryFirst,
        ];

        let mut strategy_trees: Vec<Vec<PlacedTree>> = vec![Vec::new(); strategies.len()];

        for n in 1..=max_n {
            let mut best_trees: Option<Vec<PlacedTree>> = None;
            let mut best_side = f64::INFINITY;

            for (s_idx, &strategy) in strategies.iter().enumerate() {
                let mut trees = strategy_trees[s_idx].clone();
                let new_tree = self.find_placement_with_strategy(&trees, n, max_n, strategy, &mut rng);
                trees.push(new_tree);

                for pass in 0..self.config.sa_passes {
                    self.local_search(&mut trees, n, pass, strategy, &mut rng);
                }

                // REPACK: Try to improve by removing and re-adding boundary trees
                if n >= 10 && n % 5 == 0 {
                    self.try_repack(&mut trees, &mut rng);
                }

                let side = compute_side_length(&trees);
                strategy_trees[s_idx] = trees.clone();

                if side < best_side {
                    best_side = side;
                    best_trees = Some(trees);
                }
            }

            let best = best_trees.unwrap();
            let mut packing = Packing::new();
            for t in &best { packing.trees.push(t.clone()); }
            packings.push(packing);

            for strat_trees in strategy_trees.iter_mut() {
                if compute_side_length(strat_trees) > best_side * 1.02 {
                    *strat_trees = best.clone();
                }
            }
        }
        packings
    }

    /// Try to improve by removing boundary trees and re-adding them
    fn try_repack(&self, trees: &mut Vec<PlacedTree>, rng: &mut impl Rng) {
        if trees.len() < 5 { return; }

        let current_side = compute_side_length(trees);
        let boundary_info = self.find_boundary_trees_with_edges(trees);

        // Only repack corner trees (most impact on bounding box)
        let corner_indices: Vec<usize> = boundary_info.iter()
            .filter(|(_, edge)| matches!(edge, BoundaryEdge::Corner))
            .map(|(idx, _)| *idx)
            .collect();

        if corner_indices.is_empty() { return; }

        // Try removing up to 3 corner trees and re-adding them
        let num_to_repack = corner_indices.len().min(3);
        let mut indices_to_repack: Vec<usize> = corner_indices.iter()
            .take(num_to_repack)
            .copied()
            .collect();

        // Save original trees
        let original_trees = trees.clone();

        // Sort indices in descending order for safe removal
        indices_to_repack.sort_by(|a, b| b.cmp(a));

        // Remove the trees
        let mut removed_trees: Vec<PlacedTree> = Vec::new();
        for &idx in &indices_to_repack {
            removed_trees.push(trees.remove(idx));
        }

        // Try to re-add them in different positions
        for removed in removed_trees {
            let new_tree = self.find_best_repack_position(trees, removed.angle_deg, rng);
            trees.push(new_tree);
        }

        // Keep new arrangement only if it improves
        let new_side = compute_side_length(trees);
        if new_side >= current_side {
            *trees = original_trees;
        }
    }

    /// Find best position for a repacked tree
    fn find_best_repack_position(&self, existing: &[PlacedTree], preferred_angle: f64, rng: &mut impl Rng) -> PlacedTree {
        let mut best_tree = PlacedTree::new(0.0, 0.0, preferred_angle);
        let mut best_score = f64::INFINITY;

        let angles = [preferred_angle, 0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0];
        let (min_x, min_y, max_x, max_y) = compute_bounds(existing);

        // Try many directions
        for attempt in 0..100 {
            let dir = (attempt as f64 / 100.0) * 2.0 * PI + rng.gen_range(-0.1..0.1);
            let (vx, vy) = (dir.cos(), dir.sin());

            for &tree_angle in &angles {
                let (mut low, mut high) = (0.0, 12.0);
                while high - low > 0.001 {
                    let mid = (low + high) / 2.0;
                    if is_valid(&PlacedTree::new(mid * vx, mid * vy, tree_angle), existing) { high = mid; } else { low = mid; }
                }

                let candidate = PlacedTree::new(high * vx, high * vy, tree_angle);
                if is_valid(&candidate, existing) {
                    let score = self.repack_score(&candidate, existing, min_x, min_y, max_x, max_y);
                    if score < best_score { best_score = score; best_tree = candidate; }
                }
            }
        }
        best_tree
    }

    /// Scoring for repacking - prioritize not extending bounds
    fn repack_score(&self, tree: &PlacedTree, existing: &[PlacedTree], old_min_x: f64, old_min_y: f64, old_max_x: f64, old_max_y: f64) -> f64 {
        let (tree_min_x, tree_min_y, tree_max_x, tree_max_y) = tree.bounds();
        let (mut pack_min_x, mut pack_min_y, mut pack_max_x, mut pack_max_y) = (tree_min_x, tree_min_y, tree_max_x, tree_max_y);
        for t in existing {
            let (bx1, by1, bx2, by2) = t.bounds();
            pack_min_x = pack_min_x.min(bx1); pack_min_y = pack_min_y.min(by1);
            pack_max_x = pack_max_x.max(bx2); pack_max_y = pack_max_y.max(by2);
        }
        let side = (pack_max_x - pack_min_x).max(pack_max_y - pack_min_y);

        // Heavy penalty for extending beyond original bounds
        let extension = (pack_max_x - old_max_x).max(0.0) + (old_min_x - pack_min_x).max(0.0)
                      + (pack_max_y - old_max_y).max(0.0) + (old_min_y - pack_min_y).max(0.0);

        side + extension * 0.5
    }

    fn find_placement_with_strategy(&self, existing: &[PlacedTree], n: usize, _max_n: usize, strategy: PlacementStrategy, rng: &mut impl Rng) -> PlacedTree {
        if existing.is_empty() {
            let initial_angle = match strategy {
                PlacementStrategy::ClockwiseSpiral => 0.0,
                PlacementStrategy::CounterclockwiseSpiral => 90.0,
                PlacementStrategy::Grid => 45.0,
                PlacementStrategy::Random => rng.gen_range(0..8) as f64 * 45.0,
                PlacementStrategy::BoundaryFirst => 180.0,
            };
            return PlacedTree::new(0.0, 0.0, initial_angle);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 90.0);
        let mut best_score = f64::INFINITY;

        let angles = self.select_angles_for_strategy(n, strategy);
        let (min_x, min_y, max_x, max_y) = compute_bounds(existing);
        let current_width = max_x - min_x;
        let current_height = max_y - min_y;
        let gaps = self.find_gaps(existing, min_x, min_y, max_x, max_y);

        for attempt in 0..self.config.search_attempts {
            let dir = if !gaps.is_empty() && attempt % 5 == 0 {
                let gap = &gaps[attempt % gaps.len()];
                ((gap.1 + gap.3) / 2.0).atan2((gap.0 + gap.2) / 2.0)
            } else {
                self.select_direction_for_strategy(n, current_width, current_height, strategy, attempt, rng)
            };

            let (vx, vy) = (dir.cos(), dir.sin());

            for &tree_angle in &angles {
                let (mut low, mut high) = (0.0, 12.0);
                while high - low > 0.001 {
                    let mid = (low + high) / 2.0;
                    if is_valid(&PlacedTree::new(mid * vx, mid * vy, tree_angle), existing) { high = mid; } else { low = mid; }
                }

                let candidate = PlacedTree::new(high * vx, high * vy, tree_angle);
                if is_valid(&candidate, existing) {
                    let score = self.placement_score(&candidate, existing, n);
                    if score < best_score { best_score = score; best_tree = candidate; }
                }
            }
        }
        best_tree
    }

    fn select_angles_for_strategy(&self, n: usize, strategy: PlacementStrategy) -> Vec<f64> {
        match strategy {
            PlacementStrategy::ClockwiseSpiral => vec![0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
            PlacementStrategy::CounterclockwiseSpiral => vec![315.0, 270.0, 225.0, 180.0, 135.0, 90.0, 45.0, 0.0],
            PlacementStrategy::Grid => vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0],
            PlacementStrategy::Random => match n % 4 {
                0 => vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0],
                1 => vec![90.0, 270.0, 0.0, 180.0, 135.0, 315.0, 45.0, 225.0],
                2 => vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0],
                _ => vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0],
            },
            PlacementStrategy::BoundaryFirst => vec![45.0, 135.0, 225.0, 315.0, 0.0, 90.0, 180.0, 270.0],
        }
    }

    fn select_direction_for_strategy(&self, n: usize, width: f64, height: f64, strategy: PlacementStrategy, attempt: usize, rng: &mut impl Rng) -> f64 {
        match strategy {
            PlacementStrategy::ClockwiseSpiral => {
                let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
                ((n as f64 * golden_angle) + (attempt as f64 / self.config.search_attempts as f64) * 2.0 * PI) % (2.0 * PI)
            }
            PlacementStrategy::CounterclockwiseSpiral => {
                let golden_angle = -PI * (3.0 - (5.0_f64).sqrt());
                ((n as f64 * golden_angle).rem_euclid(2.0 * PI) - (attempt as f64 / self.config.search_attempts as f64) * 2.0 * PI).rem_euclid(2.0 * PI)
            }
            PlacementStrategy::Grid => (attempt % 16) as f64 / 16.0 * 2.0 * PI + rng.gen_range(-0.03..0.03),
            PlacementStrategy::Random => {
                if rng.gen::<f64>() < 0.5 { rng.gen_range(0.0..2.0 * PI) }
                else if width < height { (if rng.gen() { 0.0 } else { PI }) + rng.gen_range(-PI / 3.0..PI / 3.0) }
                else { (if rng.gen() { PI / 2.0 } else { -PI / 2.0 }) + rng.gen_range(-PI / 3.0..PI / 3.0) }
            }
            PlacementStrategy::BoundaryFirst => {
                let prob = rng.gen::<f64>();
                if prob < 0.4 { [PI / 4.0, 3.0 * PI / 4.0, 5.0 * PI / 4.0, 7.0 * PI / 4.0][attempt % 4] + rng.gen_range(-0.1..0.1) }
                else if prob < 0.8 { [0.0, PI / 2.0, PI, 3.0 * PI / 2.0][attempt % 4] + rng.gen_range(-0.2..0.2) }
                else { rng.gen_range(0.0..2.0 * PI) }
            }
        }
    }

    fn placement_score(&self, tree: &PlacedTree, existing: &[PlacedTree], n: usize) -> f64 {
        let (tree_min_x, tree_min_y, tree_max_x, tree_max_y) = tree.bounds();
        let (mut pack_min_x, mut pack_min_y, mut pack_max_x, mut pack_max_y) = (tree_min_x, tree_min_y, tree_max_x, tree_max_y);
        for t in existing {
            let (bx1, by1, bx2, by2) = t.bounds();
            pack_min_x = pack_min_x.min(bx1); pack_min_y = pack_min_y.min(by1);
            pack_max_x = pack_max_x.max(bx2); pack_max_y = pack_max_y.max(by2);
        }
        let (width, height) = (pack_max_x - pack_min_x, pack_max_y - pack_min_y);
        let side = width.max(height);
        let balance_penalty = (width - height).abs() * 0.10;
        let local_density = self.calculate_local_density((tree_min_x + tree_max_x) / 2.0, (tree_min_y + tree_max_y) / 2.0, existing);
        let density_bonus = -self.config.gap_penalty_weight * local_density;
        let (old_min_x, old_min_y, old_max_x, old_max_y) = if !existing.is_empty() { compute_bounds(existing) } else { (0.0, 0.0, 0.0, 0.0) };
        let extension_penalty = ((pack_max_x - old_max_x).max(0.0) + (old_min_x - pack_min_x).max(0.0) + (pack_max_y - old_max_y).max(0.0) + (old_min_y - pack_min_y).max(0.0)) * 0.08;
        let gap_penalty = self.estimate_unusable_gap(tree, existing) * self.config.gap_penalty_weight;
        let center_penalty = ((pack_min_x + pack_max_x) / 2.0).abs() + ((pack_min_y + pack_max_y) / 2.0).abs() * 0.005 / (n as f64).sqrt();
        let neighbor_bonus = self.neighbor_proximity_bonus(tree, existing);
        side + balance_penalty + extension_penalty + gap_penalty + center_penalty + density_bonus - neighbor_bonus
    }

    fn calculate_local_density(&self, cx: f64, cy: f64, trees: &[PlacedTree]) -> f64 {
        let radius_sq = self.config.local_density_radius.powi(2);
        trees.iter().map(|tree| {
            let (bx1, by1, bx2, by2) = tree.bounds();
            let dist_sq = ((bx1 + bx2) / 2.0 - cx).powi(2) + ((by1 + by2) / 2.0 - cy).powi(2);
            if dist_sq < radius_sq { 1.0 - (dist_sq / radius_sq).sqrt() } else { 0.0 }
        }).sum()
    }

    fn estimate_unusable_gap(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        if existing.is_empty() { return 0.0; }
        let (tree_min_x, tree_min_y, tree_max_x, tree_max_y) = tree.bounds();
        let (min_useful, max_wasteful) = (0.15, 0.4);
        existing.iter().map(|other| {
            let (ox1, oy1, ox2, oy2) = other.bounds();
            let mut penalty = 0.0;
            if tree_min_y < oy2 && tree_max_y > oy1 {
                if tree_min_x > ox2 { let gap = tree_min_x - ox2; if gap > min_useful && gap < max_wasteful { penalty += (max_wasteful - gap) / max_wasteful * 0.1; } }
                else if tree_max_x < ox1 { let gap = ox1 - tree_max_x; if gap > min_useful && gap < max_wasteful { penalty += (max_wasteful - gap) / max_wasteful * 0.1; } }
            }
            if tree_min_x < ox2 && tree_max_x > ox1 {
                if tree_min_y > oy2 { let gap = tree_min_y - oy2; if gap > min_useful && gap < max_wasteful { penalty += (max_wasteful - gap) / max_wasteful * 0.1; } }
                else if tree_max_y < oy1 { let gap = oy1 - tree_max_y; if gap > min_useful && gap < max_wasteful { penalty += (max_wasteful - gap) / max_wasteful * 0.1; } }
            }
            penalty
        }).sum()
    }

    fn neighbor_proximity_bonus(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        if existing.is_empty() { return 0.0; }
        let (tree_min_x, tree_min_y, tree_max_x, tree_max_y) = tree.bounds();
        let (tree_cx, tree_cy) = ((tree_min_x + tree_max_x) / 2.0, (tree_min_y + tree_max_y) / 2.0);
        let (mut min_dist, mut close_neighbors) = (f64::INFINITY, 0);
        for other in existing {
            let (ox1, oy1, ox2, oy2) = other.bounds();
            let dist = ((ox1 + ox2) / 2.0 - tree_cx).hypot((oy1 + oy2) / 2.0 - tree_cy);
            min_dist = min_dist.min(dist);
            if dist < 0.8 { close_neighbors += 1; }
        }
        (if min_dist < 1.5 { 0.02 * (1.5 - min_dist) } else { 0.0 }) + 0.005 * close_neighbors as f64
    }

    fn find_gaps(&self, trees: &[PlacedTree], min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<(f64, f64, f64, f64)> {
        if trees.is_empty() { return Vec::new(); }
        let grid_res = self.config.density_grid_resolution;
        let (cell_w, cell_h) = ((max_x - min_x) / grid_res as f64, (max_y - min_y) / grid_res as f64);
        if cell_w <= 0.0 || cell_h <= 0.0 { return Vec::new(); }
        let mut occupied = vec![false; grid_res * grid_res];
        for tree in trees {
            let (bx1, by1, bx2, by2) = tree.bounds();
            for i in ((bx1 - min_x) / cell_w).floor().max(0.0) as usize..((bx2 - min_x) / cell_w).ceil().min(grid_res as f64) as usize {
                for j in ((by1 - min_y) / cell_h).floor().max(0.0) as usize..((by2 - min_y) / cell_h).ceil().min(grid_res as f64) as usize {
                    if i < grid_res && j < grid_res { occupied[j * grid_res + i] = true; }
                }
            }
        }
        let mut gaps = Vec::new();
        for i in 1..grid_res - 1 {
            for j in 1..grid_res - 1 {
                let idx = j * grid_res + i;
                if !occupied[idx] && (occupied[(j - 1) * grid_res + i] as i32 + occupied[(j + 1) * grid_res + i] as i32 + occupied[j * grid_res + i - 1] as i32 + occupied[j * grid_res + i + 1] as i32) >= 2 {
                    gaps.push((min_x + i as f64 * cell_w, min_y + j as f64 * cell_h, min_x + (i + 1) as f64 * cell_w, min_y + (j + 1) as f64 * cell_h));
                }
            }
        }
        gaps
    }

    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, pass: usize, _strategy: PlacementStrategy, rng: &mut impl Rng) {
        if trees.len() <= 1 { return; }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();
        let mut elite_pool: Vec<(f64, Vec<PlacedTree>)> = vec![(current_side, trees.clone())];

        let temp_multiplier = match pass { 0 => 1.0, _ => 0.35 };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        let base_iterations = match pass {
            0 => self.config.sa_iterations + n * 100,
            _ => self.config.sa_iterations / 2 + n * 50,
        };

        let (mut iterations_without_improvement, mut total_restarts) = (0, 0);
        let max_restarts = 4;
        let (mut boundary_cache_iter, mut boundary_info) = (0, Vec::new());

        for iter in 0..base_iterations {
            if iterations_without_improvement >= self.config.hot_restart_interval && total_restarts < max_restarts {
                let elite_idx = rng.gen_range(0..elite_pool.len());
                *trees = elite_pool[elite_idx].1.clone();
                current_side = elite_pool[elite_idx].0;
                temp = self.config.hot_restart_temp;
                iterations_without_improvement = 0;
                total_restarts += 1;
                boundary_cache_iter = 0;
            }

            if iterations_without_improvement >= self.config.early_exit_threshold && total_restarts >= max_restarts { break; }

            if iter == 0 || iter - boundary_cache_iter >= 300 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                boundary_cache_iter = iter;
            }

            let do_fill_move = rng.gen::<f64>() < self.config.fill_move_prob;
            let (idx, edge) = if do_fill_move {
                let interior_trees: Vec<usize> = (0..trees.len()).filter(|&i| !boundary_info.iter().any(|(bi, _)| *bi == i)).collect();
                if !interior_trees.is_empty() && rng.gen::<f64>() < 0.5 { (interior_trees[rng.gen_range(0..interior_trees.len())], BoundaryEdge::None) }
                else if !boundary_info.is_empty() { (boundary_info[rng.gen_range(0..boundary_info.len())].0, boundary_info[rng.gen_range(0..boundary_info.len())].1) }
                else { (rng.gen_range(0..trees.len()), BoundaryEdge::None) }
            } else if !boundary_info.is_empty() && rng.gen::<f64>() < self.config.boundary_focus_prob {
                (boundary_info[rng.gen_range(0..boundary_info.len())].0, boundary_info[rng.gen_range(0..boundary_info.len())].1)
            } else { (rng.gen_range(0..trees.len()), BoundaryEdge::None) };

            let old_tree = trees[idx].clone();
            if self.sa_move(trees, idx, temp, edge, do_fill_move, rng) {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;
                if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                    current_side = new_side;
                    if current_side < best_side {
                        best_side = current_side;
                        best_config = trees.clone();
                        iterations_without_improvement = 0;
                        self.update_elite_pool(&mut elite_pool, current_side, trees.clone());
                    } else { iterations_without_improvement += 1; }
                } else { trees[idx] = old_tree; iterations_without_improvement += 1; }
            } else { trees[idx] = old_tree; iterations_without_improvement += 1; }

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
        }

        if best_side < compute_side_length(trees) { *trees = best_config; }
    }

    fn update_elite_pool(&self, pool: &mut Vec<(f64, Vec<PlacedTree>)>, score: f64, config: Vec<PlacedTree>) {
        if !pool.iter().any(|(s, _)| *s <= score) {
            pool.push((score, config));
            pool.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            pool.truncate(self.config.elite_pool_size);
        } else if pool.len() < self.config.elite_pool_size { pool.push((score, config)); }
    }

    fn find_boundary_trees_with_edges(&self, trees: &[PlacedTree]) -> Vec<(usize, BoundaryEdge)> {
        if trees.is_empty() { return Vec::new(); }
        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let eps = 0.015;
        trees.iter().enumerate().filter_map(|(i, tree)| {
            let (bx1, by1, bx2, by2) = tree.bounds();
            let (on_left, on_right) = ((bx1 - min_x).abs() < eps, (bx2 - max_x).abs() < eps);
            let (on_bottom, on_top) = ((by1 - min_y).abs() < eps, (by2 - max_y).abs() < eps);
            match (on_left, on_right, on_top, on_bottom) {
                (true, true, _, _) | (_, _, true, true) | (true, _, true, _) | (true, _, _, true) | (_, true, true, _) | (_, true, _, true) => Some((i, BoundaryEdge::Corner)),
                (true, false, false, false) => Some((i, BoundaryEdge::Left)),
                (false, true, false, false) => Some((i, BoundaryEdge::Right)),
                (false, false, true, false) => Some((i, BoundaryEdge::Top)),
                (false, false, false, true) => Some((i, BoundaryEdge::Bottom)),
                _ => None,
            }
        }).collect()
    }

    fn sa_move(&self, trees: &mut [PlacedTree], idx: usize, temp: f64, edge: BoundaryEdge, is_fill_move: bool, rng: &mut impl Rng) -> bool {
        let (old_x, old_y, old_angle) = (trees[idx].x, trees[idx].y, trees[idx].angle_deg);
        let scale = self.config.translation_scale * (0.3 + temp * 1.5);

        if is_fill_move {
            let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
            let (bbox_cx, bbox_cy) = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0);
            match rng.gen_range(0..4) {
                0 => { trees[idx] = PlacedTree::new(old_x + (bbox_cx - old_x) * 0.1 * (0.5 + temp), old_y + (bbox_cy - old_y) * 0.1 * (0.5 + temp), old_angle); }
                1 => { trees[idx] = PlacedTree::new(old_x + rng.gen_range(-scale * 0.4..scale * 0.4), old_y + rng.gen_range(-scale * 0.4..scale * 0.4), old_angle); }
                2 => { let angles = [45.0, 90.0, -45.0, -90.0, 30.0, -30.0]; trees[idx] = PlacedTree::new(old_x, old_y, (old_angle + angles[rng.gen_range(0..6)]).rem_euclid(360.0)); }
                _ => {
                    let gaps = self.find_gaps(trees, min_x, min_y, max_x, max_y);
                    if !gaps.is_empty() { let g = &gaps[rng.gen_range(0..gaps.len())]; trees[idx] = PlacedTree::new(old_x + ((g.0 + g.2) / 2.0 - old_x) * 0.05, old_y + ((g.1 + g.3) / 2.0 - old_y) * 0.05, old_angle); }
                    else { return false; }
                }
            }
        } else {
            let move_type = match edge {
                BoundaryEdge::Left => [0,0,0,0,0,1,1,2,2,3][rng.gen_range(0..10)],
                BoundaryEdge::Right => [4,4,4,4,4,1,1,2,2,3][rng.gen_range(0..10)],
                BoundaryEdge::Top => [5,5,5,5,5,6,6,2,2,3][rng.gen_range(0..10)],
                BoundaryEdge::Bottom => [7,7,7,7,7,6,6,2,2,3][rng.gen_range(0..10)],
                BoundaryEdge::Corner => [8,8,8,8,8,2,2,9,9,3][rng.gen_range(0..10)],
                BoundaryEdge::None => rng.gen_range(0..10),
            };
            match move_type {
                0 => { trees[idx] = PlacedTree::new(old_x + rng.gen_range(scale * 0.3..scale), old_y + rng.gen_range(-scale * 0.2..scale * 0.2), old_angle); }
                1 => { trees[idx] = PlacedTree::new(old_x, old_y + rng.gen_range(-scale..scale), old_angle); }
                2 => { let angles = [45.0, 90.0, -45.0, -90.0]; trees[idx] = PlacedTree::new(old_x, old_y, (old_angle + angles[rng.gen_range(0..4)]).rem_euclid(360.0)); }
                3 => { trees[idx] = PlacedTree::new(old_x + rng.gen_range(-scale * 0.5..scale * 0.5), old_y + rng.gen_range(-scale * 0.5..scale * 0.5), old_angle); }
                4 => { trees[idx] = PlacedTree::new(old_x + rng.gen_range(-scale..-scale * 0.3), old_y + rng.gen_range(-scale * 0.2..scale * 0.2), old_angle); }
                5 => { trees[idx] = PlacedTree::new(old_x + rng.gen_range(-scale * 0.2..scale * 0.2), old_y + rng.gen_range(-scale..-scale * 0.3), old_angle); }
                6 => { trees[idx] = PlacedTree::new(old_x + rng.gen_range(-scale..scale), old_y, old_angle); }
                7 => { trees[idx] = PlacedTree::new(old_x + rng.gen_range(-scale * 0.2..scale * 0.2), old_y + rng.gen_range(scale * 0.3..scale), old_angle); }
                8 => { let (min_x, min_y, max_x, max_y) = compute_bounds(trees); let (cx, cy) = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0); trees[idx] = PlacedTree::new(old_x + (cx - old_x) * self.config.center_pull_strength * (0.5 + temp), old_y + (cy - old_y) * self.config.center_pull_strength * (0.5 + temp), old_angle); }
                9 => { let d = rng.gen_range(-scale..scale); trees[idx] = PlacedTree::new(old_x + d, old_y + if rng.gen() { d } else { -d }, old_angle); }
                _ => { let mag = old_x.hypot(old_y); if mag > 0.08 { let new_mag = (mag + rng.gen_range(-0.06..0.06) * (1.0 + temp)).max(0.0); trees[idx] = PlacedTree::new(old_x * new_mag / mag, old_y * new_mag / mag, old_angle); } else { return false; } }
            }
        }
        !has_overlap(trees, idx)
    }
}

fn is_valid(tree: &PlacedTree, existing: &[PlacedTree]) -> bool { existing.iter().all(|o| !tree.overlaps(o)) }
fn compute_side_length(trees: &[PlacedTree]) -> f64 { if trees.is_empty() { 0.0 } else { let b = compute_bounds(trees); (b.2 - b.0).max(b.3 - b.1) } }
fn compute_bounds(trees: &[PlacedTree]) -> (f64, f64, f64, f64) {
    trees.iter().fold((f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY), |acc, t| {
        let b = t.bounds(); (acc.0.min(b.0), acc.1.min(b.1), acc.2.max(b.2), acc.3.max(b.3))
    })
}
fn has_overlap(trees: &[PlacedTree], idx: usize) -> bool { trees.iter().enumerate().any(|(i, t)| i != idx && trees[idx].overlaps(t)) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculate_score;
    #[test]
    fn test_evolved_packer() { let p = EvolvedPacker::default(); let packings = p.pack_all(20); for (i, p) in packings.iter().enumerate() { assert_eq!(p.trees.len(), i + 1); assert!(!p.has_overlaps()); } }
    #[test]
    fn test_evolved_score() { let p = EvolvedPacker::default(); let packings = p.pack_all(50); println!("Score: {:.4}", calculate_score(&packings)); }
}
