//! Evolved Packing Algorithm - Generation 91b ROTATION-FIRST OPTIMIZATION
//!
//! MUTATION: Exhaustive rotation search at each candidate position
//!
//! Strategy: For each candidate position found via binary search,
//!           try ALL 8 rotations and keep the best (position, rotation) pair.
//!           This is O(8x) slower but potentially much better placements.
//!
//! Parent: Gen87d (greedy backtracking wave)

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub enum PlacementStrategy {
    ClockwiseSpiral,
    CounterclockwiseSpiral,
    Grid,
    Random,
    BoundaryFirst,
    ConcentricRings,
}

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
    pub compression_prob: f64,
    pub wave_passes: usize,
    pub late_stage_threshold: usize,
    pub fine_angle_step: f64,
    pub swap_prob: f64,  // GEN90c: Probability of boundary swap operation
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
            center_pull_strength: 0.08,
            sa_passes: 2,
            early_exit_threshold: 2500,
            boundary_focus_prob: 0.85,
            num_strategies: 6,
            density_grid_resolution: 20,
            gap_penalty_weight: 0.15,
            local_density_radius: 0.5,
            fill_move_prob: 0.15,
            hot_restart_interval: 800,
            hot_restart_temp: 0.35,
            elite_pool_size: 3,
            compression_prob: 0.20,
            wave_passes: 5,
            late_stage_threshold: 140,
            fine_angle_step: 15.0,
            swap_prob: 0.0,  // GEN90d: Disabled swap (didn't help)
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BoundaryEdge {
    Left, Right, Top, Bottom, Corner, None,
}

pub struct EvolvedPacker {
    pub config: EvolvedConfig,
}

impl Default for EvolvedPacker {
    fn default() -> Self {
        Self { config: EvolvedConfig::default() }
    }
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
            PlacementStrategy::ConcentricRings,
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

                self.wave_compaction(&mut trees);

                let side = compute_side_length(&trees);
                strategy_trees[s_idx] = trees.clone();

                if side < best_side {
                    best_side = side;
                    best_trees = Some(trees);
                }
            }

            let best = best_trees.unwrap();
            let mut packing = Packing::new();
            for t in &best {
                packing.trees.push(t.clone());
            }
            packings.push(packing);

            for strat_trees in strategy_trees.iter_mut() {
                if compute_side_length(strat_trees) > best_side * 1.02 {
                    *strat_trees = best.clone();
                }
            }
        }

        packings
    }

    // GEN87d: Find trees that are on the bounding box boundary
    fn find_boundary_defining_trees(&self, trees: &[PlacedTree]) -> Vec<(usize, BoundaryEdge)> {
        if trees.is_empty() {
            return Vec::new();
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let eps = 0.001; // Tighter tolerance for boundary-defining trees

        let mut boundary_trees = Vec::new();

        for (i, tree) in trees.iter().enumerate() {
            let (bx1, by1, bx2, by2) = tree.bounds();

            // Check if this tree defines any edge of the bounding box
            if (bx1 - min_x).abs() < eps {
                boundary_trees.push((i, BoundaryEdge::Left));
            }
            if (bx2 - max_x).abs() < eps {
                boundary_trees.push((i, BoundaryEdge::Right));
            }
            if (by1 - min_y).abs() < eps {
                boundary_trees.push((i, BoundaryEdge::Bottom));
            }
            if (by2 - max_y).abs() < eps {
                boundary_trees.push((i, BoundaryEdge::Top));
            }
        }

        boundary_trees
    }

    fn wave_compaction(&self, trees: &mut Vec<PlacedTree>) {
        if trees.len() <= 1 {
            return;
        }

        // GEN84c base: EXTREME SPLIT - outside-in first (4), then inside-out (1)
        for wave in 0..self.config.wave_passes {
            let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
            let center_x = (min_x + max_x) / 2.0;
            let center_y = (min_y + max_y) / 2.0;

            // Calculate distances from center
            let mut tree_distances: Vec<(usize, f64)> = trees.iter().enumerate()
                .map(|(i, t)| {
                    let dx = t.x - center_x;
                    let dy = t.y - center_y;
                    (i, (dx * dx + dy * dy).sqrt())
                })
                .collect();

            // CROSSOVER EXTREME: First 4 waves outside-in, last 1 wave inside-out
            if wave < 4 {
                // Outside-in: far trees first (descending)
                tree_distances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            } else {
                // Inside-out: close trees first (ascending) - final settling pass
                tree_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            }

            // Phase 1: Move RIGHT
            for &(idx, _) in &tree_distances {
                let old_x = trees[idx].x;
                let old_y = trees[idx].y;
                let old_angle = trees[idx].angle_deg;

                if old_x >= center_x { continue; }
                let dx = center_x - old_x;
                if dx < 0.02 { continue; }

                for step in [0.10, 0.05, 0.02, 0.01, 0.005] {
                    let new_x = old_x + dx * step;
                    trees[idx] = PlacedTree::new(new_x, old_y, old_angle);
                    if has_overlap(trees, idx) {
                        trees[idx] = PlacedTree::new(old_x, old_y, old_angle);
                    } else {
                        break;
                    }
                }
            }

            // Phase 2: Move LEFT
            for &(idx, _) in &tree_distances {
                let old_x = trees[idx].x;
                let old_y = trees[idx].y;
                let old_angle = trees[idx].angle_deg;

                if old_x <= center_x { continue; }
                let dx = old_x - center_x;
                if dx < 0.02 { continue; }

                for step in [0.10, 0.05, 0.02, 0.01, 0.005] {
                    let new_x = old_x - dx * step;
                    trees[idx] = PlacedTree::new(new_x, old_y, old_angle);
                    if has_overlap(trees, idx) {
                        trees[idx] = PlacedTree::new(old_x, old_y, old_angle);
                    } else {
                        break;
                    }
                }
            }

            // Phase 3: Move UP
            for &(idx, _) in &tree_distances {
                let old_x = trees[idx].x;
                let old_y = trees[idx].y;
                let old_angle = trees[idx].angle_deg;

                if old_y >= center_y { continue; }
                let dy = center_y - old_y;
                if dy < 0.02 { continue; }

                for step in [0.10, 0.05, 0.02, 0.01, 0.005] {
                    let new_y = old_y + dy * step;
                    trees[idx] = PlacedTree::new(old_x, new_y, old_angle);
                    if has_overlap(trees, idx) {
                        trees[idx] = PlacedTree::new(old_x, old_y, old_angle);
                    } else {
                        break;
                    }
                }
            }

            // Phase 4: Move DOWN
            for &(idx, _) in &tree_distances {
                let old_x = trees[idx].x;
                let old_y = trees[idx].y;
                let old_angle = trees[idx].angle_deg;

                if old_y <= center_y { continue; }
                let dy = old_y - center_y;
                if dy < 0.02 { continue; }

                for step in [0.10, 0.05, 0.02, 0.01, 0.005] {
                    let new_y = old_y - dy * step;
                    trees[idx] = PlacedTree::new(old_x, new_y, old_angle);
                    if has_overlap(trees, idx) {
                        trees[idx] = PlacedTree::new(old_x, old_y, old_angle);
                    } else {
                        break;
                    }
                }
            }

            // Phase 5: Diagonal movement (unchanged)
            for (idx, _dist) in tree_distances {
                let old_x = trees[idx].x;
                let old_y = trees[idx].y;
                let old_angle = trees[idx].angle_deg;
                let dx = center_x - old_x;
                let dy = center_y - old_y;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist < 0.05 { continue; }

                for step in [0.10, 0.05, 0.02, 0.01, 0.005] {
                    let new_x = old_x + dx * step;
                    let new_y = old_y + dy * step;
                    trees[idx] = PlacedTree::new(new_x, new_y, old_angle);
                    if has_overlap(trees, idx) {
                        trees[idx] = PlacedTree::new(old_x, old_y, old_angle);
                    } else {
                        break;
                    }
                }
            }
        }

        // GEN87d: GREEDY BACKTRACKING PASS
        // Focus on boundary-defining trees and try aggressive inward moves
        for _greedy_pass in 0..3 { // Multiple greedy passes
            let boundary_trees = self.find_boundary_defining_trees(trees);
            let current_side = compute_side_length(trees);

            for (idx, edge) in boundary_trees {
                let old_x = trees[idx].x;
                let old_y = trees[idx].y;
                let old_angle = trees[idx].angle_deg;

                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let center_x = (min_x + max_x) / 2.0;
                let center_y = (min_y + max_y) / 2.0;

                // Determine movement direction based on which edge this tree defines
                let (dx, dy) = match edge {
                    BoundaryEdge::Left => (0.1, 0.0),    // Move right
                    BoundaryEdge::Right => (-0.1, 0.0),  // Move left
                    BoundaryEdge::Top => (0.0, -0.1),    // Move down
                    BoundaryEdge::Bottom => (0.0, 0.1),  // Move up
                    BoundaryEdge::Corner => {
                        // Move toward center
                        let dx = center_x - old_x;
                        let dy = center_y - old_y;
                        let dist = (dx * dx + dy * dy).sqrt();
                        if dist > 0.01 {
                            (dx / dist * 0.1, dy / dist * 0.1)
                        } else {
                            continue;
                        }
                    }
                    BoundaryEdge::None => continue,
                };

                // Try aggressive movement with multiple step sizes
                let mut success = false;
                for scale in [1.0, 0.5, 0.25, 0.1, 0.05] {
                    let new_x = old_x + dx * scale;
                    let new_y = old_y + dy * scale;
                    trees[idx] = PlacedTree::new(new_x, new_y, old_angle);

                    if !has_overlap(trees, idx) {
                        let new_side = compute_side_length(trees);
                        if new_side < current_side {
                            success = true;
                            break;
                        }
                    }
                    // Revert
                    trees[idx] = PlacedTree::new(old_x, old_y, old_angle);
                }

                // If movement failed, try with rotation
                if !success {
                    for rot_delta in [45.0, -45.0, 90.0, -90.0] {
                        let new_angle = (old_angle + rot_delta).rem_euclid(360.0);

                        for scale in [1.0, 0.5, 0.25, 0.1] {
                            let new_x = old_x + dx * scale;
                            let new_y = old_y + dy * scale;
                            trees[idx] = PlacedTree::new(new_x, new_y, new_angle);

                            if !has_overlap(trees, idx) {
                                let new_side = compute_side_length(trees);
                                if new_side < current_side {
                                    success = true;
                                    break;
                                }
                            }
                            trees[idx] = PlacedTree::new(old_x, old_y, old_angle);
                        }

                        if success { break; }
                    }
                }
            }
        }
    }

    fn find_placement_with_strategy(
        &self,
        existing: &[PlacedTree],
        n: usize,
        _max_n: usize,
        strategy: PlacementStrategy,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            let initial_angle = match strategy {
                PlacementStrategy::ClockwiseSpiral => 0.0,
                PlacementStrategy::CounterclockwiseSpiral => 90.0,
                PlacementStrategy::Grid => 45.0,
                PlacementStrategy::Random => rng.gen_range(0..8) as f64 * 45.0,
                PlacementStrategy::BoundaryFirst => 180.0,
                PlacementStrategy::ConcentricRings => 45.0,
            };
            return PlacedTree::new(0.0, 0.0, initial_angle);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 90.0);
        let mut best_score = f64::INFINITY;

        // GEN91b: All 8 standard rotations for exhaustive search
        let all_rotations = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0];

        let (min_x, min_y, max_x, max_y) = compute_bounds(existing);
        let current_width = max_x - min_x;
        let current_height = max_y - min_y;

        let gaps = self.find_gaps(existing, min_x, min_y, max_x, max_y);

        for attempt in 0..self.config.search_attempts {
            let dir = if !gaps.is_empty() && attempt % 5 == 0 {
                let gap = &gaps[attempt % gaps.len()];
                let gap_cx = (gap.0 + gap.2) / 2.0;
                let gap_cy = (gap.1 + gap.3) / 2.0;
                gap_cy.atan2(gap_cx)
            } else {
                self.select_direction_for_strategy(n, current_width, current_height, strategy, attempt, rng)
            };

            let vx = dir.cos();
            let vy = dir.sin();

            // GEN91b: First find a valid position using any rotation
            // Use angle 0 as probe to find approximate valid distance
            let mut probe_low = 0.0;
            let mut probe_high = 12.0;
            while probe_high - probe_low > 0.01 {
                let mid = (probe_low + probe_high) / 2.0;
                // Try any rotation to see if this distance works
                let mut any_valid = false;
                for &angle in &all_rotations {
                    let candidate = PlacedTree::new(mid * vx, mid * vy, angle);
                    if is_valid(&candidate, existing) {
                        any_valid = true;
                        break;
                    }
                }
                if any_valid {
                    probe_high = mid;
                } else {
                    probe_low = mid;
                }
            }

            // GEN91b: Now at this approximate position, try ALL 8 rotations
            // with fine-tuned positioning for each
            for &tree_angle in &all_rotations {
                // Fine-tune the distance for this specific rotation
                let mut low = (probe_high - 0.5).max(0.0);
                let mut high = probe_high + 0.5;

                while high - low > 0.001 {
                    let mid = (low + high) / 2.0;
                    let candidate = PlacedTree::new(mid * vx, mid * vy, tree_angle);

                    if is_valid(&candidate, existing) {
                        high = mid;
                    } else {
                        low = mid;
                    }
                }

                let candidate = PlacedTree::new(high * vx, high * vy, tree_angle);
                if is_valid(&candidate, existing) {
                    let score = self.placement_score(&candidate, existing, n);
                    if score < best_score {
                        best_score = score;
                        best_tree = candidate;
                    }
                }
            }
        }

        best_tree
    }

    #[inline]
    fn select_angles_for_strategy(&self, n: usize, strategy: PlacementStrategy) -> Vec<f64> {
        match strategy {
            PlacementStrategy::ClockwiseSpiral => {
                vec![0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
            }
            PlacementStrategy::CounterclockwiseSpiral => {
                vec![315.0, 270.0, 225.0, 180.0, 135.0, 90.0, 45.0, 0.0]
            }
            PlacementStrategy::Grid => {
                vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0]
            }
            PlacementStrategy::Random => {
                match n % 4 {
                    0 => vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0],
                    1 => vec![90.0, 270.0, 0.0, 180.0, 135.0, 315.0, 45.0, 225.0],
                    2 => vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0],
                    _ => vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0],
                }
            }
            PlacementStrategy::BoundaryFirst => {
                vec![45.0, 135.0, 225.0, 315.0, 0.0, 90.0, 180.0, 270.0]
            }
            PlacementStrategy::ConcentricRings => {
                if n % 2 == 0 {
                    vec![45.0, 135.0, 225.0, 315.0, 0.0, 90.0, 180.0, 270.0]
                } else {
                    vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0]
                }
            }
        }
    }

    #[inline]
    fn select_fine_angles_for_strategy(&self, n: usize, strategy: PlacementStrategy) -> Vec<f64> {
        let base_angles = self.select_angles_for_strategy(n, strategy);
        let mut fine_angles = Vec::with_capacity(24);

        for &angle in &base_angles {
            fine_angles.push(angle);
        }

        for &base in &base_angles {
            let plus_15 = (base + 15.0).rem_euclid(360.0);
            let minus_15 = (base - 15.0).rem_euclid(360.0);

            if !fine_angles.contains(&plus_15) {
                fine_angles.push(plus_15);
            }
            if !fine_angles.contains(&minus_15) {
                fine_angles.push(minus_15);
            }
        }

        for &base in &base_angles {
            let plus_30 = (base + 30.0).rem_euclid(360.0);
            let minus_30 = (base - 30.0).rem_euclid(360.0);

            if !fine_angles.contains(&plus_30) {
                fine_angles.push(plus_30);
            }
            if !fine_angles.contains(&minus_30) {
                fine_angles.push(minus_30);
            }
        }

        fine_angles
    }

    #[inline]
    fn select_direction_for_strategy(
        &self,
        n: usize,
        width: f64,
        height: f64,
        strategy: PlacementStrategy,
        attempt: usize,
        rng: &mut impl Rng,
    ) -> f64 {
        match strategy {
            PlacementStrategy::ClockwiseSpiral => {
                let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
                let base = (n as f64 * golden_angle) % (2.0 * PI);
                let offset = (attempt as f64 / self.config.search_attempts as f64) * 2.0 * PI;
                (base + offset) % (2.0 * PI)
            }
            PlacementStrategy::CounterclockwiseSpiral => {
                let golden_angle = -PI * (3.0 - (5.0_f64).sqrt());
                let base = (n as f64 * golden_angle).rem_euclid(2.0 * PI);
                let offset = (attempt as f64 / self.config.search_attempts as f64) * 2.0 * PI;
                (base - offset).rem_euclid(2.0 * PI)
            }
            PlacementStrategy::Grid => {
                let num_dirs = 16;
                let base_idx = attempt % num_dirs;
                let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
                base + rng.gen_range(-0.03..0.03)
            }
            PlacementStrategy::Random => {
                let mix = rng.gen::<f64>();
                if mix < 0.5 {
                    rng.gen_range(0.0..2.0 * PI)
                } else {
                    if width < height {
                        let angle = if rng.gen() { 0.0 } else { PI };
                        angle + rng.gen_range(-PI / 3.0..PI / 3.0)
                    } else {
                        let angle = if rng.gen() { PI / 2.0 } else { -PI / 2.0 };
                        angle + rng.gen_range(-PI / 3.0..PI / 3.0)
                    }
                }
            }
            PlacementStrategy::BoundaryFirst => {
                let prob = rng.gen::<f64>();
                if prob < 0.4 {
                    let corners = [PI / 4.0, 3.0 * PI / 4.0, 5.0 * PI / 4.0, 7.0 * PI / 4.0];
                    corners[attempt % 4] + rng.gen_range(-0.1..0.1)
                } else if prob < 0.8 {
                    let edges = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0];
                    edges[attempt % 4] + rng.gen_range(-0.2..0.2)
                } else {
                    rng.gen_range(0.0..2.0 * PI)
                }
            }
            PlacementStrategy::ConcentricRings => {
                let ring = ((n as f64).sqrt() as usize).max(1);
                let trees_in_ring = (ring * 6).max(1);
                let position_in_ring = n % trees_in_ring;
                let base_angle = (position_in_ring as f64 / trees_in_ring as f64) * 2.0 * PI;
                let offset = (attempt as f64 / self.config.search_attempts as f64) * 0.5 * PI;
                (base_angle + offset).rem_euclid(2.0 * PI)
            }
        }
    }

    #[inline]
    fn placement_score(&self, tree: &PlacedTree, existing: &[PlacedTree], n: usize) -> f64 {
        let (tree_min_x, tree_min_y, tree_max_x, tree_max_y) = tree.bounds();

        let mut pack_min_x = tree_min_x;
        let mut pack_min_y = tree_min_y;
        let mut pack_max_x = tree_max_x;
        let mut pack_max_y = tree_max_y;

        for t in existing {
            let (bx1, by1, bx2, by2) = t.bounds();
            pack_min_x = pack_min_x.min(bx1);
            pack_min_y = pack_min_y.min(by1);
            pack_max_x = pack_max_x.max(bx2);
            pack_max_y = pack_max_y.max(by2);
        }

        let width = pack_max_x - pack_min_x;
        let height = pack_max_y - pack_min_y;
        let side = width.max(height);

        let side_score = side;
        let balance_penalty = (width - height).abs() * 0.10;

        let tree_cx = (tree_min_x + tree_max_x) / 2.0;
        let tree_cy = (tree_min_y + tree_max_y) / 2.0;
        let local_density = self.calculate_local_density(tree_cx, tree_cy, existing);
        let density_bonus = -self.config.gap_penalty_weight * local_density;

        let (old_min_x, old_min_y, old_max_x, old_max_y) = if !existing.is_empty() {
            compute_bounds(existing)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        let x_extension = (pack_max_x - old_max_x).max(0.0) + (old_min_x - pack_min_x).max(0.0);
        let y_extension = (pack_max_y - old_max_y).max(0.0) + (old_min_y - pack_min_y).max(0.0);
        let extension_penalty = (x_extension + y_extension) * 0.08;

        let gap_penalty = self.estimate_unusable_gap(tree, existing) * self.config.gap_penalty_weight;

        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.005 / (n as f64).sqrt();

        let neighbor_bonus = self.neighbor_proximity_bonus(tree, existing);

        side_score + balance_penalty + extension_penalty + gap_penalty + center_penalty + density_bonus - neighbor_bonus
    }

    #[inline]
    fn calculate_local_density(&self, cx: f64, cy: f64, trees: &[PlacedTree]) -> f64 {
        let radius = self.config.local_density_radius;
        let radius_sq = radius * radius;
        let mut count = 0.0;

        for tree in trees {
            let (bx1, by1, bx2, by2) = tree.bounds();
            let tree_cx = (bx1 + bx2) / 2.0;
            let tree_cy = (by1 + by2) / 2.0;

            let dx = tree_cx - cx;
            let dy = tree_cy - cy;
            let dist_sq = dx * dx + dy * dy;

            if dist_sq < radius_sq {
                count += 1.0 - (dist_sq / radius_sq).sqrt();
            }
        }

        count
    }

    #[inline]
    fn estimate_unusable_gap(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        if existing.is_empty() {
            return 0.0;
        }

        let (tree_min_x, tree_min_y, tree_max_x, tree_max_y) = tree.bounds();

        let mut gap_penalty = 0.0;
        let min_useful_gap = 0.15;
        let max_wasteful_gap = 0.4;

        for other in existing {
            let (ox1, oy1, ox2, oy2) = other.bounds();

            if tree_min_y < oy2 && tree_max_y > oy1 {
                if tree_min_x > ox2 {
                    let gap = tree_min_x - ox2;
                    if gap > min_useful_gap && gap < max_wasteful_gap {
                        gap_penalty += (max_wasteful_gap - gap) / max_wasteful_gap * 0.1;
                    }
                } else if tree_max_x < ox1 {
                    let gap = ox1 - tree_max_x;
                    if gap > min_useful_gap && gap < max_wasteful_gap {
                        gap_penalty += (max_wasteful_gap - gap) / max_wasteful_gap * 0.1;
                    }
                }
            }

            if tree_min_x < ox2 && tree_max_x > ox1 {
                if tree_min_y > oy2 {
                    let gap = tree_min_y - oy2;
                    if gap > min_useful_gap && gap < max_wasteful_gap {
                        gap_penalty += (max_wasteful_gap - gap) / max_wasteful_gap * 0.1;
                    }
                } else if tree_max_y < oy1 {
                    let gap = oy1 - tree_max_y;
                    if gap > min_useful_gap && gap < max_wasteful_gap {
                        gap_penalty += (max_wasteful_gap - gap) / max_wasteful_gap * 0.1;
                    }
                }
            }
        }

        gap_penalty
    }

    #[inline]
    fn neighbor_proximity_bonus(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        if existing.is_empty() {
            return 0.0;
        }

        let (tree_min_x, tree_min_y, tree_max_x, tree_max_y) = tree.bounds();
        let tree_cx = (tree_min_x + tree_max_x) / 2.0;
        let tree_cy = (tree_min_y + tree_max_y) / 2.0;

        let mut min_dist = f64::INFINITY;
        let mut close_neighbors = 0;

        for other in existing {
            let (ox1, oy1, ox2, oy2) = other.bounds();
            let other_cx = (ox1 + ox2) / 2.0;
            let other_cy = (oy1 + oy2) / 2.0;

            let dx = tree_cx - other_cx;
            let dy = tree_cy - other_cy;
            let dist = (dx * dx + dy * dy).sqrt();

            min_dist = min_dist.min(dist);
            if dist < 0.8 {
                close_neighbors += 1;
            }
        }

        let dist_bonus = if min_dist < 1.5 { 0.02 * (1.5 - min_dist) } else { 0.0 };
        let neighbor_bonus = 0.005 * close_neighbors as f64;

        dist_bonus + neighbor_bonus
    }

    fn find_gaps(&self, trees: &[PlacedTree], min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<(f64, f64, f64, f64)> {
        if trees.is_empty() {
            return Vec::new();
        }

        let mut gaps = Vec::new();
        let grid_res = self.config.density_grid_resolution;
        let cell_w = (max_x - min_x) / grid_res as f64;
        let cell_h = (max_y - min_y) / grid_res as f64;

        if cell_w <= 0.0 || cell_h <= 0.0 {
            return Vec::new();
        }

        let mut occupied = vec![false; grid_res * grid_res];

        for tree in trees {
            let (bx1, by1, bx2, by2) = tree.bounds();
            let i1 = ((bx1 - min_x) / cell_w).floor().max(0.0) as usize;
            let i2 = ((bx2 - min_x) / cell_w).ceil().min(grid_res as f64) as usize;
            let j1 = ((by1 - min_y) / cell_h).floor().max(0.0) as usize;
            let j2 = ((by2 - min_y) / cell_h).ceil().min(grid_res as f64) as usize;

            for i in i1..i2.min(grid_res) {
                for j in j1..j2.min(grid_res) {
                    occupied[j * grid_res + i] = true;
                }
            }
        }

        for i in 1..grid_res - 1 {
            for j in 1..grid_res - 1 {
                let idx = j * grid_res + i;
                if !occupied[idx] {
                    let neighbors_occupied =
                        occupied[(j - 1) * grid_res + i] as i32 +
                        occupied[(j + 1) * grid_res + i] as i32 +
                        occupied[j * grid_res + i - 1] as i32 +
                        occupied[j * grid_res + i + 1] as i32;

                    if neighbors_occupied >= 2 {
                        let gx1 = min_x + i as f64 * cell_w;
                        let gy1 = min_y + j as f64 * cell_h;
                        let gx2 = gx1 + cell_w;
                        let gy2 = gy1 + cell_h;
                        gaps.push((gx1, gy1, gx2, gy2));
                    }
                }
            }
        }

        gaps
    }

    fn local_search(
        &self,
        trees: &mut Vec<PlacedTree>,
        n: usize,
        pass: usize,
        _strategy: PlacementStrategy,
        rng: &mut impl Rng,
    ) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        let mut elite_pool: Vec<(f64, Vec<PlacedTree>)> = vec![(current_side, trees.clone())];

        let temp_multiplier = match pass {
            0 => 1.0,
            _ => 0.35,
        };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        let base_iterations = match pass {
            0 => self.config.sa_iterations + n * 100,
            _ => self.config.sa_iterations / 2 + n * 50,
        };

        let mut iterations_without_improvement = 0;
        let mut total_restarts = 0;
        let max_restarts = 4;

        let mut boundary_cache_iter = 0;
        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();

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

            if iterations_without_improvement >= self.config.early_exit_threshold && total_restarts >= max_restarts {
                break;
            }

            if iter == 0 || iter - boundary_cache_iter >= 300 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                boundary_cache_iter = iter;
            }

            // GEN90c: Try swap before compression
            let do_swap = rng.gen::<f64>() < self.config.swap_prob;
            let do_compression = !do_swap && rng.gen::<f64>() < self.config.compression_prob;

            if do_swap {
                let old_trees = trees.clone();
                let success = self.swap_move(trees, &boundary_info, rng);

                if success {
                    let new_side = compute_side_length(trees);
                    let delta = new_side - current_side;

                    if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                        current_side = new_side;
                        if current_side < best_side {
                            best_side = current_side;
                            best_config = trees.clone();
                            iterations_without_improvement = 0;
                            self.update_elite_pool(&mut elite_pool, current_side, trees.clone());
                        } else {
                            iterations_without_improvement += 1;
                        }
                    } else {
                        *trees = old_trees;
                        iterations_without_improvement += 1;
                    }
                } else {
                    *trees = old_trees;
                    iterations_without_improvement += 1;
                }
            } else if do_compression {
                let old_trees = trees.clone();
                let success = self.compression_move(trees, rng);

                if success {
                    let new_side = compute_side_length(trees);
                    let delta = new_side - current_side;

                    if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                        current_side = new_side;
                        if current_side < best_side {
                            best_side = current_side;
                            best_config = trees.clone();
                            iterations_without_improvement = 0;
                            self.update_elite_pool(&mut elite_pool, current_side, trees.clone());
                        } else {
                            iterations_without_improvement += 1;
                        }
                    } else {
                        *trees = old_trees;
                        iterations_without_improvement += 1;
                    }
                } else {
                    *trees = old_trees;
                    iterations_without_improvement += 1;
                }
            } else {
                let do_fill_move = rng.gen::<f64>() < self.config.fill_move_prob;

                let (idx, edge) = if do_fill_move {
                    let interior_trees: Vec<usize> = (0..trees.len())
                        .filter(|&i| !boundary_info.iter().any(|(bi, _)| *bi == i))
                        .collect();

                    if !interior_trees.is_empty() && rng.gen::<f64>() < 0.5 {
                        (interior_trees[rng.gen_range(0..interior_trees.len())], BoundaryEdge::None)
                    } else if !boundary_info.is_empty() {
                        let bi = &boundary_info[rng.gen_range(0..boundary_info.len())];
                        (bi.0, bi.1)
                    } else {
                        (rng.gen_range(0..trees.len()), BoundaryEdge::None)
                    }
                } else if !boundary_info.is_empty() && rng.gen::<f64>() < self.config.boundary_focus_prob {
                    let bi = &boundary_info[rng.gen_range(0..boundary_info.len())];
                    (bi.0, bi.1)
                } else {
                    (rng.gen_range(0..trees.len()), BoundaryEdge::None)
                };

                let old_tree = trees[idx].clone();

                let success = self.sa_move(trees, idx, temp, edge, do_fill_move, rng);

                if success {
                    let new_side = compute_side_length(trees);
                    let delta = new_side - current_side;

                    if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                        current_side = new_side;
                        if current_side < best_side {
                            best_side = current_side;
                            best_config = trees.clone();
                            iterations_without_improvement = 0;

                            self.update_elite_pool(&mut elite_pool, current_side, trees.clone());
                        } else {
                            iterations_without_improvement += 1;
                        }
                    } else {
                        trees[idx] = old_tree;
                        iterations_without_improvement += 1;
                    }
                } else {
                    trees[idx] = old_tree;
                    iterations_without_improvement += 1;
                }
            }

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
        }

        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    fn compression_move(&self, trees: &mut [PlacedTree], rng: &mut impl Rng) -> bool {
        if trees.is_empty() {
            return false;
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;

        let idx = if rng.gen::<f64>() < 0.7 {
            let mut max_dist = 0.0;
            let mut max_idx = 0;
            for (i, tree) in trees.iter().enumerate() {
                let dx = tree.x - center_x;
                let dy = tree.y - center_y;
                let dist = dx * dx + dy * dy;
                if dist > max_dist {
                    max_dist = dist;
                    max_idx = i;
                }
            }
            max_idx
        } else {
            rng.gen_range(0..trees.len())
        };

        let old_x = trees[idx].x;
        let old_y = trees[idx].y;
        let old_angle = trees[idx].angle_deg;

        let dx = center_x - old_x;
        let dy = center_y - old_y;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < 0.01 {
            return false;
        }

        let compression_factor = rng.gen_range(0.02..0.08);
        let new_x = old_x + dx * compression_factor;
        let new_y = old_y + dy * compression_factor;

        trees[idx] = PlacedTree::new(new_x, new_y, old_angle);

        !has_overlap(trees, idx)
    }

    // GEN90c: Swap positions of two boundary trees
    fn swap_move(&self, trees: &mut [PlacedTree], boundary_info: &[(usize, BoundaryEdge)], rng: &mut impl Rng) -> bool {
        if boundary_info.len() < 2 {
            return false;
        }

        // Pick two different boundary trees
        let idx1_pos = rng.gen_range(0..boundary_info.len());
        let mut idx2_pos = rng.gen_range(0..boundary_info.len() - 1);
        if idx2_pos >= idx1_pos {
            idx2_pos += 1;
        }

        let (idx1, edge1) = boundary_info[idx1_pos];
        let (idx2, edge2) = boundary_info[idx2_pos];

        // Prefer swapping trees on opposite edges
        let is_opposite = matches!(
            (edge1, edge2),
            (BoundaryEdge::Left, BoundaryEdge::Right) |
            (BoundaryEdge::Right, BoundaryEdge::Left) |
            (BoundaryEdge::Top, BoundaryEdge::Bottom) |
            (BoundaryEdge::Bottom, BoundaryEdge::Top)
        );

        // 70% chance to only swap opposite edges, 30% swap any boundary pair
        if !is_opposite && rng.gen::<f64>() > 0.30 {
            return false;
        }

        // Save old positions
        let old_x1 = trees[idx1].x;
        let old_y1 = trees[idx1].y;
        let old_angle1 = trees[idx1].angle_deg;

        let old_x2 = trees[idx2].x;
        let old_y2 = trees[idx2].y;
        let old_angle2 = trees[idx2].angle_deg;

        // Swap positions (keep angles)
        trees[idx1] = PlacedTree::new(old_x2, old_y2, old_angle1);
        trees[idx2] = PlacedTree::new(old_x1, old_y1, old_angle2);

        // Check for overlaps
        if has_overlap(trees, idx1) || has_overlap(trees, idx2) {
            // Revert
            trees[idx1] = PlacedTree::new(old_x1, old_y1, old_angle1);
            trees[idx2] = PlacedTree::new(old_x2, old_y2, old_angle2);
            return false;
        }

        true
    }

    fn update_elite_pool(&self, pool: &mut Vec<(f64, Vec<PlacedTree>)>, score: f64, config: Vec<PlacedTree>) {
        let mut dominated = false;
        for (elite_score, _) in pool.iter() {
            if *elite_score <= score {
                dominated = true;
                break;
            }
        }

        if !dominated {
            pool.push((score, config));
            pool.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            pool.truncate(self.config.elite_pool_size);
        } else if pool.len() < self.config.elite_pool_size {
            pool.push((score, config));
        }
    }

    #[inline]
    fn find_boundary_trees_with_edges(&self, trees: &[PlacedTree]) -> Vec<(usize, BoundaryEdge)> {
        if trees.is_empty() {
            return Vec::new();
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let eps = 0.015;

        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();

        for (i, tree) in trees.iter().enumerate() {
            let (bx1, by1, bx2, by2) = tree.bounds();

            let on_left = (bx1 - min_x).abs() < eps;
            let on_right = (bx2 - max_x).abs() < eps;
            let on_bottom = (by1 - min_y).abs() < eps;
            let on_top = (by2 - max_y).abs() < eps;

            let edge = match (on_left, on_right, on_top, on_bottom) {
                (true, true, _, _) | (_, _, true, true) => BoundaryEdge::Corner,
                (true, _, true, _) | (true, _, _, true) => BoundaryEdge::Corner,
                (_, true, true, _) | (_, true, _, true) => BoundaryEdge::Corner,
                (true, false, false, false) => BoundaryEdge::Left,
                (false, true, false, false) => BoundaryEdge::Right,
                (false, false, true, false) => BoundaryEdge::Top,
                (false, false, false, true) => BoundaryEdge::Bottom,
                _ => continue,
            };

            boundary_info.push((i, edge));
        }

        boundary_info
    }

    #[inline]
    fn sa_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        edge: BoundaryEdge,
        is_fill_move: bool,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let scale = self.config.translation_scale * (0.3 + temp * 1.5);

        if is_fill_move {
            let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
            let bbox_cx = (min_x + max_x) / 2.0;
            let bbox_cy = (min_y + max_y) / 2.0;

            let move_type = rng.gen_range(0..4);
            match move_type {
                0 => {
                    let dx = (bbox_cx - old_x) * 0.1 * (0.5 + temp);
                    let dy = (bbox_cy - old_y) * 0.1 * (0.5 + temp);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                1 => {
                    let dx = rng.gen_range(-scale * 0.4..scale * 0.4);
                    let dy = rng.gen_range(-scale * 0.4..scale * 0.4);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                2 => {
                    let angles = [45.0, 90.0, -45.0, -90.0];
                    let delta = angles[rng.gen_range(0..angles.len())];
                    let new_angle = (old_angle + delta).rem_euclid(360.0);
                    trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
                }
                _ => {
                    let gaps = self.find_gaps(trees, min_x, min_y, max_x, max_y);
                    if !gaps.is_empty() {
                        let gap = &gaps[rng.gen_range(0..gaps.len())];
                        let gap_cx = (gap.0 + gap.2) / 2.0;
                        let gap_cy = (gap.1 + gap.3) / 2.0;
                        let dx = (gap_cx - old_x) * 0.05;
                        let dy = (gap_cy - old_y) * 0.05;
                        trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                    } else {
                        return false;
                    }
                }
            }
        } else {
            let move_type = match edge {
                BoundaryEdge::Left => {
                    match rng.gen_range(0..10) {
                        0..=4 => 0,
                        5..=6 => 1,
                        7..=8 => 2,
                        _ => 3,
                    }
                }
                BoundaryEdge::Right => {
                    match rng.gen_range(0..10) {
                        0..=4 => 4,
                        5..=6 => 1,
                        7..=8 => 2,
                        _ => 3,
                    }
                }
                BoundaryEdge::Top => {
                    match rng.gen_range(0..10) {
                        0..=4 => 5,
                        5..=6 => 6,
                        7..=8 => 2,
                        _ => 3,
                    }
                }
                BoundaryEdge::Bottom => {
                    match rng.gen_range(0..10) {
                        0..=4 => 7,
                        5..=6 => 6,
                        7..=8 => 2,
                        _ => 3,
                    }
                }
                BoundaryEdge::Corner => {
                    match rng.gen_range(0..10) {
                        0..=4 => 8,
                        5..=6 => 2,
                        7..=8 => 9,
                        _ => 3,
                    }
                }
                BoundaryEdge::None => {
                    rng.gen_range(0..10)
                }
            };

            match move_type {
                0 => {
                    let dx = rng.gen_range(scale * 0.3..scale);
                    let dy = rng.gen_range(-scale * 0.2..scale * 0.2);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                1 => {
                    let dy = rng.gen_range(-scale..scale);
                    trees[idx] = PlacedTree::new(old_x, old_y + dy, old_angle);
                }
                2 => {
                    let angles = [45.0, 90.0, -45.0, -90.0];
                    let delta = angles[rng.gen_range(0..angles.len())];
                    let new_angle = (old_angle + delta).rem_euclid(360.0);
                    trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
                }
                3 => {
                    let dx = rng.gen_range(-scale * 0.5..scale * 0.5);
                    let dy = rng.gen_range(-scale * 0.5..scale * 0.5);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                4 => {
                    let dx = rng.gen_range(-scale..-scale * 0.3);
                    let dy = rng.gen_range(-scale * 0.2..scale * 0.2);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                5 => {
                    let dx = rng.gen_range(-scale * 0.2..scale * 0.2);
                    let dy = rng.gen_range(-scale..-scale * 0.3);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                6 => {
                    let dx = rng.gen_range(-scale..scale);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y, old_angle);
                }
                7 => {
                    let dx = rng.gen_range(-scale * 0.2..scale * 0.2);
                    let dy = rng.gen_range(scale * 0.3..scale);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                8 => {
                    let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                    let bbox_cx = (min_x + max_x) / 2.0;
                    let bbox_cy = (min_y + max_y) / 2.0;

                    let dx = (bbox_cx - old_x) * self.config.center_pull_strength * (0.5 + temp);
                    let dy = (bbox_cy - old_y) * self.config.center_pull_strength * (0.5 + temp);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                9 => {
                    let diag = rng.gen_range(-scale..scale);
                    let sign = if rng.gen() { 1.0 } else { -1.0 };
                    trees[idx] = PlacedTree::new(old_x + diag, old_y + sign * diag, old_angle);
                }
                _ => {
                    let mag = (old_x * old_x + old_y * old_y).sqrt();
                    if mag > 0.08 {
                        let delta_r = rng.gen_range(-0.06..0.06) * (1.0 + temp);
                        let new_mag = (mag + delta_r).max(0.0);
                        let scale_r = new_mag / mag;
                        trees[idx] = PlacedTree::new(old_x * scale_r, old_y * scale_r, old_angle);
                    } else {
                        return false;
                    }
                }
            }
        }

        !has_overlap(trees, idx)
    }
}

fn is_valid(tree: &PlacedTree, existing: &[PlacedTree]) -> bool {
    for other in existing {
        if tree.overlaps(other) {
            return false;
        }
    }
    true
}

fn compute_side_length(trees: &[PlacedTree]) -> f64 {
    if trees.is_empty() {
        return 0.0;
    }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for tree in trees {
        let (bx1, by1, bx2, by2) = tree.bounds();
        min_x = min_x.min(bx1);
        min_y = min_y.min(by1);
        max_x = max_x.max(bx2);
        max_y = max_y.max(by2);
    }

    (max_x - min_x).max(max_y - min_y)
}

fn compute_bounds(trees: &[PlacedTree]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for tree in trees {
        let (bx1, by1, bx2, by2) = tree.bounds();
        min_x = min_x.min(bx1);
        min_y = min_y.min(by1);
        max_x = max_x.max(bx2);
        max_y = max_y.max(by2);
    }

    (min_x, min_y, max_x, max_y)
}

fn has_overlap(trees: &[PlacedTree], idx: usize) -> bool {
    for (i, tree) in trees.iter().enumerate() {
        if i != idx && trees[idx].overlaps(tree) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculate_score;

    #[test]
    fn test_evolved_packer() {
        let packer = EvolvedPacker::default();
        let packings = packer.pack_all(20);

        for (i, p) in packings.iter().enumerate() {
            assert_eq!(p.trees.len(), i + 1);
            assert!(!p.has_overlaps());
        }
    }

    #[test]
    fn test_evolved_score() {
        let packer = EvolvedPacker::default();
        let packings = packer.pack_all(50);
        let score = calculate_score(&packings);
        println!("Evolved score for n=1..50: {:.4}", score);
    }
}
