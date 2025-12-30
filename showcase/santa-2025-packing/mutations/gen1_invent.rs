//! Evolved Packing Algorithm - Generation 1 INVENT Mutation
//!
//! NOVEL APPROACH: Physics-Inspired Packing with Temperature Restart
//!
//! Key innovations:
//! 1. Hexagonal placement grid: Trees are initially placed considering a hex pattern
//!    for optimal circle-packing-like density
//! 2. Physics-inspired repulsion: Trees repel each other but are attracted to center,
//!    creating natural compaction during local search
//! 3. Temperature restart: When SA stagnates, restart with higher temp + greedy swaps
//! 4. Greedy swap operator: Periodically try swapping positions of pairs of trees
//! 5. Multi-pass local search: SA -> compact pass -> SA with lower temp
//!
//! This combines multiple novel ideas for improved packing density.

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Evolved packing configuration
/// These parameters are tuned through evolution
pub struct EvolvedConfig {
    // Search parameters
    pub search_attempts: usize,
    pub direction_samples: usize,

    // Simulated annealing
    pub sa_iterations: usize,
    pub sa_initial_temp: f64,
    pub sa_cooling_rate: f64,
    pub sa_min_temp: f64,

    // Move parameters
    pub translation_scale: f64,
    pub rotation_granularity: f64,
    pub center_pull_strength: f64,

    // NOVEL: Physics parameters
    pub repulsion_strength: f64,
    pub attraction_strength: f64,

    // NOVEL: Temperature restart parameters
    pub stagnation_threshold: usize,
    pub restart_temp_multiplier: f64,

    // NOVEL: Swap frequency
    pub swap_frequency: usize,
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        Self {
            search_attempts: 72,          // Increased for hex grid coverage
            direction_samples: 24,        // More directions for hex pattern
            sa_iterations: 6000,          // More iterations for multi-pass
            sa_initial_temp: 0.4,         // Slightly higher initial temp
            sa_cooling_rate: 0.9991,      // Slower cooling
            sa_min_temp: 0.0005,          // Lower minimum for fine-tuning
            translation_scale: 0.1,       // Slightly larger moves
            rotation_granularity: 30.0,   // Finer rotation steps
            center_pull_strength: 0.05,   // Stronger center pull

            // Physics parameters
            repulsion_strength: 0.02,
            attraction_strength: 0.015,

            // Restart parameters
            stagnation_threshold: 200,
            restart_temp_multiplier: 3.0,

            // Swap every N iterations
            swap_frequency: 100,
        }
    }
}

/// Main evolved packer
pub struct EvolvedPacker {
    pub config: EvolvedConfig,
}

impl Default for EvolvedPacker {
    fn default() -> Self {
        Self { config: EvolvedConfig::default() }
    }
}

impl EvolvedPacker {
    /// Pack all n from 1 to max_n
    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        let mut rng = rand::thread_rng();
        let mut packings: Vec<Packing> = Vec::with_capacity(max_n);
        let mut prev_trees: Vec<PlacedTree> = Vec::new();

        for n in 1..=max_n {
            let mut trees = prev_trees.clone();

            // Place new tree using evolved heuristics with hex grid awareness
            let new_tree = self.find_placement(&trees, n, max_n, &mut rng);
            trees.push(new_tree);

            // NOVEL: Multi-pass local search
            self.multi_pass_local_search(&mut trees, n, &mut rng);

            // Store result
            let mut packing = Packing::new();
            for t in &trees {
                packing.trees.push(t.clone());
            }
            packings.push(packing);
            prev_trees = trees;
        }

        packings
    }

    /// EVOLVED FUNCTION: Find best placement for new tree
    /// NOVEL: Considers hexagonal grid pattern for denser packing
    fn find_placement(
        &self,
        existing: &[PlacedTree],
        n: usize,
        _max_n: usize,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            return PlacedTree::new(0.0, 0.0, 90.0);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 90.0);
        let mut best_score = f64::INFINITY;

        let angles = self.select_angles(n);

        // NOVEL: Mix standard radial search with hexagonal grid points
        let hex_attempts = self.config.search_attempts / 3;
        let radial_attempts = self.config.search_attempts - hex_attempts;

        // Hexagonal grid-inspired placements
        for attempt in 0..hex_attempts {
            let hex_dir = self.hex_direction(attempt, n);
            let vx = hex_dir.cos();
            let vy = hex_dir.sin();

            for &tree_angle in &angles {
                if let Some((candidate, score)) = self.try_placement(
                    existing, vx, vy, tree_angle, n
                ) {
                    if score < best_score {
                        best_score = score;
                        best_tree = candidate;
                    }
                }
            }
        }

        // Standard radial search
        for _ in 0..radial_attempts {
            let dir = self.select_direction(n, rng);
            let vx = dir.cos();
            let vy = dir.sin();

            for &tree_angle in &angles {
                if let Some((candidate, score)) = self.try_placement(
                    existing, vx, vy, tree_angle, n
                ) {
                    if score < best_score {
                        best_score = score;
                        best_tree = candidate;
                    }
                }
            }
        }

        best_tree
    }

    /// NOVEL: Generate hexagonal grid direction
    #[inline]
    fn hex_direction(&self, attempt: usize, n: usize) -> f64 {
        // Hexagonal pattern: 6 primary directions at 60-degree intervals
        // with secondary offset layer
        let layer = attempt / 6;
        let slot = attempt % 6;

        // Base hex angle
        let base_angle = (slot as f64) * PI / 3.0;

        // Add offset for subsequent layers (30 degrees)
        let offset = if layer % 2 == 1 { PI / 6.0 } else { 0.0 };

        // Small n-dependent jitter for variety
        let jitter = ((n * 7) % 13) as f64 * 0.02 - 0.13;

        base_angle + offset + jitter
    }

    /// Try a placement in a given direction, return candidate and score if valid
    #[inline]
    fn try_placement(
        &self,
        existing: &[PlacedTree],
        vx: f64,
        vy: f64,
        tree_angle: f64,
        n: usize,
    ) -> Option<(PlacedTree, f64)> {
        // Binary search for closest valid position
        let mut low = 0.0;
        let mut high = 12.0;

        while high - low > 0.002 {  // Slightly finer precision
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
            Some((candidate, score))
        } else {
            None
        }
    }

    /// EVOLVED FUNCTION: Score a placement (lower is better)
    /// NOVEL: Includes physics-inspired neighbor proximity bonus
    #[inline]
    fn placement_score(&self, tree: &PlacedTree, existing: &[PlacedTree], n: usize) -> f64 {
        let (min_x, min_y, max_x, max_y) = tree.bounds();

        // Compute combined bounds
        let mut pack_min_x = min_x;
        let mut pack_min_y = min_y;
        let mut pack_max_x = max_x;
        let mut pack_max_y = max_y;

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

        // Primary: minimize side length
        let side_score = side;

        // Secondary: prefer balanced aspect ratio
        let balance_penalty = (width - height).abs() * 0.12;

        // Tertiary: preference for compact center
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.008 / (n as f64).sqrt();

        // NOVEL: Neighbor proximity bonus - prefer positions close to existing trees
        // (encourages tighter packing without overlap)
        let mut proximity_bonus = 0.0;
        let tree_cx = tree.x;
        let tree_cy = tree.y;

        for t in existing {
            let dist = ((tree_cx - t.x).powi(2) + (tree_cy - t.y).powi(2)).sqrt();
            // Closer neighbors give bonus (but not too close as that would overlap)
            if dist < 3.0 && dist > 0.5 {
                proximity_bonus += 0.02 * (3.0 - dist) / (n as f64).sqrt();
            }
        }

        side_score + balance_penalty + center_penalty - proximity_bonus
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// NOVEL: More angles for better coverage
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // 12 angles for finer rotation granularity
        let base: Vec<f64> = match n % 6 {
            0 => vec![0.0, 90.0, 180.0, 270.0, 30.0, 60.0, 120.0, 150.0, 210.0, 240.0, 300.0, 330.0],
            1 => vec![90.0, 270.0, 0.0, 180.0, 45.0, 135.0, 225.0, 315.0, 15.0, 75.0, 165.0, 255.0],
            2 => vec![180.0, 0.0, 270.0, 90.0, 60.0, 120.0, 240.0, 300.0, 30.0, 150.0, 210.0, 330.0],
            3 => vec![270.0, 90.0, 180.0, 0.0, 45.0, 135.0, 225.0, 315.0, 15.0, 105.0, 195.0, 285.0],
            4 => vec![45.0, 135.0, 225.0, 315.0, 0.0, 90.0, 180.0, 270.0, 22.5, 67.5, 112.5, 157.5],
            _ => vec![315.0, 45.0, 135.0, 225.0, 0.0, 90.0, 180.0, 270.0, 30.0, 60.0, 120.0, 150.0],
        };
        base
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        // NOVEL: Three-tier direction selection
        let choice = rng.gen::<f64>();

        if choice < 0.5 {
            // Structured: evenly spaced with jitter
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.1..0.1)
        } else if choice < 0.8 {
            // Hex-aligned directions (60-degree intervals)
            let slot = rng.gen_range(0..6);
            let base = (slot as f64) * PI / 3.0;
            base + rng.gen_range(-0.15..0.15)
        } else {
            // Corner-weighted (45, 135, 225, 315 degrees)
            let corner_angles = [PI/4.0, 3.0*PI/4.0, 5.0*PI/4.0, 7.0*PI/4.0];
            let base = corner_angles[rng.gen_range(0..4)];
            base + rng.gen_range(-0.2..0.2)
        }
    }

    /// NOVEL: Multi-pass local search
    /// SA pass 1 -> compact pass -> SA pass 2 with lower temp
    fn multi_pass_local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        // Pass 1: Main SA
        self.local_search_with_restart(trees, n, rng, self.config.sa_initial_temp);

        // NOVEL: Compact pass - try to push all trees toward center
        self.compact_pass(trees, n, rng);

        // Pass 2: Fine-tuning SA with lower temperature
        if n >= 5 {
            self.local_search_with_restart(trees, n, rng, self.config.sa_initial_temp * 0.3);
        }
    }

    /// NOVEL: Compact pass - push trees toward center of mass
    fn compact_pass(&self, trees: &mut Vec<PlacedTree>, _n: usize, _rng: &mut impl Rng) {
        if trees.is_empty() {
            return;
        }

        // Compute center of mass
        let mut cx = 0.0;
        let mut cy = 0.0;
        for t in trees.iter() {
            cx += t.x;
            cy += t.y;
        }
        cx /= trees.len() as f64;
        cy /= trees.len() as f64;

        // Try to move each tree toward center
        for idx in 0..trees.len() {
            let old = trees[idx].clone();
            let dx = cx - old.x;
            let dy = cy - old.y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist > 0.1 {
                // Try multiple step sizes
                for step in &[0.05, 0.02, 0.01] {
                    let scale = step / dist;
                    let new_x = old.x + dx * scale;
                    let new_y = old.y + dy * scale;

                    trees[idx] = PlacedTree::new(new_x, new_y, old.angle_deg);

                    if has_overlap(trees, idx) {
                        trees[idx] = old.clone();
                    } else {
                        break; // Successfully moved
                    }
                }
            }
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// NOVEL: Includes temperature restart on stagnation and periodic swaps
    fn local_search_with_restart(&self, trees: &mut Vec<PlacedTree>, n: usize, rng: &mut impl Rng, initial_temp: f64) {
        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut temp = initial_temp;
        let mut stagnation_count = 0;

        let iterations = self.config.sa_iterations + n * 25;

        for iter in 0..iterations {
            // NOVEL: Periodic greedy swap
            if iter > 0 && iter % self.config.swap_frequency == 0 && trees.len() >= 2 {
                self.try_greedy_swap(trees, rng);
                current_side = compute_side_length(trees);
            }

            let idx = rng.gen_range(0..trees.len());
            let old_tree = trees[idx].clone();

            // EVOLVED: Move operator selection with physics
            let success = self.sa_move_physics(trees, idx, temp, iter, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                // Metropolis criterion
                if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                    current_side = new_side;

                    if current_side < best_side {
                        best_side = current_side;
                        stagnation_count = 0;
                    }
                } else {
                    trees[idx] = old_tree;
                }
            } else {
                trees[idx] = old_tree;
            }

            stagnation_count += 1;

            // NOVEL: Temperature restart on stagnation
            if stagnation_count >= self.config.stagnation_threshold {
                temp = (temp * self.config.restart_temp_multiplier).min(initial_temp);
                stagnation_count = 0;
            }

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
        }
    }

    /// NOVEL: Try swapping positions of two trees
    fn try_greedy_swap(&self, trees: &mut [PlacedTree], rng: &mut impl Rng) {
        if trees.len() < 2 {
            return;
        }

        let current_side = compute_side_length(trees);

        // Try a few random swaps
        for _ in 0..3 {
            let i = rng.gen_range(0..trees.len());
            let j = rng.gen_range(0..trees.len());
            if i == j {
                continue;
            }

            // Swap positions (keep angles)
            let old_i = trees[i].clone();
            let old_j = trees[j].clone();

            trees[i] = PlacedTree::new(old_j.x, old_j.y, old_i.angle_deg);
            trees[j] = PlacedTree::new(old_i.x, old_i.y, old_j.angle_deg);

            // Check validity and improvement
            if has_overlap(trees, i) || has_overlap(trees, j) {
                // Revert
                trees[i] = old_i;
                trees[j] = old_j;
            } else {
                let new_side = compute_side_length(trees);
                if new_side >= current_side {
                    // Revert if not better
                    trees[i] = old_i;
                    trees[j] = old_j;
                }
                // Keep if better
            }
        }
    }

    /// EVOLVED FUNCTION: SA move operator with physics-inspired moves
    /// Returns true if move is valid (no overlap)
    #[inline]
    fn sa_move_physics(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        _iter: usize,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let move_type = rng.gen_range(0..8);  // Extended move set

        match move_type {
            0 => {
                // Small translation (temperature-scaled)
                let scale = self.config.translation_scale * (0.3 + temp * 2.5);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                // 90-degree rotation
                let new_angle = (old_angle + 90.0).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            2 => {
                // Fine rotation
                let delta = if rng.gen() { self.config.rotation_granularity }
                            else { -self.config.rotation_granularity };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            3 => {
                // Move toward center (attraction)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.05 {
                    let scale = self.config.attraction_strength * (0.5 + temp);
                    let dx = -old_x / mag * scale;
                    let dy = -old_y / mag * scale;
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                } else {
                    return false;
                }
            }
            4 => {
                // Translate + rotate combo
                let scale = self.config.translation_scale * 0.6;
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                let dangle = rng.gen_range(-45.0..45.0);
                let new_angle = (old_angle + dangle).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            5 => {
                // Polar move (radial in/out)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.1 {
                    let delta_r = rng.gen_range(-0.06..0.06) * (1.0 + temp);
                    let new_mag = (mag + delta_r).max(0.0);
                    let scale = new_mag / mag;
                    trees[idx] = PlacedTree::new(old_x * scale, old_y * scale, old_angle);
                } else {
                    return false;
                }
            }
            6 => {
                // NOVEL: Physics repulsion from nearest neighbor
                if let Some((dx, dy)) = self.compute_repulsion(trees, idx) {
                    let scale = self.config.repulsion_strength * (0.3 + temp);
                    trees[idx] = PlacedTree::new(old_x + dx * scale, old_y + dy * scale, old_angle);
                } else {
                    return false;
                }
            }
            _ => {
                // NOVEL: Tangential move (orbit around center)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.1 {
                    let angle = old_y.atan2(old_x);
                    let delta_angle = rng.gen_range(-0.1..0.1) * (1.0 + temp);
                    let new_angle = angle + delta_angle;
                    trees[idx] = PlacedTree::new(
                        mag * new_angle.cos(),
                        mag * new_angle.sin(),
                        old_angle
                    );
                } else {
                    return false;
                }
            }
        }

        !has_overlap(trees, idx)
    }

    /// NOVEL: Compute repulsion vector from nearest neighbor
    fn compute_repulsion(&self, trees: &[PlacedTree], idx: usize) -> Option<(f64, f64)> {
        let tree = &trees[idx];
        let mut nearest_dist = f64::INFINITY;
        let mut repulse_x = 0.0;
        let mut repulse_y = 0.0;

        for (i, other) in trees.iter().enumerate() {
            if i == idx {
                continue;
            }
            let dx = tree.x - other.x;
            let dy = tree.y - other.y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < nearest_dist && dist > 0.01 {
                nearest_dist = dist;
                // Repulsion direction (away from neighbor)
                repulse_x = dx / dist;
                repulse_y = dy / dist;
            }
        }

        if nearest_dist < f64::INFINITY {
            Some((repulse_x, repulse_y))
        } else {
            None
        }
    }
}

// Helper functions
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
