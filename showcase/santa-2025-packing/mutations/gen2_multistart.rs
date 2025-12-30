//! Evolved Packing Algorithm - Generation 2 Multi-Restart Mutation
//!
//! MUTATION STRATEGY: MULTIPLE INDEPENDENT ATTEMPTS
//! Instead of one placement+SA sequence per n, run multiple independent attempts
//! with different random seeds and keep the best result.
//!
//! Key innovations:
//! 1. For each n, run NUM_RESTARTS independent attempts with different seeds
//! 2. Each attempt places the new tree using different random directions/angles
//! 3. Each attempt runs independent SA optimization
//! 4. Keep the result with the smallest side length
//! 5. Vary parameters between attempts for better exploration:
//!    - Different initial temperatures
//!    - Different angle orderings
//!    - Different direction biases
//!
//! This trades computation for better exploration of the solution space.

use crate::{Packing, PlacedTree};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::f64::consts::PI;

/// Multi-restart configuration
/// Parameters tuned for exploration vs. exploitation tradeoff
pub struct EvolvedConfig {
    // Multi-restart parameters
    pub num_restarts: usize,          // Number of independent attempts per n
    pub use_previous_best: bool,      // Use previous solution as one starting point

    // Search parameters (per attempt)
    pub search_attempts: usize,
    pub direction_samples: usize,

    // Simulated annealing (base values, varied across restarts)
    pub sa_iterations_base: usize,
    pub sa_initial_temp_base: f64,
    pub sa_cooling_rate: f64,
    pub sa_min_temp: f64,

    // Move parameters
    pub translation_scale: f64,
    pub rotation_granularity: f64,
    pub center_pull_strength: f64,
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        Self {
            // Multi-restart: 8 attempts balances exploration vs. computation
            num_restarts: 8,
            use_previous_best: true,

            // Slightly reduced per-attempt search since we have multiple restarts
            search_attempts: 50,        // Reduced from 75 (compensated by restarts)
            direction_samples: 16,

            // SA parameters (base values, varied per restart)
            sa_iterations_base: 4000,   // Reduced base, varied per restart
            sa_initial_temp_base: 0.40, // Base temperature
            sa_cooling_rate: 0.9993,
            sa_min_temp: 0.001,

            // Move parameters (inherited from Gen1)
            translation_scale: 0.08,
            rotation_granularity: 45.0,
            center_pull_strength: 0.04,
        }
    }
}

/// Restart variant parameters
/// Each restart uses slightly different parameters for diversity
struct RestartVariant {
    seed: u64,
    temp_multiplier: f64,
    iterations_multiplier: f64,
    angle_priority: usize,       // Which angle ordering to prefer
    direction_bias: f64,         // Bias toward corners vs. cardinal
    fresh_start: bool,           // Start fresh or from previous best
}

impl RestartVariant {
    fn generate(restart_idx: usize, n: usize, master_seed: u64) -> Self {
        // Deterministic variant generation based on restart index and n
        let seed = master_seed.wrapping_add(restart_idx as u64 * 12345 + n as u64 * 67890);

        // Vary parameters across restarts
        let variants = [
            // (temp_mult, iter_mult, angle_prio, dir_bias, fresh)
            (1.0, 1.0, 0, 0.7, false),   // Standard, use previous
            (1.3, 1.2, 1, 0.5, false),   // Higher temp, more iters
            (0.8, 0.8, 2, 0.8, true),    // Lower temp, fresh start
            (1.5, 1.5, 3, 0.6, false),   // Aggressive exploration
            (0.7, 1.3, 0, 0.9, true),    // Cool start, many iters
            (1.2, 1.0, 1, 0.4, false),   // Slightly warm
            (1.0, 1.4, 2, 0.7, true),    // Many iterations, fresh
            (1.4, 0.9, 3, 0.5, false),   // Hot start, quick cool
        ];

        let v = &variants[restart_idx % variants.len()];

        RestartVariant {
            seed,
            temp_multiplier: v.0,
            iterations_multiplier: v.1,
            angle_priority: v.2,
            direction_bias: v.3,
            fresh_start: v.4,
        }
    }
}

/// Main evolved packer with multi-restart optimization
pub struct EvolvedPacker {
    pub config: EvolvedConfig,
}

impl Default for EvolvedPacker {
    fn default() -> Self {
        Self { config: EvolvedConfig::default() }
    }
}

impl EvolvedPacker {
    /// Pack all n from 1 to max_n with multi-restart optimization
    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        let mut master_rng = rand::thread_rng();
        let master_seed: u64 = master_rng.gen();

        let mut packings: Vec<Packing> = Vec::with_capacity(max_n);
        let mut prev_best_trees: Vec<PlacedTree> = Vec::new();

        for n in 1..=max_n {
            let mut best_trees: Option<Vec<PlacedTree>> = None;
            let mut best_side = f64::INFINITY;

            // Run multiple independent restarts
            for restart_idx in 0..self.config.num_restarts {
                let variant = RestartVariant::generate(restart_idx, n, master_seed);
                let mut rng = StdRng::seed_from_u64(variant.seed);

                // Decide starting point
                let starting_trees = if n == 1 {
                    Vec::new()
                } else if variant.fresh_start || !self.config.use_previous_best {
                    // Fresh start: rebuild from scratch using previous packings
                    self.rebuild_from_packings(&packings, n - 1, &variant, &mut rng)
                } else {
                    // Use previous best as starting point (possibly perturbed)
                    if restart_idx > 0 && restart_idx % 3 == 0 {
                        // Every 3rd restart: perturb previous solution
                        self.perturb_solution(&prev_best_trees, &mut rng)
                    } else {
                        prev_best_trees.clone()
                    }
                };

                // Place new tree with variant-specific parameters
                let mut trees = starting_trees;
                let new_tree = self.find_placement(&trees, n, max_n, &variant, &mut rng);
                trees.push(new_tree);

                // Run SA with variant-specific parameters
                self.local_search(&mut trees, n, &variant, &mut rng);

                // Evaluate and keep if best
                let side = compute_side_length(&trees);
                if side < best_side {
                    best_side = side;
                    best_trees = Some(trees);
                }
            }

            // Store the best result from all restarts
            let trees = best_trees.unwrap();
            let mut packing = Packing::new();
            for t in &trees {
                packing.trees.push(t.clone());
            }
            packings.push(packing);
            prev_best_trees = trees;
        }

        packings
    }

    /// Rebuild solution from previous packings (for fresh starts)
    fn rebuild_from_packings(
        &self,
        packings: &[Packing],
        up_to_n: usize,
        variant: &RestartVariant,
        rng: &mut impl Rng,
    ) -> Vec<PlacedTree> {
        if up_to_n == 0 || packings.is_empty() {
            return Vec::new();
        }

        // Start from scratch, adding trees one by one with fresh placements
        let mut trees = Vec::new();

        for i in 1..=up_to_n {
            if i == 1 {
                trees.push(PlacedTree::new(0.0, 0.0, 90.0));
            } else {
                let new_tree = self.find_placement(&trees, i, up_to_n, variant, rng);
                trees.push(new_tree);
            }
        }

        // Quick SA pass to refine
        let quick_variant = RestartVariant {
            iterations_multiplier: 0.3, // Quick refinement
            ..*variant
        };
        self.local_search(&mut trees, up_to_n, &quick_variant, rng);

        trees
    }

    /// Perturb an existing solution for diversity
    fn perturb_solution(&self, trees: &[PlacedTree], rng: &mut impl Rng) -> Vec<PlacedTree> {
        let mut perturbed: Vec<PlacedTree> = trees.to_vec();

        // Apply small random perturbations
        for tree in &mut perturbed {
            if rng.gen::<f64>() < 0.3 {
                // 30% chance to perturb each tree
                let dx = rng.gen_range(-0.02..0.02);
                let dy = rng.gen_range(-0.02..0.02);
                let new_tree = PlacedTree::new(tree.x + dx, tree.y + dy, tree.angle_deg);

                // Only apply if still valid
                let mut temp = perturbed.clone();
                let idx = temp.iter().position(|t| t.x == tree.x && t.y == tree.y).unwrap_or(0);
                temp[idx] = new_tree.clone();
                if !has_any_overlap(&temp) {
                    *tree = new_tree;
                }
            }
        }

        perturbed
    }

    /// Find best placement for new tree (variant-aware)
    fn find_placement(
        &self,
        existing: &[PlacedTree],
        n: usize,
        _max_n: usize,
        variant: &RestartVariant,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            return PlacedTree::new(0.0, 0.0, 90.0);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 90.0);
        let mut best_score = f64::INFINITY;

        // Select angles based on variant priority
        let angles = self.select_angles(n, variant.angle_priority);

        for _ in 0..self.config.search_attempts {
            let dir = self.select_direction(n, variant.direction_bias, rng);
            let vx = dir.cos();
            let vy = dir.sin();

            for &tree_angle in &angles {
                // Binary search for closest valid position
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > 0.003 {
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

    /// Score a placement (lower is better)
    #[inline]
    fn placement_score(&self, tree: &PlacedTree, existing: &[PlacedTree], n: usize) -> f64 {
        let (min_x, min_y, max_x, max_y) = tree.bounds();

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

        // Tertiary: slight preference for compact center
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.01 / (n as f64).sqrt();

        side_score + balance_penalty + center_penalty
    }

    /// Select rotation angles based on priority scheme
    #[inline]
    fn select_angles(&self, n: usize, priority: usize) -> Vec<f64> {
        // Different angle orderings for diversity
        let orderings = [
            vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0],
            vec![90.0, 270.0, 0.0, 180.0, 135.0, 315.0, 45.0, 225.0],
            vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0],
            vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0],
        ];

        // Combine n-based and variant-based selection
        let idx = (n + priority) % orderings.len();
        orderings[idx].clone()
    }

    /// Select direction angle for placement (with configurable bias)
    #[inline]
    fn select_direction(&self, n: usize, corner_bias: f64, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        if rng.gen::<f64>() < corner_bias {
            // Structured: evenly spaced with jitter
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.15..0.15)
        } else {
            // Weighted random: favor corners
            loop {
                let angle = rng.gen_range(0.0..2.0 * PI);
                let corner_weight = (2.0 * angle).sin().abs();
                let threshold = 0.25 + 0.1 * (1.0 - (n as f64 / 200.0).min(1.0));
                if rng.gen::<f64>() < corner_weight.max(threshold) {
                    return angle;
                }
            }
        }
    }

    /// Local search with simulated annealing (variant-aware)
    fn local_search(
        &self,
        trees: &mut Vec<PlacedTree>,
        n: usize,
        variant: &RestartVariant,
        rng: &mut impl Rng,
    ) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);

        // Apply variant-specific temperature and iterations
        let mut temp = self.config.sa_initial_temp_base * variant.temp_multiplier;
        let base_iterations = self.config.sa_iterations_base + n * 20;
        let iterations = ((base_iterations as f64) * variant.iterations_multiplier) as usize;

        for iter in 0..iterations {
            let idx = rng.gen_range(0..trees.len());
            let old_tree = trees[idx].clone();

            let success = self.sa_move(trees, idx, temp, iter, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                    current_side = new_side;
                } else {
                    trees[idx] = old_tree;
                }
            } else {
                trees[idx] = old_tree;
            }

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
        }
    }

    /// SA move operator
    #[inline]
    fn sa_move(
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

        let move_type = rng.gen_range(0..6);

        match move_type {
            0 => {
                // Small translation (temperature-scaled)
                let scale = self.config.translation_scale * (0.3 + temp * 2.0);
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
                // Fine rotation (45 degrees)
                let delta = if rng.gen() { self.config.rotation_granularity }
                            else { -self.config.rotation_granularity };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            3 => {
                // Move toward center
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.05 {
                    let scale = self.config.center_pull_strength * (0.5 + temp);
                    let dx = -old_x / mag * scale;
                    let dy = -old_y / mag * scale;
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                } else {
                    return false;
                }
            }
            4 => {
                // Translate + rotate combo
                let scale = self.config.translation_scale * 0.5;
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                let dangle = rng.gen_range(-30.0..30.0);
                let new_angle = (old_angle + dangle).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            _ => {
                // Polar move (radial in/out)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.1 {
                    let delta_r = rng.gen_range(-0.05..0.05) * (1.0 + temp);
                    let new_mag = (mag + delta_r).max(0.0);
                    let scale = new_mag / mag;
                    trees[idx] = PlacedTree::new(old_x * scale, old_y * scale, old_angle);
                } else {
                    return false;
                }
            }
        }

        !has_overlap(trees, idx)
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

fn has_any_overlap(trees: &[PlacedTree]) -> bool {
    for i in 0..trees.len() {
        for j in (i + 1)..trees.len() {
            if trees[i].overlaps(&trees[j]) {
                return true;
            }
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
        println!("Gen2 Multi-restart score for n=1..50: {:.4}", score);
    }

    #[test]
    fn test_restart_variants() {
        // Verify restart variants are diverse
        for n in [5, 10, 20] {
            let mut seeds = std::collections::HashSet::new();
            for restart_idx in 0..8 {
                let variant = RestartVariant::generate(restart_idx, n, 42);
                seeds.insert(variant.seed);
            }
            assert_eq!(seeds.len(), 8, "All restart variants should have unique seeds");
        }
    }
}
