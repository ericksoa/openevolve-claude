//! Evolved Packing Algorithm - Generation 3 Geometric Extreme
//!
//! This module combines Gen2 EXTREME computational power with Gen2 Hybrid geometric insights.
//! Hypothesis: Geometric awareness + extreme computation might combine synergistically.
//!
//! From Gen2 EXTREME (computational power):
//! - 20000+ iterations, 200 search attempts, 48 direction samples
//! - Binary search precision: 0.0005
//! - Cooling rate: 0.9999 (very slow)
//! - Multi-pass SA (2 passes)
//! - Boundary tree preference for moves
//!
//! From Gen2 Hybrid (geometric insights):
//! - Interlocking bonus for trees 180 degrees apart (tip-to-trunk)
//! - Nesting bonus for tip-to-trunk alignment
//! - 60-degree rotation moves (hexagonal patterns)
//! - 180-degree flip move operator
//! - Hexagonal direction preference (60-degree increments)
//! - Interlock slide move (perpendicular to tip direction)
//!
//! NEW in Gen3:
//! - Combined extreme parameters with geometric scoring
//! - Enhanced geometric bonuses scaled for extreme computation
//! - 12 move types including all geometric moves
//! - Hexagonal + extreme direction sampling

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Evolved packing configuration
/// Combines Gen2 EXTREME parameters with Gen2 Hybrid geometric features
pub struct EvolvedConfig {
    // Search parameters - from Gen2 EXTREME
    pub search_attempts: usize,
    pub direction_samples: usize,

    // Simulated annealing - from Gen2 EXTREME
    pub sa_iterations: usize,
    pub sa_initial_temp: f64,
    pub sa_cooling_rate: f64,
    pub sa_min_temp: f64,

    // Move parameters - hybrid tuned
    pub translation_scale: f64,
    pub rotation_granularity: f64,
    pub center_pull_strength: f64,

    // EXTREME: Multi-pass settings
    pub sa_passes: usize,

    // Geometric parameters - from Gen2 Hybrid, enhanced
    pub interlock_bonus: f64,        // Bonus for tip-to-trunk alignment (180 deg apart)
    pub trunk_gap_threshold: f64,    // Distance threshold for trunk nesting
    pub hexagonal_weight: f64,       // Weight for hexagonal direction preference
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen3 Geometric Extreme: extreme computation + geometric awareness
        Self {
            // From Gen2 EXTREME - maximum computational effort
            search_attempts: 200,           // 200 attempts (3x more than Gen2 Hybrid's 75)
            direction_samples: 48,          // 48 samples (2.4x more than Gen2 Hybrid's 20)
            sa_iterations: 20000,           // 20000 base iterations (3x more than Gen2 Hybrid's 6500)
            sa_initial_temp: 0.6,           // Higher for more exploration
            sa_cooling_rate: 0.9999,        // Very slow cooling
            sa_min_temp: 0.00001,           // 100x lower minimum than Gen2 Hybrid
            translation_scale: 0.08,        // From Gen2 EXTREME
            rotation_granularity: 22.5,     // Finer rotations (16 angles)
            center_pull_strength: 0.05,     // From Gen2 EXTREME
            sa_passes: 2,                   // Double SA pass

            // From Gen2 Hybrid - geometric bonuses (boosted for extreme search)
            interlock_bonus: 0.12,          // Boosted from 0.10 (more impact with more search)
            trunk_gap_threshold: 0.30,      // Boosted from 0.28
            hexagonal_weight: 0.45,         // Balanced hexagonal preference
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

            // Place new tree using evolved heuristics
            let new_tree = self.find_placement(&trees, n, max_n, &mut rng);
            trees.push(new_tree);

            // EXTREME: Run multiple SA passes for deeper optimization
            for pass in 0..self.config.sa_passes {
                self.local_search(&mut trees, n, pass, &mut rng);
            }

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
    /// GEOMETRIC-EXTREME: Enhanced search with geometric interlocking awareness
    fn find_placement(
        &self,
        existing: &[PlacedTree],
        n: usize,
        _max_n: usize,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            // First tree: place at origin with upward orientation
            return PlacedTree::new(0.0, 0.0, 90.0);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 90.0);
        let mut best_score = f64::INFINITY;

        let angles = self.select_angles(n);

        for _ in 0..self.config.search_attempts {
            let dir = self.select_direction(n, rng);
            let vx = dir.cos();
            let vy = dir.sin();

            for &tree_angle in &angles {
                // Binary search for closest valid position
                // EXTREME: 6x finer precision 0.0005 (was 0.0025 in Gen2 Hybrid)
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > 0.0005 {
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

    /// EVOLVED FUNCTION: Score a placement (lower is better)
    /// GEOMETRIC-EXTREME: Combines extreme density heuristics with geometric bonuses
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

        // Primary: minimize side length (most important)
        let side_score = side;

        // Secondary: prefer balanced aspect ratio
        // HYBRID: Adaptive penalty based on n (from Gen2 EXTREME)
        let balance_weight = 0.11 + 0.04 * (1.0 - (n as f64 / 200.0).min(1.0));
        let balance_penalty = (width - height).abs() * balance_weight;

        // Tertiary: slight preference for compact center
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.008 / (n as f64).sqrt();

        // EXTREME: Density heuristic - prefer filling gaps
        let area = width * height;
        let density_bonus = if area > 0.0 {
            -0.006 * (n as f64 / area).min(2.0)  // Reward higher density (boosted)
        } else {
            0.0
        };

        // GEOMETRIC: Interlocking bonus - reward complementary angles
        let interlock_bonus = self.compute_interlock_bonus(tree, existing);

        // GEOMETRIC: Trunk nesting bonus - reward when tip is near trunk gap
        let nesting_bonus = self.compute_nesting_bonus(tree, existing);

        side_score + balance_penalty + center_penalty + density_bonus - interlock_bonus - nesting_bonus
    }

    /// Compute bonus for tip-to-trunk interlocking potential
    /// Trees at 180-degree angle difference can potentially nest together
    /// GEOMETRIC-EXTREME: Enhanced with hexagonal alignment bonus
    #[inline]
    fn compute_interlock_bonus(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        let tree_angle_normalized = tree.angle_deg.rem_euclid(360.0);
        let mut bonus = 0.0;

        for other in existing {
            let other_angle_normalized = other.angle_deg.rem_euclid(360.0);
            let angle_diff = (tree_angle_normalized - other_angle_normalized).abs();

            // Check for complementary angles (180 degrees apart = tip-to-trunk potential)
            // Slightly tighter tolerance (10 degrees) for precision
            let is_complementary = (angle_diff - 180.0).abs() < 10.0 ||
                                   angle_diff < 10.0 ||
                                   (360.0 - angle_diff) < 10.0;

            if is_complementary {
                // Distance-based bonus - closer trees benefit more from interlocking
                let dx = tree.x - other.x;
                let dy = tree.y - other.y;
                let dist = (dx * dx + dy * dy).sqrt();

                // Extended range (2.0 tree heights) for extreme search
                if dist < 2.0 {
                    bonus += self.config.interlock_bonus * (2.0 - dist) / 2.0;
                }
            }

            // GEOMETRIC: Also reward 60-degree alignment for hexagonal patterns
            let hex_aligned = (angle_diff % 60.0) < 8.0 || (60.0 - (angle_diff % 60.0)) < 8.0;
            if hex_aligned && !is_complementary {
                let dx = tree.x - other.x;
                let dy = tree.y - other.y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < 1.5 {
                    bonus += self.config.interlock_bonus * 0.35 * (1.5 - dist) / 1.5;
                }
            }
        }

        bonus
    }

    /// Compute bonus for nesting tip into trunk gap region
    /// The trunk gap (width 0.7 - 0.15 = 0.55 on each side) can fit a tip
    /// GEOMETRIC-EXTREME: Enhanced thresholds for extreme search
    #[inline]
    fn compute_nesting_bonus(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        // Get the tip position of the new tree
        let tip = tree.vertices()[0]; // First vertex is always the tip

        let mut bonus = 0.0;

        for other in existing {
            // Get the trunk region bounds of the other tree
            // Trunk vertices are at indices 6, 7, 8, 9 (right trunk, left trunk)
            let trunk_vertices = [other.vertices()[6], other.vertices()[7],
                                  other.vertices()[8], other.vertices()[9]];

            // Compute trunk bounding box
            let trunk_min_x = trunk_vertices.iter().map(|v| v.0).fold(f64::INFINITY, f64::min);
            let trunk_max_x = trunk_vertices.iter().map(|v| v.0).fold(f64::NEG_INFINITY, f64::max);
            let trunk_min_y = trunk_vertices.iter().map(|v| v.1).fold(f64::INFINITY, f64::min);
            let trunk_max_y = trunk_vertices.iter().map(|v| v.1).fold(f64::NEG_INFINITY, f64::max);

            // Check if tip is near the trunk gap region (the space beside the trunk)
            let base_vertices = [other.vertices()[5], other.vertices()[10]]; // Base tier edges
            let base_y = (base_vertices[0].1 + base_vertices[1].1) / 2.0;

            // The gap region is beside the trunk, below the base tier
            let gap_left_x = trunk_min_x - 0.40;  // Extended from 0.35
            let gap_right_x = trunk_max_x + 0.40;

            // Check if tip is in the gap region
            let in_gap_region = (tip.0 > gap_left_x && tip.0 < trunk_min_x - 0.03 ||
                                tip.0 < gap_right_x && tip.0 > trunk_max_x + 0.03) &&
                               tip.1 > trunk_min_y - 0.15 && tip.1 < trunk_max_y + 0.40;

            if in_gap_region {
                bonus += self.config.interlock_bonus * 0.7; // Boosted from 0.6
            }

            // Also reward if the tip is close to the base tier level
            let dist_to_base = (tip.1 - base_y).abs();
            if dist_to_base < self.config.trunk_gap_threshold {
                bonus += self.config.interlock_bonus * 0.4 * (1.0 - dist_to_base / self.config.trunk_gap_threshold);
            }
        }

        bonus
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// GEOMETRIC-EXTREME: Extended angle set with hexagonal + octagonal + complementary pairs
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // Extended 14-angle set: complementary pairs + hexagonal + octagonal
        // Primary: complementary pairs (0/180, 90/270) for interlocking
        // Secondary: hexagonal (60-degree intervals) for dense packing
        // Tertiary: fine angles (30-degree) for flexibility
        match n % 8 {
            0 => vec![0.0, 180.0, 60.0, 240.0, 90.0, 270.0, 120.0, 300.0, 45.0, 225.0, 30.0, 210.0, 150.0, 330.0],
            1 => vec![90.0, 270.0, 0.0, 180.0, 60.0, 240.0, 120.0, 300.0, 135.0, 315.0, 30.0, 150.0, 210.0, 330.0],
            2 => vec![180.0, 0.0, 240.0, 60.0, 270.0, 90.0, 300.0, 120.0, 225.0, 45.0, 150.0, 330.0, 30.0, 210.0],
            3 => vec![270.0, 90.0, 180.0, 0.0, 300.0, 120.0, 240.0, 60.0, 315.0, 135.0, 210.0, 30.0, 150.0, 330.0],
            4 => vec![60.0, 240.0, 0.0, 180.0, 120.0, 300.0, 90.0, 270.0, 30.0, 210.0, 150.0, 330.0, 45.0, 225.0],
            5 => vec![120.0, 300.0, 60.0, 240.0, 180.0, 0.0, 150.0, 330.0, 90.0, 270.0, 30.0, 210.0, 45.0, 225.0],
            6 => vec![240.0, 60.0, 300.0, 120.0, 0.0, 180.0, 270.0, 90.0, 210.0, 30.0, 330.0, 150.0, 225.0, 45.0],
            _ => vec![300.0, 120.0, 240.0, 60.0, 90.0, 270.0, 0.0, 180.0, 330.0, 150.0, 210.0, 30.0, 315.0, 135.0],
        }
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    /// GEOMETRIC-EXTREME: Combines hexagonal preference with extreme sampling
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        // Select direction strategy - four-way mix
        let strategy = rng.gen::<f64>();

        if strategy < self.config.hexagonal_weight {
            // GEOMETRIC: Hexagonal directions: 0, 60, 120, 180, 240, 300 degrees
            let hex_base = (rng.gen_range(0..6) as f64) * 60.0;
            let jitter = rng.gen_range(-8.0..8.0); // Tight jitter for precision
            (hex_base + jitter) * PI / 180.0
        } else if strategy < 0.70 {
            // EXTREME: Structured evenly spaced with small jitter
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.08..0.08)  // Tighter jitter
        } else if strategy < 0.85 {
            // EXTREME: Golden angle spiral for good coverage
            let golden_angle = PI * (3.0 - (5.0_f64).sqrt());  // ~137.5 degrees
            let base = (n as f64 * golden_angle) % (2.0 * PI);
            let offset = rng.gen_range(0..8) as f64 * PI / 4.0;
            (base + offset + rng.gen_range(-0.1..0.1)) % (2.0 * PI)
        } else {
            // HYBRID: Combined corner + hexagonal weighting
            loop {
                let angle = rng.gen_range(0.0..2.0 * PI);
                let corner_weight = ((4.0 * angle).sin().abs() + (4.0 * angle).cos().abs()) / 2.0;
                let hex_weight = (3.0 * angle).cos().abs(); // Peaks at 0, 60, 120, ...
                let combined_weight = 0.35 * corner_weight + 0.65 * hex_weight;
                let threshold = 0.20 + 0.12 * (1.0 - (n as f64 / 200.0).min(1.0));
                if rng.gen::<f64>() < combined_weight.max(threshold) {
                    return angle;
                }
            }
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// GEOMETRIC-EXTREME: Much longer search with multiple passes
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, pass: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        // EXTREME: Adjust temperature based on pass number
        let temp_multiplier = if pass == 0 { 1.0 } else { 0.3 };  // Lower temp on second pass
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        // EXTREME: Dramatically more iterations: 20000 + n*100
        let base_iterations = if pass == 0 {
            self.config.sa_iterations + n * 100
        } else {
            // Second pass: focused refinement with fewer iterations
            self.config.sa_iterations / 2 + n * 50
        };

        for iter in 0..base_iterations {
            // EXTREME: Prefer moving trees that contribute to bounding box
            let idx = self.select_tree_to_move(trees, rng);
            let old_tree = trees[idx].clone();

            // EVOLVED: Move operator selection
            let success = self.sa_move(trees, idx, temp, iter, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                // Metropolis criterion
                if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                    current_side = new_side;
                    // Track best solution found
                    if current_side < best_side {
                        best_side = current_side;
                        best_config = trees.clone();
                    }
                } else {
                    trees[idx] = old_tree;
                }
            } else {
                trees[idx] = old_tree;
            }

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
        }

        // EXTREME: Restore best configuration found during search
        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    /// EXTREME: Select tree to move with preference for boundary trees
    #[inline]
    fn select_tree_to_move(&self, trees: &[PlacedTree], rng: &mut impl Rng) -> usize {
        // 70% chance to pick randomly, 30% chance to pick boundary tree
        if trees.len() <= 2 || rng.gen::<f64>() < 0.7 {
            return rng.gen_range(0..trees.len());
        }

        // Find bounding box
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for tree in trees.iter() {
            let (bx1, by1, bx2, by2) = tree.bounds();
            min_x = min_x.min(bx1);
            min_y = min_y.min(by1);
            max_x = max_x.max(bx2);
            max_y = max_y.max(by2);
        }

        // Find trees touching the boundary
        let mut boundary_indices: Vec<usize> = Vec::new();
        let eps = 0.01;

        for (i, tree) in trees.iter().enumerate() {
            let (bx1, by1, bx2, by2) = tree.bounds();
            if (bx1 - min_x).abs() < eps || (bx2 - max_x).abs() < eps ||
               (by1 - min_y).abs() < eps || (by2 - max_y).abs() < eps {
                boundary_indices.push(i);
            }
        }

        if boundary_indices.is_empty() {
            rng.gen_range(0..trees.len())
        } else {
            boundary_indices[rng.gen_range(0..boundary_indices.len())]
        }
    }

    /// EVOLVED FUNCTION: SA move operator
    /// GEOMETRIC-EXTREME: 12 move types including all geometric moves
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

        // GEOMETRIC-EXTREME: 12 move types
        let move_type = rng.gen_range(0..12);

        match move_type {
            0 => {
                // Small translation (temperature-scaled)
                let scale = self.config.translation_scale * (0.2 + temp * 2.5);
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
                // GEOMETRIC: 60-degree rotation for hexagonal patterns
                let delta = if rng.gen() { 60.0 } else { -60.0 };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            3 => {
                // Fine rotation (22.5 degrees)
                let delta = if rng.gen() { self.config.rotation_granularity }
                            else { -self.config.rotation_granularity };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            4 => {
                // Move toward center
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.05 {
                    let scale = self.config.center_pull_strength * (0.4 + temp * 1.5);
                    let dx = -old_x / mag * scale;
                    let dy = -old_y / mag * scale;
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                } else {
                    return false;
                }
            }
            5 => {
                // Translate + rotate combo
                let scale = self.config.translation_scale * 0.4;
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                let dangle = rng.gen_range(-45.0..45.0);
                let new_angle = (old_angle + dangle).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            6 => {
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
            7 => {
                // EXTREME: Angular orbit (move around center)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.1 {
                    let current_angle = old_y.atan2(old_x);
                    let delta_angle = rng.gen_range(-0.15..0.15) * (1.0 + temp);
                    let new_ang = current_angle + delta_angle;
                    trees[idx] = PlacedTree::new(mag * new_ang.cos(), mag * new_ang.sin(), old_angle);
                } else {
                    return false;
                }
            }
            8 => {
                // GEOMETRIC: 180-degree flip for tip-to-trunk interlocking
                let new_angle = (old_angle + 180.0).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            9 => {
                // GEOMETRIC: Interlock slide - move perpendicular to tree's tip
                let tip_dir = old_angle * PI / 180.0;
                let slide_dir = tip_dir + PI / 2.0;
                let slide_dist = rng.gen_range(-0.10..0.10) * (0.5 + temp);
                let dx = slide_dir.cos() * slide_dist;
                let dy = slide_dir.sin() * slide_dist;
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            10 => {
                // EXTREME: Very small nudge for fine-tuning
                let scale = 0.015 * (0.5 + temp);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            _ => {
                // EXTREME: Directional slide toward hexagonal directions
                let hex_dir_idx = rng.gen_range(0..6);
                let hex_angle = (hex_dir_idx as f64) * PI / 3.0; // 0, 60, 120, 180, 240, 300 degrees
                let scale = 0.04 * (0.5 + temp * 1.5);
                trees[idx] = PlacedTree::new(
                    old_x + hex_angle.cos() * scale,
                    old_y + hex_angle.sin() * scale,
                    old_angle
                );
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
        println!("Gen3 Geometric-Extreme score for n=1..50: {:.4}", score);
    }
}
