//! Evolved Packing Algorithm - Generation 2 Hybrid (Tweak + Geometric)
//!
//! This module combines the best elements from Gen1 evolution:
//! - Gen1 Tweak parameters (+3% improvement): search_attempts=75, sa_iterations=6500,
//!   sa_initial_temp=0.45, balance_penalty=0.12
//! - Gen1 Geometric strategies (+2% improvement): hexagonal packing patterns,
//!   tip-to-trunk interlocking, 60-degree rotation moves, nesting bonuses
//!
//! HYBRID STRATEGY:
//! 1. Use proven parameter values from gen1_tweak
//! 2. Incorporate geometric insights for tree shape exploitation
//! 3. Combine complementary angle detection with better search parameters
//!
//! Evolution targets:
//! - placement_score(): Geometric bonuses with tuned penalties
//! - select_angles(): Hybrid octagonal + hexagonal patterns
//! - select_direction(): Enhanced hexagonal direction preference
//! - sa_move(): Extended move operators including geometric moves

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Evolved packing configuration
/// Hybrid of gen1_tweak parameters + gen1_geometric extensions
pub struct EvolvedConfig {
    // Search parameters - from gen1_tweak (proven +3%)
    pub search_attempts: usize,
    pub direction_samples: usize,

    // Simulated annealing - from gen1_tweak (proven +3%)
    pub sa_iterations: usize,
    pub sa_initial_temp: f64,
    pub sa_cooling_rate: f64,
    pub sa_min_temp: f64,

    // Move parameters - hybrid tuned
    pub translation_scale: f64,
    pub rotation_granularity: f64,
    pub center_pull_strength: f64,

    // Geometric parameters - from gen1_geometric (proven +2%)
    pub interlock_bonus: f64,        // Bonus for tip-to-trunk alignment
    pub trunk_gap_threshold: f64,    // Distance to consider for trunk nesting
    pub hexagonal_weight: f64,       // Weight for hexagonal direction preference
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen2 Hybrid: best of both worlds
        Self {
            // From gen1_tweak - proven parameters
            search_attempts: 75,         // TWEAK: 60 -> 75 (+25%)
            direction_samples: 20,       // HYBRID: compromise between 16 and 24
            sa_iterations: 6500,         // TWEAK: 5000 -> 6500 (+30%)
            sa_initial_temp: 0.45,       // TWEAK: 0.35 -> 0.45 (+29%)
            sa_cooling_rate: 0.9993,     // From gen1_tweak
            sa_min_temp: 0.0008,         // From gen1_geometric (slightly lower)

            // Hybrid move parameters
            translation_scale: 0.07,     // HYBRID: between 0.08 and 0.06
            rotation_granularity: 30.0,  // GEOMETRIC: 30-degree for hexagonal
            center_pull_strength: 0.038, // HYBRID: between 0.04 and 0.035

            // From gen1_geometric - geometric bonuses
            interlock_bonus: 0.10,       // BOOSTED: increased from 0.08
            trunk_gap_threshold: 0.28,   // BOOSTED: increased from 0.25
            hexagonal_weight: 0.55,      // HYBRID: slightly reduced from 0.6
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

            // Run evolved local search
            self.local_search(&mut trees, n, &mut rng);

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
    /// HYBRID: Enhanced search with geometric interlocking awareness
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
                // HYBRID: Finer precision (0.0025) for better interlocking detection
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > 0.0025 {
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
    /// HYBRID: Combines gen1_tweak penalties with gen1_geometric bonuses
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
        // FROM GEN1_TWEAK: 0.12 (proven -20% reduction from baseline)
        let balance_penalty = (width - height).abs() * 0.12;

        // Tertiary: slight preference for compact center
        // HYBRID: Slightly reduced to allow more geometric flexibility
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.009 / (n as f64).sqrt();

        // GEOMETRIC: Interlocking bonus - reward complementary angles
        let interlock_bonus = self.compute_interlock_bonus(tree, existing);

        // GEOMETRIC: Trunk nesting bonus - reward when tip is near trunk gap
        let nesting_bonus = self.compute_nesting_bonus(tree, existing);

        side_score + balance_penalty + center_penalty - interlock_bonus - nesting_bonus
    }

    /// Compute bonus for tip-to-trunk interlocking potential
    /// Trees at 180-degree angle difference can potentially nest together
    /// HYBRID: Enhanced distance calculation with boosted bonus
    #[inline]
    fn compute_interlock_bonus(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        let tree_angle_normalized = tree.angle_deg.rem_euclid(360.0);
        let mut bonus = 0.0;

        for other in existing {
            let other_angle_normalized = other.angle_deg.rem_euclid(360.0);
            let angle_diff = (tree_angle_normalized - other_angle_normalized).abs();

            // Check for complementary angles (180 degrees apart = tip-to-trunk potential)
            // HYBRID: Slightly tighter tolerance (12 degrees) for better precision
            let is_complementary = (angle_diff - 180.0).abs() < 12.0 ||
                                   angle_diff < 12.0 ||
                                   (360.0 - angle_diff) < 12.0;

            if is_complementary {
                // Distance-based bonus - closer trees benefit more from interlocking
                let dx = tree.x - other.x;
                let dy = tree.y - other.y;
                let dist = (dx * dx + dy * dy).sqrt();

                // HYBRID: Extended range (1.8 tree heights) with boosted bonus
                if dist < 1.8 {
                    bonus += self.config.interlock_bonus * (1.8 - dist) / 1.8;
                }
            }

            // HYBRID NEW: Also reward 60-degree alignment for hexagonal patterns
            let hex_aligned = (angle_diff % 60.0) < 10.0 || (60.0 - (angle_diff % 60.0)) < 10.0;
            if hex_aligned && !is_complementary {
                let dx = tree.x - other.x;
                let dy = tree.y - other.y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < 1.2 {
                    bonus += self.config.interlock_bonus * 0.3 * (1.2 - dist) / 1.2;
                }
            }
        }

        bonus
    }

    /// Compute bonus for nesting tip into trunk gap region
    /// The trunk gap (width 0.7 - 0.15 = 0.55 on each side) can fit a tip
    /// HYBRID: Enhanced with boosted threshold
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
            let gap_left_x = trunk_min_x - 0.35;  // HYBRID: Extended from 0.3
            let gap_right_x = trunk_max_x + 0.35;

            // Check if tip is in the gap region
            let in_gap_region = (tip.0 > gap_left_x && tip.0 < trunk_min_x - 0.04 ||
                                tip.0 < gap_right_x && tip.0 > trunk_max_x + 0.04) &&
                               tip.1 > trunk_min_y - 0.12 && tip.1 < trunk_max_y + 0.35;

            if in_gap_region {
                bonus += self.config.interlock_bonus * 0.6; // HYBRID: Boosted from 0.5
            }

            // Also reward if the tip is close to the base tier level (y ~ 0 in local coords)
            let dist_to_base = (tip.1 - base_y).abs();
            if dist_to_base < self.config.trunk_gap_threshold {
                bonus += self.config.interlock_bonus * 0.35 * (1.0 - dist_to_base / self.config.trunk_gap_threshold);
            }
        }

        bonus
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// HYBRID: Combines octagonal (45-degree) with hexagonal (60-degree) patterns
    /// Plus complementary pairs for tip-to-trunk interlocking
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // HYBRID: Extended angle set combining both patterns
        // Primary: complementary pairs (0/180, 90/270) for interlocking
        // Secondary: hexagonal (60-degree intervals) for dense packing
        // Tertiary: octagonal (45-degree intervals) for flexibility
        match n % 8 {
            0 => vec![0.0, 180.0, 60.0, 240.0, 90.0, 270.0, 120.0, 300.0, 45.0, 225.0],
            1 => vec![90.0, 270.0, 0.0, 180.0, 60.0, 240.0, 120.0, 300.0, 135.0, 315.0],
            2 => vec![180.0, 0.0, 240.0, 60.0, 270.0, 90.0, 300.0, 120.0, 225.0, 45.0],
            3 => vec![270.0, 90.0, 180.0, 0.0, 300.0, 120.0, 240.0, 60.0, 315.0, 135.0],
            4 => vec![60.0, 240.0, 0.0, 180.0, 120.0, 300.0, 90.0, 270.0, 30.0, 210.0],
            5 => vec![120.0, 300.0, 60.0, 240.0, 180.0, 0.0, 150.0, 330.0, 90.0, 270.0],
            6 => vec![240.0, 60.0, 300.0, 120.0, 0.0, 180.0, 270.0, 90.0, 210.0, 30.0],
            _ => vec![300.0, 120.0, 240.0, 60.0, 90.0, 270.0, 0.0, 180.0, 330.0, 150.0],
        }
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    /// HYBRID: Enhanced hexagonal preference with gen1_tweak corner weighting
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        // Select direction strategy
        let strategy = rng.gen::<f64>();

        if strategy < self.config.hexagonal_weight {
            // GEOMETRIC: Hexagonal directions: 0, 60, 120, 180, 240, 300 degrees
            let hex_base = (rng.gen_range(0..6) as f64) * 60.0;
            let jitter = rng.gen_range(-10.0..10.0); // HYBRID: Slightly more jitter
            (hex_base + jitter) * PI / 180.0
        } else if strategy < 0.80 {
            // TWEAK: Structured evenly spaced with jitter (proven approach)
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.15..0.15)
        } else {
            // HYBRID: Combined corner + hexagonal weighting
            loop {
                let angle = rng.gen_range(0.0..2.0 * PI);
                let corner_weight = (2.0 * angle).sin().abs();
                let hex_weight = (3.0 * angle).cos().abs(); // Peaks at 0, 60, 120, ...
                // HYBRID: Balanced combination
                let combined_weight = 0.4 * corner_weight + 0.6 * hex_weight;
                // TWEAK: Adaptive threshold from gen1_tweak
                let threshold = 0.25 + 0.1 * (1.0 - (n as f64 / 200.0).min(1.0));
                if rng.gen::<f64>() < combined_weight.max(threshold) {
                    return angle;
                }
            }
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// HYBRID: Uses gen1_tweak parameters with gen1_geometric iteration scaling
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut temp = self.config.sa_initial_temp;

        // HYBRID: Balanced iteration scaling (between 20 and 25)
        let iterations = self.config.sa_iterations + n * 22;

        for iter in 0..iterations {
            let idx = rng.gen_range(0..trees.len());
            let old_tree = trees[idx].clone();

            // EVOLVED: Move operator selection
            let success = self.sa_move(trees, idx, temp, iter, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                // Metropolis criterion
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

    /// EVOLVED FUNCTION: SA move operator
    /// HYBRID: All moves from both gen1_tweak and gen1_geometric
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

        // HYBRID: Extended move set (9 moves)
        let move_type = rng.gen_range(0..9);

        match move_type {
            0 => {
                // Small translation (temperature-scaled) - from both
                let scale = self.config.translation_scale * (0.3 + temp * 2.0);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                // 90-degree rotation - from gen1_tweak
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
                // Move toward center - from both
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
                // Translate + rotate combo - from gen1_tweak
                let scale = self.config.translation_scale * 0.5;
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                let dangle = rng.gen_range(-self.config.rotation_granularity..self.config.rotation_granularity);
                let new_angle = (old_angle + dangle).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            5 => {
                // Polar move (radial in/out) - hybrid parameters
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.1 {
                    let delta_r = rng.gen_range(-0.045..0.045) * (1.0 + temp);
                    let new_mag = (mag + delta_r).max(0.0);
                    let scale = new_mag / mag;
                    trees[idx] = PlacedTree::new(old_x * scale, old_y * scale, old_angle);
                } else {
                    return false;
                }
            }
            6 => {
                // GEOMETRIC: 180-degree flip for tip-to-trunk interlocking
                let new_angle = (old_angle + 180.0).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            7 => {
                // GEOMETRIC: Interlock slide - move perpendicular to tree's tip
                let tip_dir = old_angle * PI / 180.0;
                let slide_dir = tip_dir + PI / 2.0;
                let slide_dist = rng.gen_range(-0.08..0.08) * (0.5 + temp);
                let dx = slide_dir.cos() * slide_dist;
                let dy = slide_dir.sin() * slide_dist;
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            _ => {
                // TWEAK: Fine rotation (45 degrees) - from gen1_tweak
                let delta = if rng.gen() { 45.0 } else { -45.0 };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
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
        println!("Gen2 Hybrid (Tweak+Geo) score for n=1..50: {:.4}", score);
    }
}
