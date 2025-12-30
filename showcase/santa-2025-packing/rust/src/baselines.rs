//! Baseline packing algorithms

use crate::{Packing, PackingAlgorithm, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Greedy radial packing - place trees moving from outside toward center
pub struct GreedyRadial {
    pub attempts_per_tree: usize,
}

impl Default for GreedyRadial {
    fn default() -> Self {
        Self { attempts_per_tree: 10 }
    }
}

impl PackingAlgorithm for GreedyRadial {
    fn pack(&self, n: usize) -> Packing {
        let mut rng = rand::thread_rng();
        let mut packing = Packing::new();

        if n == 0 {
            return packing;
        }

        // Place first tree at origin
        let angle = rng.gen_range(0.0..360.0);
        packing.try_add(PlacedTree::new(0.0, 0.0, angle));

        // Place remaining trees
        for _ in 1..n {
            let tree_angle = rng.gen_range(0.0..360.0);

            let mut best_x = 0.0;
            let mut best_y = 0.0;
            let mut best_radius = f64::INFINITY;

            for _ in 0..self.attempts_per_tree {
                // Random direction weighted toward corners
                let dir_angle = generate_weighted_angle(&mut rng);
                let vx = dir_angle.cos();
                let vy = dir_angle.sin();

                // Start far away, move toward center
                let mut radius = 20.0;
                let step_in = 0.5;

                // Move in until collision
                let mut collision_radius = None;
                while radius >= 0.0 {
                    let px = radius * vx;
                    let py = radius * vy;
                    let candidate = PlacedTree::new(px, py, tree_angle);

                    if !packing.can_place(&candidate) {
                        collision_radius = Some(radius);
                        break;
                    }
                    radius -= step_in;
                }

                // Back up until no collision
                let final_radius = if let Some(cr) = collision_radius {
                    let mut r = cr;
                    let step_out = 0.05;
                    loop {
                        r += step_out;
                        let px = r * vx;
                        let py = r * vy;
                        let candidate = PlacedTree::new(px, py, tree_angle);
                        if packing.can_place(&candidate) {
                            break;
                        }
                    }
                    r
                } else {
                    0.0
                };

                if final_radius < best_radius {
                    best_radius = final_radius;
                    best_x = final_radius * vx;
                    best_y = final_radius * vy;
                }
            }

            let tree = PlacedTree::new(best_x, best_y, tree_angle);
            packing.try_add(tree);
        }

        packing
    }

    fn name(&self) -> &'static str {
        "greedy_radial"
    }
}

/// Generate angle weighted by |sin(2*angle)| to favor corners
fn generate_weighted_angle(rng: &mut impl Rng) -> f64 {
    loop {
        let angle = rng.gen_range(0.0..2.0 * PI);
        if rng.gen::<f64>() < (2.0 * angle).sin().abs() {
            return angle;
        }
    }
}

/// Grid-based packing with rotation optimization
pub struct GridPacking;

impl PackingAlgorithm for GridPacking {
    fn pack(&self, n: usize) -> Packing {
        let mut rng = rand::thread_rng();
        let mut packing = Packing::new();

        if n == 0 {
            return packing;
        }

        // Estimate grid size needed
        let trees_per_side = ((n as f64).sqrt().ceil() as usize).max(1);
        let spacing = 0.8; // Slightly larger than tree width

        let mut positions: Vec<(f64, f64)> = Vec::new();
        for row in 0..trees_per_side {
            for col in 0..trees_per_side {
                let x = (col as f64 - (trees_per_side - 1) as f64 / 2.0) * spacing;
                let y = (row as f64 - (trees_per_side - 1) as f64 / 2.0) * spacing;
                positions.push((x, y));
            }
        }

        // Place trees at grid positions
        for i in 0..n {
            if i < positions.len() {
                let (x, y) = positions[i];
                // Try different rotations
                let angles = [0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0];
                let mut placed = false;

                for &angle in &angles {
                    let tree = PlacedTree::new(x, y, angle);
                    if packing.can_place(&tree) {
                        packing.try_add(tree);
                        placed = true;
                        break;
                    }
                }

                if !placed {
                    // Try random angle
                    let angle = rng.gen_range(0.0..360.0);
                    let tree = PlacedTree::new(x, y, angle);
                    packing.try_add(tree);
                }
            }
        }

        packing
    }

    fn name(&self) -> &'static str {
        "grid_packing"
    }
}

/// Simple bottom-left packing
pub struct BottomLeft;

impl PackingAlgorithm for BottomLeft {
    fn pack(&self, n: usize) -> Packing {
        let mut rng = rand::thread_rng();
        let mut packing = Packing::new();

        if n == 0 {
            return packing;
        }

        // Place first tree at origin
        packing.try_add(PlacedTree::new(0.0, 0.0, 0.0));

        for _ in 1..n {
            let tree_angle = rng.gen_range(0.0..360.0);

            // Search for bottom-left position
            let mut best_x = 0.0;
            let mut best_y = 0.0;
            let mut best_score = f64::INFINITY;

            // Grid search
            for ix in -50..=50 {
                for iy in -50..=50 {
                    let x = ix as f64 * 0.2;
                    let y = iy as f64 * 0.2;

                    let candidate = PlacedTree::new(x, y, tree_angle);
                    if packing.can_place(&candidate) {
                        // Score: prefer bottom-left
                        let score = x + y;
                        if score < best_score {
                            best_score = score;
                            best_x = x;
                            best_y = y;
                        }
                    }
                }
            }

            packing.try_add(PlacedTree::new(best_x, best_y, tree_angle));
        }

        packing
    }

    fn name(&self) -> &'static str {
        "bottom_left"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_radial() {
        let algo = GreedyRadial::default();
        let packing = algo.pack(5);
        assert_eq!(packing.trees.len(), 5);
        assert!(!packing.has_overlaps());
    }

    #[test]
    fn test_grid_packing() {
        let algo = GridPacking;
        let packing = algo.pack(9);
        assert_eq!(packing.trees.len(), 9);
    }
}
