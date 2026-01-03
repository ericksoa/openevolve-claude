//! Hybrid algorithm: Evolved placement + Sparrow-style refinement
//!
//! Uses the evolved greedy algorithm for initial placement,
//! then applies Sparrow-inspired compression refinement.

use crate::{Packing, PlacedTree};
use crate::evolved::EvolvedPacker;
use rand::Rng;

/// Tree state for optimization
#[derive(Clone, Copy, Debug)]
struct TreeState {
    x: f64,
    y: f64,
    rotation: usize,
}

impl TreeState {
    fn from_placed(t: &PlacedTree) -> Self {
        let rotation = ((t.angle_deg / 45.0).round() as usize) % 8;
        Self {
            x: t.x,
            y: t.y,
            rotation,
        }
    }

    fn to_placed(&self) -> PlacedTree {
        PlacedTree::new(self.x, self.y, self.rotation as f64 * 45.0)
    }
}

fn bbox_side(trees: &[TreeState]) -> f64 {
    if trees.is_empty() {
        return 0.0;
    }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for t in trees {
        let placed = t.to_placed();
        let (bx1, by1, bx2, by2) = placed.bounds();
        min_x = min_x.min(bx1);
        min_y = min_y.min(by1);
        max_x = max_x.max(bx2);
        max_y = max_y.max(by2);
    }

    (max_x - min_x).max(max_y - min_y)
}

fn is_feasible(trees: &[TreeState]) -> bool {
    let placed: Vec<PlacedTree> = trees.iter().map(|t| t.to_placed()).collect();
    for i in 0..placed.len() {
        for j in (i + 1)..placed.len() {
            if placed[i].overlaps(&placed[j]) {
                return false;
            }
        }
    }
    true
}

fn center_trees(trees: &mut [TreeState]) {
    if trees.is_empty() {
        return;
    }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for t in trees.iter() {
        let placed = t.to_placed();
        let (bx1, by1, bx2, by2) = placed.bounds();
        min_x = min_x.min(bx1);
        min_y = min_y.min(by1);
        max_x = max_x.max(bx2);
        max_y = max_y.max(by2);
    }

    let cx = (min_x + max_x) / 2.0;
    let cy = (min_y + max_y) / 2.0;

    for t in trees.iter_mut() {
        t.x -= cx;
        t.y -= cy;
    }
}

/// Intensive local search refinement
fn refine_packing(trees: &mut [TreeState], iterations: usize) -> f64 {
    let mut rng = rand::thread_rng();
    let n = trees.len();

    if n == 0 {
        return 0.0;
    }

    let mut best_side = bbox_side(trees);
    let mut best_trees = trees.to_vec();

    for iter in 0..iterations {
        let progress = iter as f64 / iterations as f64;
        let temp = 0.5 * (1.0 - progress);

        // Random move
        let idx = rng.gen_range(0..n);
        let old = trees[idx].clone();

        let step = 0.03 * (1.0 - 0.5 * progress);

        match rng.gen_range(0..6) {
            0 => {
                // Small translation
                trees[idx].x += rng.gen_range(-step..step);
                trees[idx].y += rng.gen_range(-step..step);
            }
            1 => {
                // Move toward center
                let (min_x, min_y, max_x, max_y) = {
                    let mut mi_x = f64::INFINITY;
                    let mut mi_y = f64::INFINITY;
                    let mut ma_x = f64::NEG_INFINITY;
                    let mut ma_y = f64::NEG_INFINITY;
                    for t in trees.iter() {
                        let p = t.to_placed();
                        let (bx1, by1, bx2, by2) = p.bounds();
                        mi_x = mi_x.min(bx1);
                        mi_y = mi_y.min(by1);
                        ma_x = ma_x.max(bx2);
                        ma_y = ma_y.max(by2);
                    }
                    (mi_x, mi_y, ma_x, ma_y)
                };
                let cx = (min_x + max_x) / 2.0;
                let cy = (min_y + max_y) / 2.0;
                trees[idx].x += (cx - old.x) * step * 2.0;
                trees[idx].y += (cy - old.y) * step * 2.0;
            }
            2 => {
                // Rotation change
                trees[idx].rotation = rng.gen_range(0..8);
            }
            3 => {
                // Move along one axis
                if rng.gen() {
                    trees[idx].x += rng.gen_range(-step * 2.0..step * 2.0);
                } else {
                    trees[idx].y += rng.gen_range(-step * 2.0..step * 2.0);
                }
            }
            4 => {
                // Swap with another tree's position
                let other = rng.gen_range(0..n);
                if other != idx {
                    let tmp_x = trees[idx].x;
                    let tmp_y = trees[idx].y;
                    trees[idx].x = trees[other].x;
                    trees[idx].y = trees[other].y;
                    trees[other].x = tmp_x;
                    trees[other].y = tmp_y;
                }
            }
            _ => {
                // Combined move
                trees[idx].x += rng.gen_range(-step..step);
                trees[idx].y += rng.gen_range(-step..step);
                if rng.gen::<f64>() < 0.3 {
                    trees[idx].rotation = rng.gen_range(0..8);
                }
            }
        }

        if is_feasible(trees) {
            let side = bbox_side(trees);
            if side < best_side {
                best_side = side;
                best_trees = trees.to_vec();
            } else if side < best_side * 1.005 {
                // Accept slightly worse moves
                // (simulated annealing)
            } else if rng.gen::<f64>() < temp * 0.1 {
                // Accept with very small probability
            } else {
                trees[idx] = old;
            }
        } else {
            trees[idx] = old;
        }
    }

    trees.copy_from_slice(&best_trees);
    best_side
}

pub struct HybridPacker {
    pub refinement_iters: usize,
}

impl Default for HybridPacker {
    fn default() -> Self {
        Self {
            refinement_iters: 20000,
        }
    }
}

impl HybridPacker {
    pub fn pack(&self, n: usize) -> Packing {
        // Use evolved for initial placement
        let evolved = EvolvedPacker::default();
        let packings = evolved.pack_all(n);
        let initial = &packings[n - 1];

        // Convert to TreeState for refinement
        let mut trees: Vec<TreeState> = initial
            .trees
            .iter()
            .map(|t| TreeState::from_placed(t))
            .collect();

        // Apply refinement
        let _final_side = refine_packing(&mut trees, self.refinement_iters);

        center_trees(&mut trees);

        // Convert back to Packing
        let mut packing = Packing::new();
        for t in trees {
            packing.trees.push(t.to_placed());
        }
        packing
    }

    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        // Get all evolved packings first
        let evolved = EvolvedPacker::default();
        let evolved_packings = evolved.pack_all(max_n);

        // Refine each one
        let mut result = Vec::with_capacity(max_n);

        for (n_idx, initial) in evolved_packings.iter().enumerate() {
            let n = n_idx + 1;

            let mut trees: Vec<TreeState> = initial
                .trees
                .iter()
                .map(|t| TreeState::from_placed(t))
                .collect();

            // More iterations for larger n
            let iters = self.refinement_iters.min(5000 + n * 500);
            let _final_side = refine_packing(&mut trees, iters);

            center_trees(&mut trees);

            let mut packing = Packing::new();
            for t in trees {
                packing.trees.push(t.to_placed());
            }
            result.push(packing);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid() {
        let packer = HybridPacker::default();
        let packing = packer.pack(10);
        assert_eq!(packing.trees.len(), 10);
        assert!(!packing.has_overlaps());
    }
}
