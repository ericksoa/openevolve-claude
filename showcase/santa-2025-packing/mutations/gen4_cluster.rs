//! Evolved Packing Algorithm - Generation 4 HIERARCHICAL CLUSTERING
//!
//! This module contains the evolved packing heuristics.
//! The code is designed to be mutated by LLM-guided evolution.
//!
//! Evolution targets:
//! - placement_score(): How to score candidate placements
//! - select_angles(): Which rotation angles to try
//! - select_direction(): How to choose placement directions
//! - sa_move(): Local search move operators
//!
//! MUTATION STRATEGY: HIERARCHICAL CLUSTERING (Gen4)
//! Building on Gen3 champion (score 101.90) with clustering-based optimization:
//!
//! Key innovations:
//! 1. Cluster detection - identify groups of trees that pack well together
//!    based on spatial proximity and complementary orientations
//! 2. Cluster-level optimization - optimize cluster positions/rotations as units
//!    to find better global arrangements more efficiently
//! 3. Intra-cluster fine-tuning - optimize individual trees within clusters
//!    for local compaction
//! 4. Hierarchical SA - first optimize at cluster level, then refine individuals
//!
//! Hypothesis: Trees should be packed in clusters that can be efficiently arranged.
//! By treating clusters as units, we can reduce the search space and find better
//! global arrangements while still achieving local optimality.
//!
//! Parameters:
//! - cluster_distance_threshold: 0.8 (trees within this distance form clusters)
//! - min_cluster_size: 2 (minimum trees to form a cluster)
//! - cluster_sa_iterations: 15000 (iterations for cluster-level optimization)
//! - intra_cluster_iterations: 8000 (iterations for within-cluster optimization)
//! - final_global_iterations: 20000 (final global refinement)

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Evolved packing configuration with clustering parameters
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

    // Multi-pass settings
    pub sa_passes: usize,

    // Restart mechanism
    pub restart_threshold: usize,
    pub reheat_temp: f64,

    // Greedy compaction
    pub compaction_iterations: usize,

    // CLUSTERING: New parameters for hierarchical clustering
    pub cluster_distance_threshold: f64,  // Max distance between tree centers to be in same cluster
    pub min_cluster_size: usize,          // Minimum trees to form a meaningful cluster
    pub cluster_sa_iterations: usize,     // SA iterations for cluster-level optimization
    pub intra_cluster_iterations: usize,  // SA iterations within each cluster
    pub cluster_move_scale: f64,          // Scale for moving entire clusters
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        Self {
            search_attempts: 400,
            direction_samples: 96,
            sa_iterations: 20000,              // Reduced base - we have multiple phases
            sa_initial_temp: 0.7,
            sa_cooling_rate: 0.99995,          // Slightly faster cooling per phase
            sa_min_temp: 0.000001,
            translation_scale: 0.08,
            rotation_granularity: 22.5,
            center_pull_strength: 0.06,
            sa_passes: 2,                       // Fewer passes, clustering adds phases
            restart_threshold: 4000,
            reheat_temp: 0.4,
            compaction_iterations: 1500,

            // CLUSTERING parameters
            cluster_distance_threshold: 0.9,   // Trees within 0.9 units form clusters
            min_cluster_size: 2,               // At least 2 trees to form cluster
            cluster_sa_iterations: 12000,      // Cluster-level optimization
            intra_cluster_iterations: 6000,    // Within-cluster optimization
            cluster_move_scale: 0.12,          // Larger moves for clusters
        }
    }
}

/// Represents a cluster of trees
#[derive(Clone)]
struct TreeCluster {
    tree_indices: Vec<usize>,    // Indices of trees in this cluster
    center_x: f64,               // Cluster centroid X
    center_y: f64,               // Cluster centroid Y
}

impl TreeCluster {
    fn new() -> Self {
        Self {
            tree_indices: Vec::new(),
            center_x: 0.0,
            center_y: 0.0,
        }
    }

    fn compute_centroid(&mut self, trees: &[PlacedTree]) {
        if self.tree_indices.is_empty() {
            return;
        }
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for &idx in &self.tree_indices {
            sum_x += trees[idx].x;
            sum_y += trees[idx].y;
        }
        let n = self.tree_indices.len() as f64;
        self.center_x = sum_x / n;
        self.center_y = sum_y / n;
    }
}

/// Main evolved packer with clustering
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

            // CLUSTERING: Multi-phase optimization
            // Phase 1: Standard SA passes for initial optimization
            for pass in 0..self.config.sa_passes {
                self.local_search(&mut trees, n, pass, &mut rng);
            }

            // Phase 2: Cluster-based optimization (only when we have enough trees)
            if trees.len() >= 4 {
                self.cluster_optimization(&mut trees, &mut rng);
            }

            // Phase 3: Final greedy compaction
            self.greedy_compaction(&mut trees, &mut rng);

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

    /// CLUSTERING: Main cluster-based optimization routine
    fn cluster_optimization(&self, trees: &mut Vec<PlacedTree>, rng: &mut impl Rng) {
        // Step 1: Detect clusters
        let clusters = self.detect_clusters(trees);

        // Only proceed if we found meaningful clusters
        if clusters.len() < 2 {
            return;
        }

        // Step 2: Cluster-level optimization - move clusters as units
        self.optimize_cluster_positions(trees, &clusters, rng);

        // Step 3: Intra-cluster optimization - fine-tune within each cluster
        for cluster in &clusters {
            if cluster.tree_indices.len() >= self.config.min_cluster_size {
                self.optimize_within_cluster(trees, cluster, rng);
            }
        }
    }

    /// CLUSTERING: Detect clusters using distance-based grouping
    fn detect_clusters(&self, trees: &[PlacedTree]) -> Vec<TreeCluster> {
        let n = trees.len();
        if n < 2 {
            return vec![];
        }

        // Union-Find for clustering
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank: Vec<usize> = vec![0; n];

        fn find(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
            let px = find(parent, x);
            let py = find(parent, y);
            if px != py {
                if rank[px] < rank[py] {
                    parent[px] = py;
                } else if rank[px] > rank[py] {
                    parent[py] = px;
                } else {
                    parent[py] = px;
                    rank[px] += 1;
                }
            }
        }

        // Group trees by distance
        let threshold = self.config.cluster_distance_threshold;
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = trees[i].x - trees[j].x;
                let dy = trees[i].y - trees[j].y;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist < threshold {
                    union(&mut parent, &mut rank, i, j);
                }
            }
        }

        // Build clusters from union-find results
        let mut cluster_map: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            cluster_map.entry(root).or_default().push(i);
        }

        // Convert to TreeCluster objects
        let mut clusters: Vec<TreeCluster> = Vec::new();
        for (_, indices) in cluster_map {
            if indices.len() >= self.config.min_cluster_size {
                let mut cluster = TreeCluster::new();
                cluster.tree_indices = indices;
                cluster.compute_centroid(trees);
                clusters.push(cluster);
            }
        }

        clusters
    }

    /// CLUSTERING: Optimize cluster positions - move entire clusters as units
    fn optimize_cluster_positions(
        &self,
        trees: &mut Vec<PlacedTree>,
        clusters: &[TreeCluster],
        rng: &mut impl Rng,
    ) {
        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config = trees.clone();

        let mut temp = self.config.sa_initial_temp * 0.6;  // Start slightly cooler
        let iterations = self.config.cluster_sa_iterations;

        for _ in 0..iterations {
            // Pick a random cluster
            let cluster_idx = rng.gen_range(0..clusters.len());
            let cluster = &clusters[cluster_idx];

            // Save old positions
            let old_positions: Vec<(f64, f64, f64)> = cluster.tree_indices.iter()
                .map(|&i| (trees[i].x, trees[i].y, trees[i].angle_deg))
                .collect();

            // Apply cluster move
            let success = self.cluster_move(trees, cluster, temp, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                    current_side = new_side;
                    if current_side < best_side {
                        best_side = current_side;
                        best_config = trees.clone();
                    }
                } else {
                    // Revert
                    for (i, &idx) in cluster.tree_indices.iter().enumerate() {
                        trees[idx] = PlacedTree::new(
                            old_positions[i].0,
                            old_positions[i].1,
                            old_positions[i].2,
                        );
                    }
                }
            } else {
                // Revert
                for (i, &idx) in cluster.tree_indices.iter().enumerate() {
                    trees[idx] = PlacedTree::new(
                        old_positions[i].0,
                        old_positions[i].1,
                        old_positions[i].2,
                    );
                }
            }

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
        }

        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    /// CLUSTERING: Move an entire cluster as a unit
    fn cluster_move(
        &self,
        trees: &mut [PlacedTree],
        cluster: &TreeCluster,
        temp: f64,
        rng: &mut impl Rng,
    ) -> bool {
        let move_type = rng.gen_range(0..5);

        match move_type {
            0 => {
                // Translate entire cluster
                let scale = self.config.cluster_move_scale * (0.2 + temp * 2.0);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);

                for &idx in &cluster.tree_indices {
                    let old = &trees[idx];
                    trees[idx] = PlacedTree::new(old.x + dx, old.y + dy, old.angle_deg);
                }
            }
            1 => {
                // Rotate cluster around its centroid
                let delta_angle = rng.gen_range(-15.0..15.0) * (0.5 + temp);
                let rad = delta_angle * PI / 180.0;
                let cos_a = rad.cos();
                let sin_a = rad.sin();

                for &idx in &cluster.tree_indices {
                    let old = &trees[idx];
                    // Rotate position around cluster center
                    let rel_x = old.x - cluster.center_x;
                    let rel_y = old.y - cluster.center_y;
                    let new_x = cluster.center_x + rel_x * cos_a - rel_y * sin_a;
                    let new_y = cluster.center_y + rel_x * sin_a + rel_y * cos_a;
                    // Also rotate the tree itself
                    let new_angle = (old.angle_deg + delta_angle).rem_euclid(360.0);
                    trees[idx] = PlacedTree::new(new_x, new_y, new_angle);
                }
            }
            2 => {
                // Move cluster toward global center
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let global_cx = (min_x + max_x) / 2.0;
                let global_cy = (min_y + max_y) / 2.0;

                let dx = global_cx - cluster.center_x;
                let dy = global_cy - cluster.center_y;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist > 0.05 {
                    let scale = 0.08 * (0.3 + temp * 1.5);
                    let move_x = dx / dist * scale;
                    let move_y = dy / dist * scale;

                    for &idx in &cluster.tree_indices {
                        let old = &trees[idx];
                        trees[idx] = PlacedTree::new(old.x + move_x, old.y + move_y, old.angle_deg);
                    }
                } else {
                    return false;
                }
            }
            3 => {
                // Scale cluster (move trees toward/away from cluster center)
                let scale_factor = rng.gen_range(0.92..1.08);

                for &idx in &cluster.tree_indices {
                    let old = &trees[idx];
                    let rel_x = old.x - cluster.center_x;
                    let rel_y = old.y - cluster.center_y;
                    let new_x = cluster.center_x + rel_x * scale_factor;
                    let new_y = cluster.center_y + rel_y * scale_factor;
                    trees[idx] = PlacedTree::new(new_x, new_y, old.angle_deg);
                }
            }
            _ => {
                // Small nudge + uniform rotation of all trees in cluster
                let scale = 0.04 * (0.4 + temp);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                let delta_angle = if rng.gen() { 22.5 } else { -22.5 };

                for &idx in &cluster.tree_indices {
                    let old = &trees[idx];
                    let new_angle = (old.angle_deg + delta_angle).rem_euclid(360.0);
                    trees[idx] = PlacedTree::new(old.x + dx, old.y + dy, new_angle);
                }
            }
        }

        // Check for overlaps - only between cluster trees and non-cluster trees
        // Trees within cluster are allowed to have moved relative to each other
        for &idx in &cluster.tree_indices {
            if has_overlap(trees, idx) {
                return false;
            }
        }
        true
    }

    /// CLUSTERING: Optimize trees within a single cluster
    fn optimize_within_cluster(
        &self,
        trees: &mut Vec<PlacedTree>,
        cluster: &TreeCluster,
        rng: &mut impl Rng,
    ) {
        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config = trees.clone();

        let mut temp = self.config.sa_initial_temp * 0.3;  // Lower temp for fine-tuning
        let iterations = self.config.intra_cluster_iterations;

        for iter in 0..iterations {
            // Pick a random tree from this cluster
            let local_idx = rng.gen_range(0..cluster.tree_indices.len());
            let tree_idx = cluster.tree_indices[local_idx];

            let old_tree = trees[tree_idx].clone();

            // Apply individual tree move (reuse existing sa_move logic)
            let success = self.sa_move(trees, tree_idx, temp, iter, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                    current_side = new_side;
                    if current_side < best_side {
                        best_side = current_side;
                        best_config = trees.clone();
                    }
                } else {
                    trees[tree_idx] = old_tree;
                }
            } else {
                trees[tree_idx] = old_tree;
            }

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
        }

        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    /// EVOLVED FUNCTION: Find best placement for new tree
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

        for _ in 0..self.config.search_attempts {
            let dir = self.select_direction(n, rng);
            let vx = dir.cos();
            let vy = dir.sin();

            for &tree_angle in &angles {
                // Binary search for closest valid position
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > 0.0001 {
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
    /// CLUSTERING: Added cluster affinity bonus
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
        let balance_weight = 0.12 + 0.06 * (1.0 - (n as f64 / 200.0).min(1.0));
        let balance_penalty = (width - height).abs() * balance_weight;

        // Tertiary: slight preference for compact center
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.010 / (n as f64).sqrt();

        // Density heuristic
        let area = width * height;
        let density_bonus = if area > 0.0 {
            -0.008 * (n as f64 / area).min(2.5)
        } else {
            0.0
        };

        // Perimeter minimization bonus
        let perimeter_bonus = -0.002 * (2.0 * (width + height)) / (n as f64).sqrt();

        // CLUSTERING: Cluster affinity bonus - prefer placing near existing trees
        // to facilitate cluster formation
        let cluster_bonus = if !existing.is_empty() {
            let mut min_dist = f64::INFINITY;
            for other in existing {
                let dx = tree.x - other.x;
                let dy = tree.y - other.y;
                let dist = (dx * dx + dy * dy).sqrt();
                min_dist = min_dist.min(dist);
            }
            // Small bonus for being close but not too close
            if min_dist < self.config.cluster_distance_threshold && min_dist > 0.3 {
                -0.005 * (1.0 - min_dist / self.config.cluster_distance_threshold)
            } else {
                0.0
            }
        } else {
            0.0
        };

        side_score + balance_penalty + center_penalty + density_bonus + perimeter_bonus + cluster_bonus
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        let base = match n % 8 {
            0 => vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0,
                      22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5],
            1 => vec![90.0, 270.0, 0.0, 180.0, 135.0, 315.0, 45.0, 225.0,
                      67.5, 112.5, 247.5, 292.5, 22.5, 157.5, 202.5, 337.5],
            2 => vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0,
                      157.5, 202.5, 337.5, 22.5, 67.5, 112.5, 247.5, 292.5],
            3 => vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0,
                      247.5, 292.5, 67.5, 112.5, 157.5, 202.5, 337.5, 22.5],
            4 => vec![45.0, 225.0, 135.0, 315.0, 0.0, 90.0, 180.0, 270.0,
                      22.5, 67.5, 202.5, 247.5, 112.5, 157.5, 292.5, 337.5],
            5 => vec![135.0, 315.0, 45.0, 225.0, 90.0, 270.0, 0.0, 180.0,
                      112.5, 157.5, 292.5, 337.5, 22.5, 67.5, 202.5, 247.5],
            6 => vec![22.5, 202.5, 67.5, 247.5, 112.5, 292.5, 157.5, 337.5,
                      0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
            _ => vec![67.5, 247.5, 22.5, 202.5, 112.5, 292.5, 157.5, 337.5,
                      45.0, 135.0, 225.0, 315.0, 0.0, 90.0, 180.0, 270.0],
        };
        base
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        let strategy = rng.gen::<f64>();

        if strategy < 0.45 {
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.05..0.05)
        } else if strategy < 0.65 {
            loop {
                let angle = rng.gen_range(0.0..2.0 * PI);
                let corner_weight = ((4.0 * angle).sin().abs() + (4.0 * angle).cos().abs()) / 2.0;
                let threshold = 0.15 + 0.12 * (1.0 - (n as f64 / 200.0).min(1.0));
                if rng.gen::<f64>() < corner_weight.max(threshold) {
                    return angle;
                }
            }
        } else if strategy < 0.85 {
            let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
            let base = (n as f64 * golden_angle) % (2.0 * PI);
            let offset = rng.gen_range(0..16) as f64 * PI / 8.0;
            (base + offset + rng.gen_range(-0.08..0.08)) % (2.0 * PI)
        } else {
            let idx = rng.gen_range(0..num_dirs);
            let golden_ratio = (1.0 + (5.0_f64).sqrt()) / 2.0;
            ((idx as f64 * 2.0 * PI / golden_ratio) % (2.0 * PI)) + rng.gen_range(-0.03..0.03)
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, pass: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        let temp_multiplier = match pass {
            0 => 1.0,
            _ => 0.4,
        };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        let base_iterations = match pass {
            0 => self.config.sa_iterations + n * 150,
            _ => self.config.sa_iterations / 2 + n * 75,
        };

        let mut iterations_without_improvement = 0;

        for iter in 0..base_iterations {
            let idx = self.select_tree_to_move(trees, rng);
            let old_tree = trees[idx].clone();

            let success = self.sa_move(trees, idx, temp, iter, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                    current_side = new_side;
                    if current_side < best_side {
                        best_side = current_side;
                        best_config = trees.clone();
                        iterations_without_improvement = 0;
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

            if iterations_without_improvement >= self.config.restart_threshold {
                temp = self.config.reheat_temp * temp_multiplier;
                iterations_without_improvement = 0;
                *trees = best_config.clone();
                current_side = best_side;
            }

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
        }

        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    /// Greedy compaction phase after SA
    fn greedy_compaction(&self, trees: &mut Vec<PlacedTree>, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut best_side = compute_side_length(trees);
        let mut improved = true;
        let mut iterations = 0;

        while improved && iterations < self.config.compaction_iterations {
            improved = false;
            iterations += 1;

            for idx in 0..trees.len() {
                let old_tree = trees[idx].clone();
                let (old_x, old_y, old_angle) = (old_tree.x, old_tree.y, old_tree.angle_deg);

                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let center_x = (min_x + max_x) / 2.0;
                let center_y = (min_y + max_y) / 2.0;

                let dx = center_x - old_x;
                let dy = center_y - old_y;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist > 0.01 {
                    let step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002];

                    for &step in &step_sizes {
                        let new_x = old_x + dx / dist * step;
                        let new_y = old_y + dy / dist * step;

                        trees[idx] = PlacedTree::new(new_x, new_y, old_angle);

                        if !has_overlap(trees, idx) {
                            let new_side = compute_side_length(trees);
                            if new_side < best_side - 1e-9 {
                                best_side = new_side;
                                improved = true;
                                break;
                            }
                        }

                        trees[idx] = old_tree.clone();
                    }
                }

                if !improved && rng.gen::<f64>() < 0.3 {
                    for delta_angle in &[22.5, -22.5, 45.0, -45.0, 11.25, -11.25] {
                        let new_angle = (old_angle + delta_angle).rem_euclid(360.0);
                        trees[idx] = PlacedTree::new(old_x, old_y, new_angle);

                        if !has_overlap(trees, idx) {
                            let new_side = compute_side_length(trees);
                            if new_side < best_side - 1e-9 {
                                best_side = new_side;
                                improved = true;
                                break;
                            }
                        }

                        trees[idx] = old_tree.clone();
                    }
                }
            }
        }
    }

    /// Select tree to move with preference for boundary trees
    #[inline]
    fn select_tree_to_move(&self, trees: &[PlacedTree], rng: &mut impl Rng) -> usize {
        if trees.len() <= 2 || rng.gen::<f64>() < 0.6 {
            return rng.gen_range(0..trees.len());
        }

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

        let mut boundary_indices: Vec<usize> = Vec::new();
        let eps = 0.02;

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

        let move_type = rng.gen_range(0..12);

        match move_type {
            0 => {
                let scale = self.config.translation_scale * (0.15 + temp * 2.8);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                let new_angle = (old_angle + 90.0).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            2 => {
                let delta = if rng.gen() { self.config.rotation_granularity }
                            else { -self.config.rotation_granularity };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            3 => {
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.04 {
                    let scale = self.config.center_pull_strength * (0.35 + temp * 1.8);
                    let dx = -old_x / mag * scale;
                    let dy = -old_y / mag * scale;
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                } else {
                    return false;
                }
            }
            4 => {
                let scale = self.config.translation_scale * 0.35;
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                let dangle = rng.gen_range(-45.0..45.0);
                let new_angle = (old_angle + dangle).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            5 => {
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.08 {
                    let delta_r = rng.gen_range(-0.07..0.07) * (1.0 + temp);
                    let new_mag = (mag + delta_r).max(0.0);
                    let scale = new_mag / mag;
                    trees[idx] = PlacedTree::new(old_x * scale, old_y * scale, old_angle);
                } else {
                    return false;
                }
            }
            6 => {
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.08 {
                    let current_angle = old_y.atan2(old_x);
                    let delta_angle = rng.gen_range(-0.18..0.18) * (1.0 + temp);
                    let new_ang = current_angle + delta_angle;
                    trees[idx] = PlacedTree::new(mag * new_ang.cos(), mag * new_ang.sin(), old_angle);
                } else {
                    return false;
                }
            }
            7 => {
                let scale = 0.012 * (0.4 + temp);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            8 => {
                let new_angle = (old_angle + 180.0).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            9 => {
                let corner_idx = rng.gen_range(0..4);
                let (dir_x, dir_y) = match corner_idx {
                    0 => (-1.0, -1.0),
                    1 => (1.0, -1.0),
                    2 => (-1.0, 1.0),
                    _ => (1.0, 1.0),
                };
                let scale = 0.025 * (0.4 + temp * 1.6);
                let norm = (2.0_f64).sqrt();
                trees[idx] = PlacedTree::new(
                    old_x + dir_x * scale / norm,
                    old_y + dir_y * scale / norm,
                    old_angle
                );
            }
            10 => {
                let delta = if rng.gen() { 11.25 } else { -11.25 };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            _ => {
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.1 {
                    let current_angle = old_y.atan2(old_x);
                    let delta_r = rng.gen_range(-0.04..0.04) * (1.0 + temp);
                    let delta_angle = rng.gen_range(-0.1..0.1) * (1.0 + temp);
                    let new_mag = (mag + delta_r).max(0.0);
                    let new_ang = current_angle + delta_angle;
                    trees[idx] = PlacedTree::new(new_mag * new_ang.cos(), new_mag * new_ang.sin(), old_angle);
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

    #[test]
    fn test_cluster_detection() {
        let packer = EvolvedPacker::default();
        // Create some trees close together
        let trees = vec![
            PlacedTree::new(0.0, 0.0, 0.0),
            PlacedTree::new(0.5, 0.0, 90.0),
            PlacedTree::new(0.0, 0.5, 180.0),
            PlacedTree::new(3.0, 3.0, 0.0),  // Far away - separate cluster
        ];
        let clusters = packer.detect_clusters(&trees);
        assert!(clusters.len() >= 1);
    }
}
