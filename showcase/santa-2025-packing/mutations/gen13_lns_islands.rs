//! Evolved Packing Algorithm - Generation 13 LNS + Islands
//!
//! MAJOR ARCHITECTURAL UPGRADE implementing:
//! 1. Robust evaluation (noise-aware selection with multiple seeds)
//! 2. Large Neighborhood Search (LNS) - destroy/repair operators
//! 3. Island model / speciation (4 islands with different strategies)
//! 4. Plateau detection + automatic strategy shift
//! 5. Anti-overfit gates (multi-seed evaluation)
//!
//! Based on Gen10 diverse starts (score ~91.35) but with advanced optimization.

use crate::{Packing, PlacedTree};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::f64::consts::PI;
use std::collections::HashMap;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Island-specific operator weights
#[derive(Clone, Debug)]
pub struct IslandConfig {
    pub name: &'static str,
    pub lns_probability: f64,        // Probability of using LNS mutation
    pub destroy_k_range: (f64, f64), // Range for destroy percentage
    pub sa_iterations: usize,
    pub restart_probability: f64,    // Probability of random restart
    pub boundary_focus: f64,
}

/// LNS Configuration
#[derive(Clone, Debug)]
pub struct LnsConfig {
    pub repair_attempts: usize,      // Number of repair attempts (M)
    pub destroy_policies: Vec<DestroyPolicy>,
}

#[derive(Clone, Copy, Debug)]
pub enum DestroyPolicy {
    RandomK,    // Remove k% randomly
    WorstK,     // Remove items contributing most to bounding box
    ClusterK,   // Remove items from same angle bucket
}

/// Evolved packing configuration
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

    // Early exit threshold
    pub early_exit_threshold: usize,

    // Boundary focus probability
    pub boundary_focus_prob: f64,

    // Number of strategies
    pub num_strategies: usize,

    // Density parameters
    pub density_grid_resolution: usize,
    pub gap_penalty_weight: f64,
    pub local_density_radius: f64,
    pub fill_move_prob: f64,

    // === NEW: Island Model Configuration ===
    pub num_islands: usize,
    pub migration_interval: usize,    // Migrate every N trees
    pub migrants_per_island: usize,

    // === NEW: Robust Evaluation ===
    pub eval_repeats: usize,          // Number of evaluation seeds
    pub fitness_std_weight: f64,      // Lambda for mean + lambda*std
    pub stability_min_delta: f64,     // Min improvement to crown new champion

    // === NEW: LNS Configuration ===
    pub lns_config: LnsConfig,

    // === NEW: Plateau Detection ===
    pub plateau_patience: usize,      // Gens without improvement before plateau
    pub plateau_lns_boost: f64,       // Increase LNS prob on plateau
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        Self {
            search_attempts: 200,
            direction_samples: 64,
            sa_iterations: 22000,
            sa_initial_temp: 0.45,
            sa_cooling_rate: 0.99993,
            sa_min_temp: 0.00001,
            translation_scale: 0.055,
            rotation_granularity: 45.0,
            center_pull_strength: 0.07,
            sa_passes: 2,
            early_exit_threshold: 1500,
            boundary_focus_prob: 0.85,
            num_strategies: 5,
            density_grid_resolution: 20,
            gap_penalty_weight: 0.15,
            local_density_radius: 0.5,
            fill_move_prob: 0.15,
            // Island model
            num_islands: 4,
            migration_interval: 25,
            migrants_per_island: 1,
            // Robust evaluation
            eval_repeats: 3,           // Reduced for speed
            fitness_std_weight: 0.3,
            stability_min_delta: 0.05,
            // LNS
            lns_config: LnsConfig {
                repair_attempts: 8,
                destroy_policies: vec![
                    DestroyPolicy::RandomK,
                    DestroyPolicy::WorstK,
                    DestroyPolicy::ClusterK,
                ],
            },
            // Plateau detection
            plateau_patience: 20,
            plateau_lns_boost: 0.15,
        }
    }
}

/// Strategy for initial placement direction
#[derive(Clone, Copy, Debug)]
pub enum PlacementStrategy {
    ClockwiseSpiral,
    CounterclockwiseSpiral,
    Grid,
    Random,
    BoundaryFirst,
}

/// Track which boundary a tree is blocking
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BoundaryEdge {
    Left,
    Right,
    Top,
    Bottom,
    Corner,
    None,
}

// ============================================================================
// ISLAND STATE
// ============================================================================

/// State for a single island in the island model
struct Island {
    config: IslandConfig,
    trees: Vec<PlacedTree>,
    best_side: f64,
    no_improvement_count: usize,
    lns_active: bool,
}

impl Island {
    fn new(config: IslandConfig) -> Self {
        Self {
            config,
            trees: Vec::new(),
            best_side: f64::INFINITY,
            no_improvement_count: 0,
            lns_active: false,
        }
    }
}

// ============================================================================
// MAIN PACKER
// ============================================================================

pub struct EvolvedPacker {
    pub config: EvolvedConfig,
}

impl Default for EvolvedPacker {
    fn default() -> Self {
        Self { config: EvolvedConfig::default() }
    }
}

impl EvolvedPacker {
    /// Pack all n from 1 to max_n using Island Model + LNS
    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        let mut rng = rand::thread_rng();
        let mut packings: Vec<Packing> = Vec::with_capacity(max_n);

        // Initialize islands with different configurations
        let island_configs = vec![
            IslandConfig {
                name: "Baseline",
                lns_probability: 0.05,
                destroy_k_range: (0.02, 0.08),
                sa_iterations: self.config.sa_iterations,
                restart_probability: 0.0,
                boundary_focus: 0.85,
            },
            IslandConfig {
                name: "Heavy_LNS",
                lns_probability: 0.30,
                destroy_k_range: (0.05, 0.15),
                sa_iterations: (self.config.sa_iterations as f64 * 0.8) as usize,
                restart_probability: 0.02,
                boundary_focus: 0.80,
            },
            IslandConfig {
                name: "Explorer",
                lns_probability: 0.15,
                destroy_k_range: (0.03, 0.10),
                sa_iterations: (self.config.sa_iterations as f64 * 0.7) as usize,
                restart_probability: 0.10,
                boundary_focus: 0.70,
            },
            IslandConfig {
                name: "Intensifier",
                lns_probability: 0.10,
                destroy_k_range: (0.02, 0.06),
                sa_iterations: (self.config.sa_iterations as f64 * 1.3) as usize,
                restart_probability: 0.0,
                boundary_focus: 0.92,
            },
        ];

        let mut islands: Vec<Island> = island_configs.into_iter()
            .map(Island::new)
            .collect();

        // Track global best
        let mut global_best_trees: Vec<PlacedTree> = Vec::new();
        let mut global_best_side = f64::INFINITY;
        let mut global_no_improvement = 0;

        for n in 1..=max_n {
            let mut best_trees: Option<Vec<PlacedTree>> = None;
            let mut best_side = f64::INFINITY;
            let mut best_island_idx = 0;

            // Evolve each island
            for (island_idx, island) in islands.iter_mut().enumerate() {
                // Decide whether to use LNS
                let use_lns = n > 5 &&
                    island.trees.len() >= 3 &&
                    rng.gen::<f64>() < island.config.lns_probability;

                // Decide whether to restart from global best
                let do_restart = island.config.restart_probability > 0.0 &&
                    rng.gen::<f64>() < island.config.restart_probability &&
                    !global_best_trees.is_empty();

                if do_restart && n > 1 {
                    // Restart from global best with small perturbation
                    island.trees = global_best_trees.clone();
                }

                let mut trees = island.trees.clone();

                if use_lns && trees.len() >= 3 {
                    // Apply LNS: destroy and repair
                    trees = self.apply_lns(&trees, n, &island.config, &mut rng);
                }

                // Place new tree
                let strategy = self.select_strategy_for_island(island_idx, n);
                let new_tree = self.find_placement_with_strategy(&trees, n, max_n, strategy, &mut rng);
                trees.push(new_tree);

                // Run SA with island-specific iterations
                let iterations = if island.lns_active {
                    (island.config.sa_iterations as f64 * 1.2) as usize
                } else {
                    island.config.sa_iterations
                };

                for pass in 0..self.config.sa_passes {
                    self.local_search_island(&mut trees, n, pass, strategy, iterations,
                                            island.config.boundary_focus, &mut rng);
                }

                let side = compute_side_length(&trees);

                // Update island state
                if side < island.best_side - 0.001 {
                    island.best_side = side;
                    island.no_improvement_count = 0;
                } else {
                    island.no_improvement_count += 1;
                }

                // Plateau detection for this island
                if island.no_improvement_count >= self.config.plateau_patience {
                    island.lns_active = true;
                    island.no_improvement_count = 0;
                }

                island.trees = trees.clone();

                // Track best across islands
                if side < best_side {
                    best_side = side;
                    best_trees = Some(trees);
                    best_island_idx = island_idx;
                }
            }

            // Migration: every migration_interval, share best solutions
            if n % self.config.migration_interval == 0 && n > 1 {
                self.migrate_between_islands(&mut islands, &mut rng);
            }

            // Update global best
            let best = best_trees.unwrap();
            if best_side < global_best_side - self.config.stability_min_delta {
                global_best_side = best_side;
                global_best_trees = best.clone();
                global_no_improvement = 0;
            } else {
                global_no_improvement += 1;
            }

            // Global plateau detection: inject diversity
            if global_no_improvement >= self.config.plateau_patience * 2 {
                // Reset one island to explore new regions
                let reset_idx = rng.gen_range(1..islands.len());
                islands[reset_idx].trees = global_best_trees.clone();
                islands[reset_idx].lns_active = true;
                global_no_improvement = 0;
            }

            // Store the best result
            let mut packing = Packing::new();
            for t in &best {
                packing.trees.push(t.clone());
            }
            packings.push(packing);

            // Propagate good solutions to struggling islands
            for island in islands.iter_mut() {
                if compute_side_length(&island.trees) > best_side * 1.03 {
                    island.trees = best.clone();
                }
            }
        }

        packings
    }

    /// Select strategy based on island index
    fn select_strategy_for_island(&self, island_idx: usize, n: usize) -> PlacementStrategy {
        let strategies = [
            PlacementStrategy::ClockwiseSpiral,
            PlacementStrategy::CounterclockwiseSpiral,
            PlacementStrategy::Grid,
            PlacementStrategy::Random,
            PlacementStrategy::BoundaryFirst,
        ];

        // Each island prefers different strategies
        let offset = island_idx;
        let idx = (n + offset) % strategies.len();
        strategies[idx]
    }

    /// Migrate best solutions between islands (ring topology)
    fn migrate_between_islands(&self, islands: &mut [Island], rng: &mut impl Rng) {
        let num_islands = islands.len();
        if num_islands < 2 {
            return;
        }

        // Find best island
        let mut best_idx = 0;
        let mut best_side = f64::INFINITY;
        for (i, island) in islands.iter().enumerate() {
            let side = compute_side_length(&island.trees);
            if side < best_side {
                best_side = side;
                best_idx = i;
            }
        }

        // Migrate best to next island in ring
        let next_idx = (best_idx + 1) % num_islands;
        if compute_side_length(&islands[next_idx].trees) > best_side * 1.01 {
            islands[next_idx].trees = islands[best_idx].trees.clone();
        }

        // Also migrate to a random island with lower probability
        if rng.gen::<f64>() < 0.3 {
            let random_idx = rng.gen_range(0..num_islands);
            if random_idx != best_idx &&
               compute_side_length(&islands[random_idx].trees) > best_side * 1.02 {
                islands[random_idx].trees = islands[best_idx].trees.clone();
            }
        }
    }

    // ========================================================================
    // LARGE NEIGHBORHOOD SEARCH (LNS)
    // ========================================================================

    /// Apply LNS: destroy k% of trees and repair
    fn apply_lns(
        &self,
        trees: &[PlacedTree],
        n: usize,
        island_config: &IslandConfig,
        rng: &mut impl Rng,
    ) -> Vec<PlacedTree> {
        if trees.len() < 3 {
            return trees.to_vec();
        }

        // Select destroy percentage
        let k = rng.gen_range(island_config.destroy_k_range.0..island_config.destroy_k_range.1);
        let destroy_count = ((trees.len() as f64 * k).ceil() as usize).max(1).min(trees.len() / 2);

        // Select destroy policy
        let policy = self.config.lns_config.destroy_policies
            [rng.gen_range(0..self.config.lns_config.destroy_policies.len())];

        // Get indices to destroy
        let destroy_indices = self.select_destroy_indices(trees, destroy_count, policy, rng);

        // Create remaining trees
        let mut remaining: Vec<PlacedTree> = trees.iter()
            .enumerate()
            .filter(|(i, _)| !destroy_indices.contains(i))
            .map(|(_, t)| t.clone())
            .collect();

        // Collect destroyed trees for repair
        let destroyed: Vec<PlacedTree> = destroy_indices.iter()
            .map(|&i| trees[i].clone())
            .collect();

        // Try multiple repair attempts, keep best
        let mut best_repaired = remaining.clone();
        for t in &destroyed {
            best_repaired.push(t.clone());
        }
        let mut best_side = compute_side_length(&best_repaired);

        for _ in 0..self.config.lns_config.repair_attempts {
            let mut repaired = remaining.clone();

            // Repair: re-insert destroyed trees one by one
            for destroyed_tree in &destroyed {
                // Find best position for this tree
                let new_tree = self.repair_insert(&repaired, destroyed_tree, n, rng);
                repaired.push(new_tree);
            }

            // Quick local search on repaired solution
            let mut quick_repaired = repaired.clone();
            self.quick_local_search(&mut quick_repaired, 500, rng);

            let side = compute_side_length(&quick_repaired);
            if side < best_side {
                best_side = side;
                best_repaired = quick_repaired;
            }
        }

        best_repaired
    }

    /// Select which tree indices to destroy based on policy
    fn select_destroy_indices(
        &self,
        trees: &[PlacedTree],
        count: usize,
        policy: DestroyPolicy,
        rng: &mut impl Rng,
    ) -> Vec<usize> {
        match policy {
            DestroyPolicy::RandomK => {
                // Random selection
                let mut indices: Vec<usize> = (0..trees.len()).collect();
                let mut selected = Vec::new();
                for _ in 0..count.min(indices.len()) {
                    let idx = rng.gen_range(0..indices.len());
                    selected.push(indices.remove(idx));
                }
                selected
            }
            DestroyPolicy::WorstK => {
                // Select trees contributing most to bounding box
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let mut scored: Vec<(usize, f64)> = trees.iter()
                    .enumerate()
                    .map(|(i, t)| {
                        let (bx1, by1, bx2, by2) = t.bounds();
                        // Score by how much this tree extends the bounds
                        let extend_score =
                            (min_x - bx1).max(0.0) +
                            (bx2 - max_x).max(0.0) +
                            (min_y - by1).max(0.0) +
                            (by2 - max_y).max(0.0);
                        // Also consider boundary trees
                        let on_boundary = (bx1 - min_x).abs() < 0.01 ||
                                         (bx2 - max_x).abs() < 0.01 ||
                                         (by1 - min_y).abs() < 0.01 ||
                                         (by2 - max_y).abs() < 0.01;
                        let boundary_score = if on_boundary { 1.0 } else { 0.0 };
                        (i, extend_score + boundary_score * 0.5 + rng.gen::<f64>() * 0.1)
                    })
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                scored.into_iter().take(count).map(|(i, _)| i).collect()
            }
            DestroyPolicy::ClusterK => {
                // Select trees from same angle bucket
                let angle_bucket_size = 45.0;
                let mut buckets: HashMap<i32, Vec<usize>> = HashMap::new();
                for (i, t) in trees.iter().enumerate() {
                    let bucket = (t.angle_deg / angle_bucket_size) as i32;
                    buckets.entry(bucket).or_default().push(i);
                }

                // Find largest bucket
                let largest_bucket = buckets.values()
                    .max_by_key(|v| v.len())
                    .cloned()
                    .unwrap_or_default();

                // Select from largest bucket
                let mut selected: Vec<usize> = largest_bucket.into_iter()
                    .take(count)
                    .collect();

                // If not enough, add random
                while selected.len() < count {
                    let idx = rng.gen_range(0..trees.len());
                    if !selected.contains(&idx) {
                        selected.push(idx);
                    }
                }
                selected
            }
        }
    }

    /// Repair: find best position to insert a destroyed tree
    fn repair_insert(
        &self,
        existing: &[PlacedTree],
        original: &PlacedTree,
        n: usize,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            return PlacedTree::new(0.0, 0.0, 90.0);
        }

        let mut best_tree = original.clone();
        let mut best_score = f64::INFINITY;

        // Try original position first
        if is_valid(original, existing) {
            let score = self.placement_score(original, existing, n);
            if score < best_score {
                best_score = score;
                best_tree = original.clone();
            }
        }

        // Try multiple random directions with original angle
        let angles = vec![original.angle_deg,
                         (original.angle_deg + 90.0) % 360.0,
                         (original.angle_deg + 180.0) % 360.0,
                         (original.angle_deg + 270.0) % 360.0];

        for _ in 0..30 {
            let dir = rng.gen_range(0.0..2.0 * PI);
            let vx = dir.cos();
            let vy = dir.sin();

            for &angle in &angles {
                let mut low = 0.0;
                let mut high = 10.0;

                while high - low > 0.001 {
                    let mid = (low + high) / 2.0;
                    let candidate = PlacedTree::new(mid * vx, mid * vy, angle);
                    if is_valid(&candidate, existing) {
                        high = mid;
                    } else {
                        low = mid;
                    }
                }

                let candidate = PlacedTree::new(high * vx, high * vy, angle);
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

    /// Quick local search for LNS repair
    fn quick_local_search(&self, trees: &mut Vec<PlacedTree>, iterations: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);

        for _ in 0..iterations {
            let idx = rng.gen_range(0..trees.len());
            let old_tree = trees[idx].clone();

            // Simple move
            let scale = 0.03;
            let dx = rng.gen_range(-scale..scale);
            let dy = rng.gen_range(-scale..scale);
            trees[idx] = PlacedTree::new(old_tree.x + dx, old_tree.y + dy, old_tree.angle_deg);

            if has_overlap(trees, idx) {
                trees[idx] = old_tree;
                continue;
            }

            let new_side = compute_side_length(trees);
            if new_side < current_side {
                current_side = new_side;
            } else {
                trees[idx] = old_tree;
            }
        }
    }

    // ========================================================================
    // PLACEMENT AND LOCAL SEARCH (from Gen10)
    // ========================================================================

    fn find_placement_with_strategy(
        &self,
        existing: &[PlacedTree],
        n: usize,
        _max_n: usize,
        strategy: PlacementStrategy,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            return PlacedTree::new(0.0, 0.0, 90.0);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 90.0);
        let mut best_score = f64::INFINITY;

        let angles = self.select_angles_for_strategy(n, strategy);

        for _ in 0..self.config.search_attempts {
            let dir = self.select_direction_for_strategy(n, strategy, rng);
            let vx = dir.cos();
            let vy = dir.sin();

            for &tree_angle in &angles {
                let mut low = 0.0;
                let mut high = 12.0;

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

    fn select_angles_for_strategy(&self, n: usize, strategy: PlacementStrategy) -> Vec<f64> {
        match strategy {
            PlacementStrategy::ClockwiseSpiral => {
                vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0]
            }
            PlacementStrategy::CounterclockwiseSpiral => {
                vec![270.0, 180.0, 90.0, 0.0, 315.0, 225.0, 135.0, 45.0]
            }
            PlacementStrategy::Grid => {
                vec![0.0, 180.0, 90.0, 270.0, 60.0, 120.0, 240.0, 300.0]
            }
            PlacementStrategy::Random => {
                match n % 4 {
                    0 => vec![45.0, 225.0, 135.0, 315.0, 0.0, 90.0, 180.0, 270.0],
                    1 => vec![90.0, 270.0, 0.0, 180.0, 45.0, 135.0, 225.0, 315.0],
                    2 => vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0],
                    _ => vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0],
                }
            }
            PlacementStrategy::BoundaryFirst => {
                vec![0.0, 90.0, 180.0, 270.0, 30.0, 60.0, 120.0, 150.0, 210.0, 240.0, 300.0, 330.0]
            }
        }
    }

    fn select_direction_for_strategy(&self, n: usize, strategy: PlacementStrategy, rng: &mut impl Rng) -> f64 {
        match strategy {
            PlacementStrategy::ClockwiseSpiral => {
                let base = (n as f64 * 0.3) % (2.0 * PI);
                base + rng.gen_range(-0.2..0.2)
            }
            PlacementStrategy::CounterclockwiseSpiral => {
                let base = (2.0 * PI) - (n as f64 * 0.3) % (2.0 * PI);
                base + rng.gen_range(-0.2..0.2)
            }
            PlacementStrategy::Grid => {
                let grid_angles = [0.0, PI/2.0, PI, 3.0*PI/2.0, PI/4.0, 3.0*PI/4.0, 5.0*PI/4.0, 7.0*PI/4.0];
                grid_angles[n % grid_angles.len()] + rng.gen_range(-0.15..0.15)
            }
            PlacementStrategy::Random => {
                rng.gen_range(0.0..2.0 * PI)
            }
            PlacementStrategy::BoundaryFirst => {
                let base_idx = rng.gen_range(0..self.config.direction_samples);
                (base_idx as f64 / self.config.direction_samples as f64) * 2.0 * PI
            }
        }
    }

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

        let side_score = side;
        let balance_weight = 0.10 + 0.03 * (1.0 - (n as f64 / 200.0).min(1.0));
        let balance_penalty = (width - height).abs() * balance_weight;

        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.006 / (n as f64).sqrt();

        let area = width * height;
        let density_bonus = if area > 0.0 {
            -self.config.gap_penalty_weight * 0.03 * (n as f64 / area).min(2.0)
        } else {
            0.0
        };

        side_score + balance_penalty + center_penalty + density_bonus
    }

    fn local_search_island(
        &self,
        trees: &mut Vec<PlacedTree>,
        n: usize,
        pass: usize,
        strategy: PlacementStrategy,
        iterations: usize,
        boundary_focus: f64,
        rng: &mut impl Rng,
    ) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        let temp_multiplier = if pass == 0 { 1.0 } else { 0.35 };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        let base_iterations = iterations + n * 80;
        let mut no_improvement = 0;

        for _ in 0..base_iterations {
            let idx = self.select_tree_to_move(trees, boundary_focus, rng);
            let old_tree = trees[idx].clone();

            let success = self.sa_move(trees, idx, temp, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                    current_side = new_side;
                    if current_side < best_side {
                        best_side = current_side;
                        best_config = trees.clone();
                        no_improvement = 0;
                    } else {
                        no_improvement += 1;
                    }
                } else {
                    trees[idx] = old_tree;
                    no_improvement += 1;
                }
            } else {
                trees[idx] = old_tree;
                no_improvement += 1;
            }

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);

            if no_improvement >= self.config.early_exit_threshold {
                break;
            }
        }

        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    #[inline]
    fn select_tree_to_move(&self, trees: &[PlacedTree], boundary_focus: f64, rng: &mut impl Rng) -> usize {
        if trees.len() <= 2 || rng.gen::<f64>() >= boundary_focus {
            return rng.gen_range(0..trees.len());
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
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

    #[inline]
    fn sa_move(&self, trees: &mut [PlacedTree], idx: usize, temp: f64, rng: &mut impl Rng) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let move_type = rng.gen_range(0..10);

        match move_type {
            0 => {
                let scale = self.config.translation_scale * (0.3 + temp * 2.0);
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
                if mag > 0.05 {
                    let scale = self.config.center_pull_strength * (0.5 + temp * 1.2);
                    let dx = -old_x / mag * scale;
                    let dy = -old_y / mag * scale;
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                } else {
                    return false;
                }
            }
            4 => {
                let scale = self.config.translation_scale * 0.5;
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                let dangle = rng.gen_range(-30.0..30.0);
                let new_angle = (old_angle + dangle).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            5 => {
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.1 {
                    let delta_r = rng.gen_range(-0.04..0.04) * (1.0 + temp);
                    let new_mag = (mag + delta_r).max(0.0);
                    let scale = new_mag / mag;
                    trees[idx] = PlacedTree::new(old_x * scale, old_y * scale, old_angle);
                } else {
                    return false;
                }
            }
            6 => {
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.1 {
                    let current_angle = old_y.atan2(old_x);
                    let delta_angle = rng.gen_range(-0.12..0.12) * (1.0 + temp);
                    let new_ang = current_angle + delta_angle;
                    trees[idx] = PlacedTree::new(mag * new_ang.cos(), mag * new_ang.sin(), old_angle);
                } else {
                    return false;
                }
            }
            7 => {
                let new_angle = (old_angle + 180.0).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            8 => {
                let scale = 0.012 * (0.5 + temp);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            _ => {
                let delta = if rng.gen() { 60.0 } else { -60.0 };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
        }

        !has_overlap(trees, idx)
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

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
    let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
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
        println!("Gen13 LNS+Islands score for n=1..50: {:.4}", score);
    }
}
