//! TSP 2-opt GPU Showcase
//!
//! Evolves priority functions for 2-opt move selection in the Traveling Salesman Problem.
//! Uses Metal GPU for parallel evaluation of O(n²) potential moves.

pub mod baselines;
pub mod evolved;

/// Trait for 2-opt move priority functions.
///
/// Given information about a potential 2-opt move (swapping two edges),
/// return a priority score. Higher priority moves are tried first.
///
/// A 2-opt move removes edges (a,b) and (c,d) and reconnects as (a,c) and (b,d),
/// reversing the tour segment between b and c.
pub trait TwoOptPriority {
    /// Score a potential 2-opt move.
    ///
    /// # Arguments
    /// * `delta` - Tour length change if move is made (negative = improvement)
    /// * `edge1_len` - Length of first edge being removed (a,b)
    /// * `edge2_len` - Length of second edge being removed (c,d)
    /// * `new_edge1_len` - Length of first new edge (a,c)
    /// * `new_edge2_len` - Length of second new edge (b,d)
    /// * `tour_len` - Current total tour length
    /// * `n` - Number of cities
    ///
    /// # Returns
    /// Priority score (higher = more likely to be selected)
    fn priority(
        &self,
        delta: f64,
        edge1_len: f64,
        edge2_len: f64,
        new_edge1_len: f64,
        new_edge2_len: f64,
        tour_len: f64,
        n: usize,
    ) -> f64;

    fn name(&self) -> &'static str;
}

/// Distance matrix for a TSP instance
pub struct DistanceMatrix {
    pub n: usize,
    pub distances: Vec<f64>,
}

impl DistanceMatrix {
    /// Create distance matrix from city coordinates
    pub fn from_coords(coords: &[(f64, f64)]) -> Self {
        let n = coords.len();
        let mut distances = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..n {
                let dx = coords[i].0 - coords[j].0;
                let dy = coords[i].1 - coords[j].1;
                distances[i * n + j] = (dx * dx + dy * dy).sqrt();
            }
        }

        Self { n, distances }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.distances[i * self.n + j]
    }
}

/// Calculate tour length
pub fn tour_length(tour: &[usize], dm: &DistanceMatrix) -> f64 {
    let n = tour.len();
    (0..n)
        .map(|i| dm.get(tour[i], tour[(i + 1) % n]))
        .sum()
}

/// Run 2-opt local search with a priority function
pub fn two_opt_search<P: TwoOptPriority>(
    tour: &mut Vec<usize>,
    dm: &DistanceMatrix,
    priority: &P,
    max_iterations: usize,
) -> f64 {
    let n = tour.len();
    let mut current_len = tour_length(tour, dm);
    let mut improved = true;
    let mut iterations = 0;

    while improved && iterations < max_iterations {
        improved = false;
        iterations += 1;

        // Collect all potential 2-opt moves with their priorities
        let mut moves: Vec<(usize, usize, f64, f64)> = Vec::new(); // (i, j, delta, priority)

        for i in 0..n - 1 {
            for j in i + 2..n {
                if i == 0 && j == n - 1 {
                    continue; // Skip: would just reverse entire tour
                }

                // Current edges: (tour[i], tour[i+1]) and (tour[j], tour[(j+1) % n])
                let a = tour[i];
                let b = tour[i + 1];
                let c = tour[j];
                let d = tour[(j + 1) % n];

                let edge1_len = dm.get(a, b);
                let edge2_len = dm.get(c, d);
                let new_edge1_len = dm.get(a, c);
                let new_edge2_len = dm.get(b, d);

                let delta = new_edge1_len + new_edge2_len - edge1_len - edge2_len;

                let prio = priority.priority(
                    delta,
                    edge1_len,
                    edge2_len,
                    new_edge1_len,
                    new_edge2_len,
                    current_len,
                    n,
                );

                moves.push((i, j, delta, prio));
            }
        }

        // Sort by priority (descending)
        moves.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        // Try moves in priority order, apply first improving move
        for (i, j, delta, _) in moves {
            if delta < -1e-10 {
                // Apply the 2-opt move: reverse segment from i+1 to j
                tour[i + 1..=j].reverse();
                current_len += delta;
                improved = true;
                break;
            }
        }
    }

    current_len
}

/// Generate nearest neighbor tour
pub fn nearest_neighbor_tour(dm: &DistanceMatrix) -> Vec<usize> {
    let n = dm.n;
    let mut tour = Vec::with_capacity(n);
    let mut visited = vec![false; n];

    // Start from city 0
    let mut current = 0;
    tour.push(current);
    visited[current] = true;

    for _ in 1..n {
        let mut best_next = 0;
        let mut best_dist = f64::MAX;

        for j in 0..n {
            if !visited[j] && dm.get(current, j) < best_dist {
                best_dist = dm.get(current, j);
                best_next = j;
            }
        }

        tour.push(best_next);
        visited[best_next] = true;
        current = best_next;
    }

    tour
}

/// Benchmark result for a single instance
#[derive(Clone)]
pub struct InstanceResult {
    pub name: String,
    pub optimal: f64,
    pub found: f64,
    pub gap_percent: f64,
    pub iterations: usize,
}

/// Benchmark result for an algorithm
#[derive(Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub instances: Vec<InstanceResult>,
    pub avg_gap_percent: f64,
    pub total_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baselines::GreedyDelta;

    #[test]
    fn test_tour_length() {
        // Simple 4-city square
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let dm = DistanceMatrix::from_coords(&coords);
        let tour = vec![0, 1, 2, 3];
        let len = tour_length(&tour, &dm);
        assert!((len - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_neighbor() {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
        let dm = DistanceMatrix::from_coords(&coords);
        let tour = nearest_neighbor_tour(&dm);
        assert_eq!(tour.len(), 4);
    }

    #[test]
    fn test_two_opt() {
        // 6-city case where 2-opt should improve
        let coords = vec![
            (0.0, 0.0),
            (2.0, 0.0),
            (1.0, 1.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (5.0, 0.0),
        ];
        let dm = DistanceMatrix::from_coords(&coords);
        let mut tour = nearest_neighbor_tour(&dm);
        let initial = tour_length(&tour, &dm);

        let greedy = GreedyDelta;
        let final_len = two_opt_search(&mut tour, &dm, &greedy, 1000);

        assert!(final_len <= initial);
    }
}
