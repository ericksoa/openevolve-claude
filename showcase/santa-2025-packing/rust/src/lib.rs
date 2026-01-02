//! Santa 2025 - Christmas Tree Packing
//!
//! Pack Christmas tree-shaped polygons into the smallest square box.

pub mod baselines;
pub mod evolved;
pub mod incremental;
pub mod multistart;
pub mod simulated_annealing;

use std::f64::consts::PI;

/// The Christmas tree polygon vertices (centered at origin, unrotated)
/// 15 vertices defining the tree shape
pub const TREE_VERTICES: [(f64, f64); 15] = [
    // Tip
    (0.0, 0.8),
    // Right side - Top Tier
    (0.125, 0.5),   // top_w/2
    (0.0625, 0.5),  // top_w/4
    // Right side - Middle Tier
    (0.2, 0.25),    // mid_w/2
    (0.1, 0.25),    // mid_w/4
    // Right side - Bottom Tier
    (0.35, 0.0),    // base_w/2
    // Right Trunk
    (0.075, 0.0),   // trunk_w/2
    (0.075, -0.2),  // trunk_w/2, trunk_bottom
    // Left Trunk
    (-0.075, -0.2), // -trunk_w/2, trunk_bottom
    (-0.075, 0.0),  // -trunk_w/2
    // Left side - Bottom Tier
    (-0.35, 0.0),   // -base_w/2
    // Left side - Middle Tier
    (-0.1, 0.25),   // -mid_w/4
    (-0.2, 0.25),   // -mid_w/2
    // Left side - Top Tier
    (-0.0625, 0.5), // -top_w/4
    (-0.125, 0.5),  // -top_w/2
];

/// Tree dimensions
pub const TREE_HEIGHT: f64 = 1.0;  // from -0.2 to 0.8
pub const TREE_WIDTH: f64 = 0.7;   // base width

/// A placed tree with position and rotation
#[derive(Clone, Debug)]
pub struct PlacedTree {
    pub x: f64,
    pub y: f64,
    pub angle_deg: f64,
    vertices: Vec<(f64, f64)>,
}

impl PlacedTree {
    pub fn new(x: f64, y: f64, angle_deg: f64) -> Self {
        let angle_rad = angle_deg * PI / 180.0;
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let vertices: Vec<(f64, f64)> = TREE_VERTICES
            .iter()
            .map(|&(vx, vy)| {
                // Rotate then translate
                let rx = vx * cos_a - vy * sin_a;
                let ry = vx * sin_a + vy * cos_a;
                (rx + x, ry + y)
            })
            .collect();

        Self { x, y, angle_deg, vertices }
    }

    pub fn vertices(&self) -> &[(f64, f64)] {
        &self.vertices
    }

    /// Check if this tree overlaps with another (excluding touching)
    pub fn overlaps(&self, other: &PlacedTree) -> bool {
        polygons_overlap(&self.vertices, &other.vertices)
    }

    /// Get bounding box (min_x, min_y, max_x, max_y)
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for &(x, y) in &self.vertices {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        (min_x, min_y, max_x, max_y)
    }
}

/// Check if two convex/concave polygons overlap
/// Uses separating axis theorem for edge normals
/// Added epsilon buffer to be conservative (Kaggle validation is strict)
fn polygons_overlap(poly1: &[(f64, f64)], poly2: &[(f64, f64)]) -> bool {
    const EPSILON: f64 = 1e-9;  // Small buffer for numerical precision

    // First check if bounding boxes overlap (with epsilon)
    let (min1x, min1y, max1x, max1y) = polygon_bounds(poly1);
    let (min2x, min2y, max2x, max2y) = polygon_bounds(poly2);

    if max1x + EPSILON < min2x || max2x + EPSILON < min1x ||
       max1y + EPSILON < min2y || max2y + EPSILON < min1y {
        return false;
    }

    // Use polygon intersection check
    // For non-convex polygons, we need to check edge intersections
    // and point containment

    // Check if any edges intersect
    for i in 0..poly1.len() {
        let j = (i + 1) % poly1.len();
        let (a1, a2) = (poly1[i], poly1[j]);

        for k in 0..poly2.len() {
            let l = (k + 1) % poly2.len();
            let (b1, b2) = (poly2[k], poly2[l]);

            if segments_intersect_proper(a1, a2, b1, b2) {
                return true;
            }
        }
    }

    // Check if any vertex of poly1 is strictly inside poly2
    for &p in poly1 {
        if point_strictly_inside_polygon(p, poly2) {
            return true;
        }
    }

    // Check if any vertex of poly2 is strictly inside poly1
    for &p in poly2 {
        if point_strictly_inside_polygon(p, poly1) {
            return true;
        }
    }

    false
}

fn polygon_bounds(poly: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for &(x, y) in poly {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    (min_x, min_y, max_x, max_y)
}

/// Check if two segments intersect (including touching/collinear cases)
/// More conservative than proper intersection to match Kaggle validation
fn segments_intersect_proper(a1: (f64, f64), a2: (f64, f64), b1: (f64, f64), b2: (f64, f64)) -> bool {
    const EPSILON: f64 = 1e-9;

    let d1 = cross_product_sign(b1, b2, a1);
    let d2 = cross_product_sign(b1, b2, a2);
    let d3 = cross_product_sign(a1, a2, b1);
    let d4 = cross_product_sign(a1, a2, b2);

    // Proper intersection: opposite signs
    if ((d1 > EPSILON && d2 < -EPSILON) || (d1 < -EPSILON && d2 > EPSILON)) &&
       ((d3 > EPSILON && d4 < -EPSILON) || (d3 < -EPSILON && d4 > EPSILON)) {
        return true;
    }

    // Also check for collinear/touching cases (conservative)
    // If signs are very close to zero, segments might be touching
    if d1.abs() < EPSILON || d2.abs() < EPSILON || d3.abs() < EPSILON || d4.abs() < EPSILON {
        // Check if segments actually overlap using parametric intersection
        return segments_touch_or_overlap(a1, a2, b1, b2);
    }

    false
}

/// Check if segments touch or overlap (for collinear/near-collinear cases)
fn segments_touch_or_overlap(a1: (f64, f64), a2: (f64, f64), b1: (f64, f64), b2: (f64, f64)) -> bool {
    const EPSILON: f64 = 1e-9;

    // Check if any endpoint of one segment is very close to the other segment
    if point_near_segment(a1, b1, b2, EPSILON) ||
       point_near_segment(a2, b1, b2, EPSILON) ||
       point_near_segment(b1, a1, a2, EPSILON) ||
       point_near_segment(b2, a1, a2, EPSILON) {
        return true;
    }
    false
}

/// Check if point p is within epsilon of segment (s1, s2)
fn point_near_segment(p: (f64, f64), s1: (f64, f64), s2: (f64, f64), epsilon: f64) -> bool {
    let dx = s2.0 - s1.0;
    let dy = s2.1 - s1.1;
    let len_sq = dx * dx + dy * dy;

    if len_sq < epsilon * epsilon {
        // Degenerate segment, check distance to point
        let d = ((p.0 - s1.0).powi(2) + (p.1 - s1.1).powi(2)).sqrt();
        return d < epsilon;
    }

    // Project p onto line, get parameter t
    let t = ((p.0 - s1.0) * dx + (p.1 - s1.1) * dy) / len_sq;

    // Check if projection is within segment
    if t < -epsilon || t > 1.0 + epsilon {
        return false;
    }

    // Calculate distance from p to closest point on segment
    let t_clamped = t.clamp(0.0, 1.0);
    let closest_x = s1.0 + t_clamped * dx;
    let closest_y = s1.1 + t_clamped * dy;
    let dist = ((p.0 - closest_x).powi(2) + (p.1 - closest_y).powi(2)).sqrt();

    dist < epsilon
}

fn cross_product_sign(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// Check if point is strictly inside polygon (not on boundary)
fn point_strictly_inside_polygon(p: (f64, f64), poly: &[(f64, f64)]) -> bool {
    let mut winding = 0i32;
    let n = poly.len();

    for i in 0..n {
        let j = (i + 1) % n;
        let (x1, y1) = poly[i];
        let (x2, y2) = poly[j];

        if y1 <= p.1 {
            if y2 > p.1 {
                // Upward crossing
                let cross = (x2 - x1) * (p.1 - y1) - (p.0 - x1) * (y2 - y1);
                if cross > 1e-10 {
                    winding += 1;
                }
            }
        } else if y2 <= p.1 {
            // Downward crossing
            let cross = (x2 - x1) * (p.1 - y1) - (p.0 - x1) * (y2 - y1);
            if cross < -1e-10 {
                winding -= 1;
            }
        }
    }

    winding != 0
}

/// A packing solution for n trees
#[derive(Clone)]
pub struct Packing {
    pub trees: Vec<PlacedTree>,
}

impl Packing {
    pub fn new() -> Self {
        Self { trees: Vec::new() }
    }

    /// Add a tree if it doesn't overlap with existing trees
    pub fn try_add(&mut self, tree: PlacedTree) -> bool {
        for existing in &self.trees {
            if tree.overlaps(existing) {
                return false;
            }
        }
        self.trees.push(tree);
        true
    }

    /// Check if a tree can be placed without overlap
    pub fn can_place(&self, tree: &PlacedTree) -> bool {
        for existing in &self.trees {
            if tree.overlaps(existing) {
                return false;
            }
        }
        true
    }

    /// Get the side length of the smallest square containing all trees
    pub fn side_length(&self) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for tree in &self.trees {
            let (bmin_x, bmin_y, bmax_x, bmax_y) = tree.bounds();
            min_x = min_x.min(bmin_x);
            min_y = min_y.min(bmin_y);
            max_x = max_x.max(bmax_x);
            max_y = max_y.max(bmax_y);
        }

        let width = max_x - min_x;
        let height = max_y - min_y;
        width.max(height)
    }

    /// Check if any trees overlap (validation)
    pub fn has_overlaps(&self) -> bool {
        for i in 0..self.trees.len() {
            for j in (i + 1)..self.trees.len() {
                if self.trees[i].overlaps(&self.trees[j]) {
                    return true;
                }
            }
        }
        false
    }
}

impl Default for Packing {
    fn default() -> Self {
        Self::new()
    }
}

impl Packing {
    /// Generate an SVG visualization of the packing
    pub fn to_svg(&self, width: usize, height: usize) -> String {
        if self.trees.is_empty() {
            return format!(
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n\
                  <rect width=\"100%\" height=\"100%\" fill=\"#f0f0f0\"/>\n\
                  <text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#666\">No trees</text>\n\
                </svg>",
                width, height, width, height
            );
        }

        // Calculate bounds
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for tree in &self.trees {
            let (bmin_x, bmin_y, bmax_x, bmax_y) = tree.bounds();
            min_x = min_x.min(bmin_x);
            min_y = min_y.min(bmin_y);
            max_x = max_x.max(bmax_x);
            max_y = max_y.max(bmax_y);
        }

        let data_width = max_x - min_x;
        let data_height = max_y - min_y;
        let side = data_width.max(data_height);

        // Add margin
        let margin = side * 0.05;
        let view_min_x = min_x - margin;
        let view_min_y = min_y - margin;
        let view_size = side + 2.0 * margin;

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"{} {} {} {}\">\n\
              <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"#f8f8f8\" stroke=\"#ccc\" stroke-width=\"0.01\"/>\n\
              <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"#2196F3\" stroke-width=\"0.02\" stroke-dasharray=\"0.05,0.05\"/>\n",
            width, height,
            view_min_x, -view_min_y - view_size, view_size, view_size,
            view_min_x, -view_min_y - view_size, view_size, view_size,
            min_x, -max_y, side, side
        );

        // Draw trees with gradient colors
        for (i, tree) in self.trees.iter().enumerate() {
            let hue = (i as f64 / self.trees.len() as f64 * 120.0) as u32; // Green gradient
            let color = format!("hsl({}, 70%, 40%)", hue);
            let stroke_color = format!("hsl({}, 70%, 30%)", hue);

            let points: String = tree.vertices()
                .iter()
                .map(|(x, y)| format!("{:.4},{:.4}", x, -y))
                .collect::<Vec<_>>()
                .join(" ");

            svg.push_str(&format!(
                "  <polygon points=\"{}\" fill=\"{}\" stroke=\"{}\" stroke-width=\"0.005\" opacity=\"0.8\"/>\n",
                points, color, stroke_color
            ));
        }

        // Add info text
        svg.push_str(&format!(
            "  <text x=\"{}\" y=\"{}\" font-size=\"0.15\" fill=\"#333\">n={}, side={:.4}</text>\n</svg>",
            view_min_x + 0.05, -view_min_y - view_size + 0.2,
            self.trees.len(), self.side_length()
        ));

        svg
    }
}

/// Trait for packing algorithms
pub trait PackingAlgorithm: Send + Sync {
    /// Pack n trees and return the packing
    fn pack(&self, n: usize) -> Packing;

    /// Name of the algorithm
    fn name(&self) -> &'static str;
}

/// Calculate the competition score for packings of 1 to max_n trees
pub fn calculate_score(packings: &[Packing]) -> f64 {
    let mut score = 0.0;
    for (i, packing) in packings.iter().enumerate() {
        let n = i + 1;
        let side = packing.side_length();
        score += (side * side) / (n as f64);
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_creation() {
        let tree = PlacedTree::new(0.0, 0.0, 0.0);
        assert_eq!(tree.vertices().len(), 15);

        // Check tip is at (0, 0.8)
        let (tip_x, tip_y) = tree.vertices()[0];
        assert!((tip_x - 0.0).abs() < 1e-10);
        assert!((tip_y - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_tree_rotation() {
        let tree = PlacedTree::new(0.0, 0.0, 90.0);
        // After 90 degree rotation, tip should be at (-0.8, 0)
        let (tip_x, tip_y) = tree.vertices()[0];
        assert!((tip_x - (-0.8)).abs() < 1e-10);
        assert!(tip_y.abs() < 1e-10);
    }

    #[test]
    fn test_no_self_overlap() {
        let tree1 = PlacedTree::new(0.0, 0.0, 0.0);
        let tree2 = PlacedTree::new(0.0, 0.0, 0.0);
        // Same position should overlap
        assert!(tree1.overlaps(&tree2));
    }

    #[test]
    fn test_separate_trees() {
        let tree1 = PlacedTree::new(0.0, 0.0, 0.0);
        let tree2 = PlacedTree::new(2.0, 0.0, 0.0);
        // Far apart should not overlap
        assert!(!tree1.overlaps(&tree2));
    }
}
