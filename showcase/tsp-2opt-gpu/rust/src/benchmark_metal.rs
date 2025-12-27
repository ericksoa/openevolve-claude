//! Metal GPU-Accelerated TSP 2-opt Benchmark
//!
//! Uses Apple Metal to evaluate all O(n^2) potential 2-opt moves in parallel.
//! This dramatically speeds up the neighborhood search.

use metal::*;
use objc::rc::autoreleasepool;
use std::time::Instant;
use tsp_2opt::{nearest_neighbor_tour, tour_length, DistanceMatrix, InstanceResult};

// ============================================================================
// TSPLIB Benchmark Instances (Same as CPU benchmark)
// ============================================================================

const EIL51_COORDS: [(f64, f64); 51] = [
    (37.0, 52.0), (49.0, 49.0), (52.0, 64.0), (20.0, 26.0), (40.0, 30.0),
    (21.0, 47.0), (17.0, 63.0), (31.0, 62.0), (52.0, 33.0), (51.0, 21.0),
    (42.0, 41.0), (31.0, 32.0), (5.0, 25.0), (12.0, 42.0), (36.0, 16.0),
    (52.0, 41.0), (27.0, 23.0), (17.0, 33.0), (13.0, 13.0), (57.0, 58.0),
    (62.0, 42.0), (42.0, 57.0), (16.0, 57.0), (8.0, 52.0), (7.0, 38.0),
    (27.0, 68.0), (30.0, 48.0), (43.0, 67.0), (58.0, 48.0), (58.0, 27.0),
    (37.0, 69.0), (38.0, 46.0), (46.0, 10.0), (61.0, 33.0), (62.0, 63.0),
    (63.0, 69.0), (32.0, 22.0), (45.0, 35.0), (59.0, 15.0), (5.0, 6.0),
    (10.0, 17.0), (21.0, 10.0), (5.0, 64.0), (30.0, 15.0), (39.0, 10.0),
    (32.0, 39.0), (25.0, 32.0), (25.0, 55.0), (48.0, 28.0), (56.0, 37.0),
    (30.0, 40.0),
];

const BERLIN52_COORDS: [(f64, f64); 52] = [
    (565.0, 575.0), (25.0, 185.0), (345.0, 750.0), (945.0, 685.0), (845.0, 655.0),
    (880.0, 660.0), (25.0, 230.0), (525.0, 1000.0), (580.0, 1175.0), (650.0, 1130.0),
    (1605.0, 620.0), (1220.0, 580.0), (1465.0, 200.0), (1530.0, 5.0), (845.0, 680.0),
    (725.0, 370.0), (145.0, 665.0), (415.0, 635.0), (510.0, 875.0), (560.0, 365.0),
    (300.0, 465.0), (520.0, 585.0), (480.0, 415.0), (835.0, 625.0), (975.0, 580.0),
    (1215.0, 245.0), (1320.0, 315.0), (1250.0, 400.0), (660.0, 180.0), (410.0, 250.0),
    (420.0, 555.0), (575.0, 665.0), (1150.0, 1160.0), (700.0, 580.0), (685.0, 595.0),
    (685.0, 610.0), (770.0, 610.0), (795.0, 645.0), (720.0, 635.0), (760.0, 650.0),
    (475.0, 960.0), (95.0, 260.0), (875.0, 920.0), (700.0, 500.0), (555.0, 815.0),
    (830.0, 485.0), (1170.0, 65.0), (830.0, 610.0), (605.0, 625.0), (595.0, 360.0),
    (1340.0, 725.0), (1740.0, 245.0),
];

const KROA100_COORDS: [(f64, f64); 100] = [
    (1380.0, 939.0), (2848.0, 96.0), (3510.0, 1671.0), (457.0, 334.0), (3888.0, 666.0),
    (984.0, 965.0), (2721.0, 1482.0), (1286.0, 525.0), (2716.0, 1432.0), (738.0, 1325.0),
    (1251.0, 1832.0), (2728.0, 1698.0), (3815.0, 169.0), (3683.0, 1533.0), (1247.0, 1945.0),
    (123.0, 862.0), (1234.0, 1946.0), (252.0, 1240.0), (611.0, 673.0), (2576.0, 1676.0),
    (928.0, 1700.0), (53.0, 857.0), (1807.0, 1711.0), (274.0, 1420.0), (2574.0, 946.0),
    (178.0, 24.0), (2678.0, 1825.0), (1795.0, 962.0), (3384.0, 1498.0), (3520.0, 1079.0),
    (1256.0, 61.0), (1424.0, 1728.0), (3913.0, 192.0), (3085.0, 1528.0), (2573.0, 1969.0),
    (463.0, 1670.0), (3875.0, 598.0), (298.0, 1513.0), (3479.0, 821.0), (2542.0, 236.0),
    (3955.0, 1743.0), (1323.0, 280.0), (3447.0, 1830.0), (2936.0, 337.0), (1621.0, 1830.0),
    (3373.0, 1646.0), (1393.0, 1368.0), (3874.0, 1318.0), (938.0, 955.0), (3022.0, 474.0),
    (2482.0, 1183.0), (3854.0, 923.0), (376.0, 825.0), (2519.0, 135.0), (2945.0, 1622.0),
    (953.0, 268.0), (2628.0, 1479.0), (2097.0, 981.0), (890.0, 1846.0), (2139.0, 1806.0),
    (2421.0, 1007.0), (2290.0, 1810.0), (1115.0, 1052.0), (2588.0, 302.0), (327.0, 265.0),
    (241.0, 341.0), (1917.0, 687.0), (2991.0, 792.0), (2573.0, 599.0), (19.0, 674.0),
    (3911.0, 1673.0), (872.0, 1559.0), (2863.0, 558.0), (929.0, 1766.0), (839.0, 620.0),
    (3893.0, 102.0), (2178.0, 1619.0), (3822.0, 899.0), (378.0, 1048.0), (1178.0, 100.0),
    (2599.0, 901.0), (3416.0, 143.0), (2961.0, 1605.0), (611.0, 1384.0), (3113.0, 885.0),
    (2597.0, 1830.0), (2586.0, 1286.0), (161.0, 906.0), (1429.0, 134.0), (742.0, 1025.0),
    (1625.0, 1651.0), (1187.0, 706.0), (1787.0, 1009.0), (22.0, 987.0), (3640.0, 43.0),
    (3756.0, 882.0), (776.0, 392.0), (1724.0, 1642.0), (198.0, 1810.0), (3950.0, 1558.0),
];

/// Metal shader source for parallel 2-opt delta computation
const METAL_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Compute delta for a single 2-opt move
// Inputs: tour indices, distance matrix, move position (i, j)
// Output: delta (tour length change) for swapping edges at positions i and j

kernel void compute_deltas(
    device const uint* tour [[buffer(0)]],
    device const float* distances [[buffer(1)]],
    device float* deltas [[buffer(2)]],
    device const uint& n [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;

    // Only compute for valid 2-opt moves: i < j-1, not (i==0 && j==n-1)
    if (i >= n - 1 || j <= i + 1 || j >= n || (i == 0 && j == n - 1)) {
        deltas[i * n + j] = INFINITY;
        return;
    }

    // Get cities at positions
    uint a = tour[i];
    uint b = tour[i + 1];
    uint c = tour[j];
    uint d = tour[(j + 1) % n];

    // Current edges: (a,b) and (c,d)
    // New edges: (a,c) and (b,d)
    float edge1 = distances[a * n + b];
    float edge2 = distances[c * n + d];
    float new_edge1 = distances[a * n + c];
    float new_edge2 = distances[b * n + d];

    deltas[i * n + j] = new_edge1 + new_edge2 - edge1 - edge2;
}

// Find best move (minimum delta)
kernel void find_best_move(
    device const float* deltas [[buffer(0)]],
    device atomic_int* best_i [[buffer(1)]],
    device atomic_int* best_j [[buffer(2)]],
    device atomic_float* best_delta [[buffer(3)]],
    device const uint& n [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;

    if (i >= n || j >= n) return;

    float d = deltas[i * n + j];

    // Atomic min operation (simplified - real implementation needs atomic compare-exchange)
    // For now, we'll do reduction on CPU
}
"#;

/// Metal-accelerated 2-opt solver
struct MetalTwoOpt {
    device: Device,
    command_queue: CommandQueue,
    pipeline: ComputePipelineState,
}

impl MetalTwoOpt {
    fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let command_queue = device.new_command_queue();

        let library = device
            .new_library_with_source(METAL_SHADER, &CompileOptions::new())
            .ok()?;

        let kernel = library.get_function("compute_deltas", None).ok()?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .ok()?;

        Some(Self {
            device,
            command_queue,
            pipeline,
        })
    }

    fn solve(&self, tour: &mut Vec<usize>, dm: &DistanceMatrix, max_iterations: usize) -> f64 {
        let n = tour.len();

        // Create Metal buffers
        let tour_u32: Vec<u32> = tour.iter().map(|&x| x as u32).collect();
        let distances_f32: Vec<f32> = dm.distances.iter().map(|&x| x as f32).collect();

        let tour_buffer = self.device.new_buffer_with_data(
            tour_u32.as_ptr() as *const _,
            (n * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let dist_buffer = self.device.new_buffer_with_data(
            distances_f32.as_ptr() as *const _,
            (n * n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let delta_buffer = self.device.new_buffer(
            (n * n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let n_u32 = n as u32;
        let n_buffer = self.device.new_buffer_with_data(
            &n_u32 as *const _ as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let mut current_len = tour_length(tour, dm);
        let mut improved = true;
        let mut iterations = 0;

        while improved && iterations < max_iterations {
            improved = false;
            iterations += 1;

            // Update tour buffer
            let tour_u32: Vec<u32> = tour.iter().map(|&x| x as u32).collect();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tour_u32.as_ptr(),
                    tour_buffer.contents() as *mut u32,
                    n,
                );
            }

            autoreleasepool(|| {
                let command_buffer = self.command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                encoder.set_compute_pipeline_state(&self.pipeline);
                encoder.set_buffer(0, Some(&tour_buffer), 0);
                encoder.set_buffer(1, Some(&dist_buffer), 0);
                encoder.set_buffer(2, Some(&delta_buffer), 0);
                encoder.set_buffer(3, Some(&n_buffer), 0);

                let threads_per_grid = MTLSize::new(n as u64, n as u64, 1);
                let threads_per_group = MTLSize::new(16, 16, 1);

                encoder.dispatch_threads(threads_per_grid, threads_per_group);
                encoder.end_encoding();

                command_buffer.commit();
                command_buffer.wait_until_completed();
            });

            // Read deltas back and find best move
            let deltas: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    delta_buffer.contents() as *const f32,
                    n * n,
                )
            };

            let mut best_i = 0;
            let mut best_j = 0;
            let mut best_delta = 0.0f32;

            for i in 0..n - 1 {
                for j in i + 2..n {
                    if i == 0 && j == n - 1 {
                        continue;
                    }
                    let d = deltas[i * n + j];
                    if d < best_delta {
                        best_delta = d;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            if best_delta < -1e-6 {
                tour[best_i + 1..=best_j].reverse();
                current_len += best_delta as f64;
                improved = true;
            }
        }

        current_len
    }
}

/// CPU fallback 2-opt (same as original)
fn cpu_two_opt(tour: &mut Vec<usize>, dm: &DistanceMatrix, max_iterations: usize) -> f64 {
    let n = tour.len();
    let mut current_len = tour_length(tour, dm);
    let mut improved = true;
    let mut iterations = 0;

    while improved && iterations < max_iterations {
        improved = false;
        iterations += 1;

        let mut best_i = 0;
        let mut best_j = 0;
        let mut best_delta = 0.0;

        for i in 0..n - 1 {
            for j in i + 2..n {
                if i == 0 && j == n - 1 {
                    continue;
                }

                let a = tour[i];
                let b = tour[i + 1];
                let c = tour[j];
                let d = tour[(j + 1) % n];

                let delta = dm.get(a, c) + dm.get(b, d) - dm.get(a, b) - dm.get(c, d);

                if delta < best_delta {
                    best_delta = delta;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if best_delta < -1e-10 {
            tour[best_i + 1..=best_j].reverse();
            current_len += best_delta;
            improved = true;
        }
    }

    current_len
}

struct TspInstance {
    name: &'static str,
    optimal: f64,
    coords: &'static [(f64, f64)],
}

fn get_instances() -> Vec<TspInstance> {
    vec![
        TspInstance { name: "eil51", optimal: 426.0, coords: &EIL51_COORDS },
        TspInstance { name: "berlin52", optimal: 7542.0, coords: &BERLIN52_COORDS },
        TspInstance { name: "kroA100", optimal: 21282.0, coords: &KROA100_COORDS },
    ]
}

fn main() {
    let instances = get_instances();
    let max_iterations = 10000;

    // Try to initialize Metal
    let metal = MetalTwoOpt::new();
    let use_metal = metal.is_some();

    if use_metal {
        eprintln!("Using Metal GPU acceleration");
    } else {
        eprintln!("Metal not available, using CPU fallback");
    }

    // Benchmark CPU
    let cpu_start = Instant::now();
    let mut cpu_results: Vec<InstanceResult> = Vec::new();

    for inst in &instances {
        let dm = DistanceMatrix::from_coords(inst.coords);
        let mut tour = nearest_neighbor_tour(&dm);
        let final_len = cpu_two_opt(&mut tour, &dm, max_iterations);
        let gap = (final_len - inst.optimal) / inst.optimal * 100.0;

        cpu_results.push(InstanceResult {
            name: inst.name.to_string(),
            optimal: inst.optimal,
            found: final_len,
            gap_percent: gap,
            iterations: max_iterations,
        });
    }

    let cpu_time = cpu_start.elapsed().as_secs_f64() * 1000.0;
    let cpu_avg_gap = cpu_results.iter().map(|r| r.gap_percent).sum::<f64>() / cpu_results.len() as f64;

    // Benchmark Metal (if available)
    let (gpu_results, gpu_time, gpu_avg_gap) = if let Some(ref metal_solver) = metal {
        let gpu_start = Instant::now();
        let mut results: Vec<InstanceResult> = Vec::new();

        for inst in &instances {
            let dm = DistanceMatrix::from_coords(inst.coords);
            let mut tour = nearest_neighbor_tour(&dm);
            let final_len = metal_solver.solve(&mut tour, &dm, max_iterations);
            let gap = (final_len - inst.optimal) / inst.optimal * 100.0;

            results.push(InstanceResult {
                name: inst.name.to_string(),
                optimal: inst.optimal,
                found: final_len,
                gap_percent: gap,
                iterations: max_iterations,
            });
        }

        let time = gpu_start.elapsed().as_secs_f64() * 1000.0;
        let avg = results.iter().map(|r| r.gap_percent).sum::<f64>() / results.len() as f64;
        (results, time, avg)
    } else {
        (cpu_results.clone(), cpu_time, cpu_avg_gap)
    };

    // Output JSON
    println!("{{");
    println!("  \"benchmark\": \"tsp-2opt-metal\",");
    println!("  \"metal_available\": {},", use_metal);
    println!("  \"instances\": [\"eil51\", \"berlin52\", \"kroA100\"],");
    println!("  \"results\": [");

    // CPU results
    println!("    {{");
    println!("      \"name\": \"cpu_2opt\",");
    println!("      \"avg_gap_percent\": {:.4},", cpu_avg_gap);
    println!("      \"time_ms\": {:.2},", cpu_time);
    println!("      \"instances\": [");
    for (i, r) in cpu_results.iter().enumerate() {
        let comma = if i < cpu_results.len() - 1 { "," } else { "" };
        println!("        {{\"name\": \"{}\", \"found\": {:.0}, \"gap_percent\": {:.4}}}{}", r.name, r.found, r.gap_percent, comma);
    }
    println!("      ]");
    println!("    }},");

    // GPU results
    println!("    {{");
    println!("      \"name\": \"metal_2opt\",");
    println!("      \"avg_gap_percent\": {:.4},", gpu_avg_gap);
    println!("      \"time_ms\": {:.2},", gpu_time);
    println!("      \"speedup\": {:.2},", cpu_time / gpu_time);
    println!("      \"instances\": [");
    for (i, r) in gpu_results.iter().enumerate() {
        let comma = if i < gpu_results.len() - 1 { "," } else { "" };
        println!("        {{\"name\": \"{}\", \"found\": {:.0}, \"gap_percent\": {:.4}}}{}", r.name, r.found, r.gap_percent, comma);
    }
    println!("      ]");
    println!("    }}");

    println!("  ]");
    println!("}}");
}
