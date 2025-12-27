//! TSP 2-opt Benchmark
//!
//! Benchmarks various 2-opt priority functions on TSPLIB instances.
//! Outputs JSON for fitness evaluation.

use std::time::Instant;
use tsp_2opt::{
    baselines::{Balanced, BestImprovement, EdgeRatio, GreedyDelta, LKInspired, LongEdgeRemoval, RelativeGain},
    evolved::Evolved,
    nearest_neighbor_tour, tour_length, two_opt_search, BenchmarkResult, DistanceMatrix,
    InstanceResult, TwoOptPriority,
};

/// TSPLIB instance data
struct TspInstance {
    name: &'static str,
    optimal: f64,
    coords: &'static [(f64, f64)],
}

// ============================================================================
// TSPLIB Benchmark Instances (Embedded Data)
// ============================================================================

/// eil51 - 51 cities, optimal = 426
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

/// berlin52 - 52 cities, optimal = 7542
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

/// kroA100 - 100 cities, optimal = 21282
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

/// All benchmark instances
fn get_instances() -> Vec<TspInstance> {
    vec![
        TspInstance {
            name: "eil51",
            optimal: 426.0,
            coords: &EIL51_COORDS,
        },
        TspInstance {
            name: "berlin52",
            optimal: 7542.0,
            coords: &BERLIN52_COORDS,
        },
        TspInstance {
            name: "kroA100",
            optimal: 21282.0,
            coords: &KROA100_COORDS,
        },
    ]
}

/// Run benchmark for a single priority function
fn benchmark_priority<P: TwoOptPriority>(priority: &P, instances: &[TspInstance]) -> BenchmarkResult {
    let start = Instant::now();
    let mut instance_results = Vec::new();
    let max_iterations = 10000;

    for inst in instances {
        let dm = DistanceMatrix::from_coords(inst.coords);

        // Start with nearest neighbor tour
        let mut tour = nearest_neighbor_tour(&dm);
        let _initial_len = tour_length(&tour, &dm);

        // Run 2-opt with this priority function
        let final_len = two_opt_search(&mut tour, &dm, priority, max_iterations);

        let gap_percent = (final_len - inst.optimal) / inst.optimal * 100.0;

        instance_results.push(InstanceResult {
            name: inst.name.to_string(),
            optimal: inst.optimal,
            found: final_len,
            gap_percent,
            iterations: max_iterations,
        });
    }

    let avg_gap = instance_results.iter().map(|r| r.gap_percent).sum::<f64>()
        / instance_results.len() as f64;

    BenchmarkResult {
        name: priority.name().to_string(),
        instances: instance_results,
        avg_gap_percent: avg_gap,
        total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
    }
}

fn main() {
    let instances = get_instances();

    // Run all priority functions
    let results = vec![
        benchmark_priority(&GreedyDelta, &instances),
        benchmark_priority(&BestImprovement, &instances),
        benchmark_priority(&RelativeGain, &instances),
        benchmark_priority(&LongEdgeRemoval, &instances),
        benchmark_priority(&EdgeRatio, &instances),
        benchmark_priority(&LKInspired, &instances),
        benchmark_priority(&Balanced, &instances),
        benchmark_priority(&Evolved, &instances),
    ];

    // Output JSON
    println!("{{");
    println!("  \"benchmark\": \"tsp-2opt\",");
    println!("  \"instances\": [\"eil51\", \"berlin52\", \"kroA100\"],");
    println!("  \"optimal_values\": {{\"eil51\": 426, \"berlin52\": 7542, \"kroA100\": 21282}},");
    println!("  \"results\": [");

    for (i, result) in results.iter().enumerate() {
        let comma = if i < results.len() - 1 { "," } else { "" };
        println!("    {{");
        println!("      \"name\": \"{}\",", result.name);
        println!("      \"avg_gap_percent\": {:.4},", result.avg_gap_percent);
        println!("      \"time_ms\": {:.2},", result.total_time_ms);
        println!("      \"instances\": [");

        for (j, inst) in result.instances.iter().enumerate() {
            let inst_comma = if j < result.instances.len() - 1 { "," } else { "" };
            println!(
                "        {{\"name\": \"{}\", \"optimal\": {}, \"found\": {:.0}, \"gap_percent\": {:.4}}}{}",
                inst.name, inst.optimal, inst.found, inst.gap_percent, inst_comma
            );
        }

        println!("      ]");
        println!("    }}{}", comma);
    }

    println!("  ]");
    println!("}}");
}
