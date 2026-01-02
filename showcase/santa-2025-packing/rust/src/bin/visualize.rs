//! Generate SVG visualization of the packing

use santa_packing::evolved::EvolvedPacker;
use std::fs;

fn main() {
    let packer = EvolvedPacker::default();

    // Generate packing for n=200
    println!("Generating packing for n=200...");
    let packings = packer.pack_all(200);

    // Get the n=200 packing
    let packing = &packings[199];
    let side = packing.side_length();

    println!("n=200 packing: side={:.4}, trees={}", side, packing.trees.len());

    // Generate SVG
    let svg = packing.to_svg(800, 800);

    // Write to file
    let output_path = "packing_n200.svg";
    fs::write(output_path, &svg).expect("Failed to write SVG");

    println!("SVG saved to {}", output_path);

    // Also generate for a few other n values
    for n in [50, 100, 150] {
        let p = &packings[n - 1];
        let svg = p.to_svg(600, 600);
        let path = format!("packing_n{}.svg", n);
        fs::write(&path, &svg).expect("Failed to write SVG");
        println!("SVG saved to {} (side={:.4})", path, p.side_length());
    }
}
