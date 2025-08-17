#!/bin/bash
set -e

echo "Building Black Hole Renderer..."

# Build the Zig compute library
echo "Building compute library..."
cd compute
zig build -Doptimize=ReleaseFast
cd ..

# Build the Rust viewer
echo "Building viewer application..."
cd viewer
cargo build --release
cd ..

echo "Build complete!"
echo "Run with: ./scripts/run.sh"
