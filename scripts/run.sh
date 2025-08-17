#!/bin/bash
set -e

# Set library path for the compute library
export LD_LIBRARY_PATH="$PWD/compute/zig-out/lib:$LD_LIBRARY_PATH"

# Set ROCm paths if needed
export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"

# Force Wayland backend
export WINIT_UNIX_BACKEND=wayland

# Enable logging
export RUST_LOG=info

# Run the viewer
cd viewer
cargo run --release
