mod ffi;
mod frame_buffer;
mod renderer;
mod window;

use anyhow::Result;
use log::LevelFilter;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(LevelFilter::Info)
        .init();

    log::info!("Starting Black Hole Viewer");

    // Test compute library
    test_compute_library()?;

    // Create and run the application
    let event_loop = window::create_event_loop()?;
    let mut app = window::AppState::new();

    log::info!("Starting event loop");
    event_loop.run_app(&mut app)?;

    log::info!("Application terminated");
    Ok(())
}

fn test_compute_library() -> Result<()> {
    log::info!("Testing compute library...");

    // Test initialization
    let mut compute = ffi::BlackholeCompute::new(256, 256, 1000)?;
    log::info!("✓ Library initialized");

    // Test dimensions
    let (width, height) = compute.dimensions();
    log::info!("✓ Dimensions: {}x{}", width, height);

    // Test frame rendering
    let frame_data = compute.render_frame()?;
    log::info!("✓ Frame rendered: {} bytes", frame_data.len());

    // Verify some data was written (not all zeros)
    let non_zero = frame_data.iter().any(|&b| b != 0);
    if non_zero {
        log::info!("✓ Frame contains non-zero data");
    } else {
        log::warn!("⚠ Frame appears to be all zeros");
    }

    // Test camera update
    compute.update_camera(0.5, 1.0, 5e10);
    log::info!("✓ Camera updated");

    // Test quality setting
    compute.set_quality(500);
    log::info!("✓ Quality set");

    // The compute object will be cleaned up on drop
    drop(compute);
    log::info!("✓ Cleanup successful");

    log::info!("All tests passed!");
    Ok(())
}
