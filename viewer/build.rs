use std::env;
use std::path::PathBuf;

fn main() {
    // Get the compute library directory
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_root = PathBuf::from(&manifest_dir).parent().unwrap().to_path_buf();
    let compute_lib_dir = project_root.join("compute").join("zig-out").join("lib");

    // Link to the blackhole_compute library
    println!(
        "cargo:rustc-link-search=native={}",
        compute_lib_dir.display()
    );
    println!("cargo:rustc-link-lib=dylib=blackhole_compute");

    // Also check in standard library paths in case it's installed system-wide
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-search=native=/usr/lib");

    // Set rpath for runtime library loading
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        compute_lib_dir.display()
    );

    // Rerun if the library changes
    println!("cargo:rerun-if-changed={}", compute_lib_dir.display());

    // For Wayland support
    if cfg!(target_os = "linux") {
        // Try to find Wayland libraries
        if let Ok(lib) = pkg_config::probe_library("wayland-client") {
            for path in &lib.link_paths {
                println!("cargo:rustc-link-search=native={}", path.display());
            }
        }
    }
}
