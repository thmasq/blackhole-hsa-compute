use anyhow::{anyhow, Result};
use std::ffi::{c_char, c_int, CStr};

// Error codes matching the C header
pub const BLACKHOLE_SUCCESS: c_int = 0;
pub const BLACKHOLE_ERROR_NOT_INITIALIZED: c_int = -1;
pub const BLACKHOLE_ERROR_INVALID_PARAMETER: c_int = -2;
pub const BLACKHOLE_ERROR_GPU: c_int = -3;
pub const BLACKHOLE_ERROR_MEMORY: c_int = -4;
pub const BLACKHOLE_ERROR_KERNEL: c_int = -5;

// FFI function declarations
#[link(name = "blackhole_compute")]
extern "C" {
    fn blackhole_init(width: u32, height: u32, max_steps: u32) -> c_int;
    fn blackhole_render_frame(output_buffer: *mut u8, buffer_size: usize) -> c_int;
    fn blackhole_update_camera(azimuth: f32, elevation: f32, radius: f32);
    fn blackhole_set_camera_target(x: f32, y: f32, z: f32);
    fn blackhole_set_quality(max_steps: u32);
    fn blackhole_set_disk_params(inner_radius: f32, outer_radius: f32, thickness: f32);
    fn blackhole_get_dimensions(width: *mut u32, height: *mut u32);
    fn blackhole_cleanup();
    fn blackhole_get_last_error() -> *const c_char;
    fn blackhole_is_initialized() -> bool;
}

/// Safe wrapper around the blackhole compute library
pub struct BlackholeCompute {
    width: u32,
    height: u32,
    frame_buffer: Vec<u8>,
    initialized: bool,
}

impl BlackholeCompute {
    /// Initialize the compute library with given dimensions
    pub fn new(width: u32, height: u32, max_steps: u32) -> Result<Self> {
        log::info!(
            "Initializing BlackholeCompute with {}x{}, {} steps",
            width,
            height,
            max_steps
        );

        let result = unsafe { blackhole_init(width, height, max_steps) };

        if result != BLACKHOLE_SUCCESS {
            let error_msg = Self::get_last_error_string();
            return Err(anyhow!(
                "Failed to initialize blackhole compute: {} (code: {})",
                error_msg,
                result
            ));
        }

        // Verify dimensions
        let mut actual_width = 0u32;
        let mut actual_height = 0u32;
        unsafe {
            blackhole_get_dimensions(&mut actual_width, &mut actual_height);
        }

        if actual_width != width || actual_height != height {
            log::warn!(
                "Dimension mismatch: requested {}x{}, got {}x{}",
                width,
                height,
                actual_width,
                actual_height
            );
        }

        let buffer_size = (width * height * 4) as usize;
        let frame_buffer = vec![0u8; buffer_size];

        log::info!("BlackholeCompute initialized successfully");

        Ok(Self {
            width,
            height,
            frame_buffer,
            initialized: true,
        })
    }

    /// Render a frame and return the buffer
    pub fn render_frame(&mut self) -> Result<&[u8]> {
        if !self.initialized {
            return Err(anyhow!("Compute library not initialized"));
        }

        let buffer_size = self.frame_buffer.len();
        let result = unsafe { blackhole_render_frame(self.frame_buffer.as_mut_ptr(), buffer_size) };

        if result != BLACKHOLE_SUCCESS {
            let error_msg = Self::get_last_error_string();
            return Err(anyhow!(
                "Failed to render frame: {} (code: {})",
                error_msg,
                result
            ));
        }

        Ok(&self.frame_buffer)
    }

    /// Update camera position using spherical coordinates
    pub fn update_camera(&self, azimuth: f32, elevation: f32, radius: f32) {
        if self.initialized {
            unsafe {
                blackhole_update_camera(azimuth, elevation, radius);
            }
        }
    }

    /// Set camera target position
    pub fn set_camera_target(&self, x: f32, y: f32, z: f32) {
        if self.initialized {
            unsafe {
                blackhole_set_camera_target(x, y, z);
            }
        }
    }

    /// Set rendering quality
    pub fn set_quality(&self, max_steps: u32) {
        if self.initialized {
            unsafe {
                blackhole_set_quality(max_steps);
            }
        }
    }

    /// Set accretion disk parameters
    pub fn set_disk_params(&self, inner_radius: f32, outer_radius: f32, thickness: f32) {
        if self.initialized {
            unsafe {
                blackhole_set_disk_params(inner_radius, outer_radius, thickness);
            }
        }
    }

    /// Get dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized && unsafe { blackhole_is_initialized() }
    }

    /// Get the last error as a string
    fn get_last_error_string() -> String {
        unsafe {
            let error_ptr = blackhole_get_last_error();
            if error_ptr.is_null() {
                "Unknown error".to_string()
            } else {
                CStr::from_ptr(error_ptr).to_string_lossy().into_owned()
            }
        }
    }
}

impl Drop for BlackholeCompute {
    fn drop(&mut self) {
        if self.initialized {
            log::info!("Cleaning up BlackholeCompute");
            unsafe {
                blackhole_cleanup();
            }
            self.initialized = false;
        }
    }
}

// Safety: The compute library uses mutexes internally for thread safety
unsafe impl Send for BlackholeCompute {}
unsafe impl Sync for BlackholeCompute {}
