#ifndef BLACKHOLE_COMPUTE_H
#define BLACKHOLE_COMPUTE_H

#include <cstddef>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
#define BLACKHOLE_SUCCESS 0
#define BLACKHOLE_ERROR_NOT_INITIALIZED -1
#define BLACKHOLE_ERROR_INVALID_PARAMETER -2
#define BLACKHOLE_ERROR_GPU -3
#define BLACKHOLE_ERROR_MEMORY -4
#define BLACKHOLE_ERROR_KERNEL -5

// Initialize the black hole compute library
// Returns 0 on success, negative error code on failure
int blackhole_init(uint32_t width, uint32_t height, uint32_t max_steps);

// Render a frame to the provided buffer
// Buffer must be at least width * height * 4 bytes (RGBA format)
// Returns 0 on success, negative error code on failure
int blackhole_render_frame(uint8_t* output_buffer, size_t buffer_size);

// Update camera position (spherical coordinates)
void blackhole_update_camera(float azimuth, float elevation, float radius);

// Set camera target position
void blackhole_set_camera_target(float x, float y, float z);

// Set rendering quality (number of integration steps)
void blackhole_set_quality(uint32_t max_steps);

// Set accretion disk parameters
void blackhole_set_disk_params(float inner_radius, float outer_radius, float thickness);

// Get current dimensions
void blackhole_get_dimensions(uint32_t* width, uint32_t* height);

// Cleanup resources
void blackhole_cleanup(void);

// Get last error message
const char* blackhole_get_last_error(void);

// Check if library is initialized
bool blackhole_is_initialized(void);

#ifdef __cplusplus
}
#endif

#endif // BLACKHOLE_COMPUTE_H
