const std = @import("std");
const hsa_context = @import("hsa_context.zig");
const compute_manager = @import("compute_manager.zig");
const camera = @import("camera.zig");
const c_api = @import("c_api.zig");

// Global state (protected by mutex for thread safety)
var global_context: ?*hsa_context.HsaContext = null;
var global_manager: ?*compute_manager.ComputeManager = null;
var global_camera: camera.Camera = undefined;
var global_mutex: std.Thread.Mutex = .{};
var last_error: [256]u8 = undefined;
var last_error_len: usize = 0;

fn setError(comptime fmt: []const u8, args: anytype) void {
    const msg = std.fmt.bufPrint(&last_error, fmt, args) catch "Unknown error";
    last_error_len = msg.len;
}

// Initialize the black hole compute library
export fn blackhole_init(width: u32, height: u32, max_steps: u32) c_int {
    global_mutex.lock();
    defer global_mutex.unlock();

    // Clean up any existing context
    if (global_context) |ctx| {
        ctx.deinit();
        std.heap.c_allocator.destroy(ctx);
        global_context = null;
    }
    if (global_manager) |mgr| {
        mgr.deinit();
        std.heap.c_allocator.destroy(mgr);
        global_manager = null;
    }

    // Initialize HSA context
    const ctx = std.heap.c_allocator.create(hsa_context.HsaContext) catch {
        setError("Failed to allocate HSA context", .{});
        return -1;
    };
    errdefer std.heap.c_allocator.destroy(ctx);

    ctx.* = hsa_context.HsaContext.init() catch |err| {
        setError("Failed to initialize HSA: {}", .{err});
        std.heap.c_allocator.destroy(ctx);
        return -1;
    };
    global_context = ctx;

    // Initialize compute manager
    const mgr = std.heap.c_allocator.create(compute_manager.ComputeManager) catch {
        setError("Failed to allocate compute manager", .{});
        ctx.deinit();
        std.heap.c_allocator.destroy(ctx);
        return -1;
    };
    errdefer std.heap.c_allocator.destroy(mgr);

    mgr.* = compute_manager.ComputeManager.init(ctx, width, height, max_steps) catch |err| {
        setError("Failed to initialize compute manager: {}", .{err});
        ctx.deinit();
        std.heap.c_allocator.destroy(ctx);
        std.heap.c_allocator.destroy(mgr);
        return -1;
    };
    global_manager = mgr;

    // Initialize camera with default position
    global_camera = camera.Camera{
        .target_x = 0.0,
        .target_y = 0.0,
        .target_z = 0.0,
        .radius = 6.34194e10,
        .min_radius = 1e10,
        .max_radius = 1e12,
        .azimuth = 0.0,
        .elevation = std.math.pi / 2.0,
        .orbit_speed = 0.01,
        .zoom_speed = 25e9,
        .moving = false,
    };

    return 0;
}

// Render a frame to the provided buffer
export fn blackhole_render_frame(output_buffer: [*]u8, buffer_size: usize) c_int {
    global_mutex.lock();
    defer global_mutex.unlock();

    const mgr = global_manager orelse {
        setError("Library not initialized", .{});
        return -1;
    };

    const expected_size = mgr.width * mgr.height * 4;
    if (buffer_size < expected_size) {
        setError("Buffer too small: need {} bytes, got {}", .{ expected_size, buffer_size });
        return -1;
    }

    // Update camera data
    mgr.updateCamera(&global_camera) catch |err| {
        setError("Failed to update camera: {}", .{err});
        return -1;
    };

    // Execute kernel
    mgr.executeKernel() catch |err| {
        setError("Failed to execute kernel: {}", .{err});
        return -1;
    };

    // Copy output to provided buffer
    mgr.readOutput(output_buffer[0..expected_size]) catch |err| {
        setError("Failed to read output: {}", .{err});
        return -1;
    };

    return 0;
}

// Update camera position
export fn blackhole_update_camera(
    azimuth: f32,
    elevation: f32,
    radius: f32,
) void {
    global_mutex.lock();
    defer global_mutex.unlock();

    global_camera.azimuth = azimuth;
    global_camera.elevation = elevation;
    global_camera.radius = radius;
    global_camera.moving = true;
}

// Set camera target
export fn blackhole_set_camera_target(x: f32, y: f32, z: f32) void {
    global_mutex.lock();
    defer global_mutex.unlock();

    global_camera.target_x = x;
    global_camera.target_y = y;
    global_camera.target_z = z;
}

// Set rendering quality
export fn blackhole_set_quality(max_steps: u32) void {
    global_mutex.lock();
    defer global_mutex.unlock();

    if (global_manager) |mgr| {
        mgr.max_steps = max_steps;
    }
}

// Set disk parameters
export fn blackhole_set_disk_params(inner_radius: f32, outer_radius: f32, thickness: f32) void {
    global_mutex.lock();
    defer global_mutex.unlock();

    if (global_manager) |mgr| {
        mgr.updateDiskParams(inner_radius, outer_radius, thickness) catch {
            setError("Failed to update disk parameters", .{});
        };
    }
}

// Get dimensions
export fn blackhole_get_dimensions(width: *u32, height: *u32) void {
    global_mutex.lock();
    defer global_mutex.unlock();

    if (global_manager) |mgr| {
        width.* = mgr.width;
        height.* = mgr.height;
    } else {
        width.* = 0;
        height.* = 0;
    }
}

// Cleanup resources
export fn blackhole_cleanup() void {
    global_mutex.lock();
    defer global_mutex.unlock();

    if (global_manager) |mgr| {
        mgr.deinit();
        std.heap.c_allocator.destroy(mgr);
        global_manager = null;
    }

    if (global_context) |ctx| {
        ctx.deinit();
        std.heap.c_allocator.destroy(ctx);
        global_context = null;
    }
}

// Get last error message
export fn blackhole_get_last_error() [*:0]const u8 {
    if (last_error_len > 0) {
        last_error[last_error_len] = 0;
        return @ptrCast(&last_error);
    }
    return "No error";
}

// Check if library is initialized
export fn blackhole_is_initialized() bool {
    return global_context != null and global_manager != null;
}
