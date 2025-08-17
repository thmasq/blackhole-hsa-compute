const std = @import("std");
const print = std.debug.print;
const Allocator = std.mem.Allocator;
const math = std.math;

// Import the embedded GPU kernel
const gpu_kernel_binary = @embedFile("gpu-kernel");

// HSA Runtime C bindings
const hsa = @cImport({
    @cInclude("hsa/hsa.h");
    @cInclude("hsa/hsa_ext_amd.h");
});

const HsaError = error{
    HsaInitFailed,
    AgentNotFound,
    QueueCreationFailed,
    MemoryAllocationFailed,
    CodeObjectLoadFailed,
    KernelNotFound,
    ExecutionFailed,
    RequiredMemoryRegionNotFound,
};

// Camera data structure matching kernel
const CameraData = struct {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    _pad0: f32,
    right_x: f32,
    right_y: f32,
    right_z: f32,
    _pad1: f32,
    up_x: f32,
    up_y: f32,
    up_z: f32,
    _pad2: f32,
    forward_x: f32,
    forward_y: f32,
    forward_z: f32,
    _pad3: f32,
    tan_half_fov: f32,
    aspect: f32,
    moving: u32,
    _pad4: u32,
};

// Disk parameters
const DiskData = struct {
    r1: f32,
    r2: f32,
    num: f32,
    thickness: f32,
};

// Black hole parameters
const BlackHole = struct {
    mass: f64,
    r_s: f64, // Schwarzschild radius

    const G: f64 = 6.67430e-11;
    const c: f64 = 299792458.0;

    fn init(mass: f64) BlackHole {
        return .{
            .mass = mass,
            .r_s = 2.0 * G * mass / (c * c),
        };
    }
};

// Camera controller
const Camera = struct {
    target_x: f32 = 0.0,
    target_y: f32 = 0.0,
    target_z: f32 = 0.0,
    radius: f32 = 6.34194e10,
    min_radius: f32 = 1e10,
    max_radius: f32 = 1e12,
    azimuth: f32 = 0.0,
    elevation: f32 = math.pi / 2.0,
    orbit_speed: f32 = 0.01,
    zoom_speed: f64 = 25e9,
    moving: bool = false,

    fn position(self: *const Camera) struct { x: f32, y: f32, z: f32 } {
        const clamped_elevation = @min(@max(self.elevation, 0.01), math.pi - 0.01);
        return .{
            .x = self.radius * @sin(clamped_elevation) * @cos(self.azimuth),
            .y = self.radius * @cos(clamped_elevation),
            .z = self.radius * @sin(clamped_elevation) * @sin(self.azimuth),
        };
    }

    fn getCameraData(self: *const Camera, width: u32, height: u32) CameraData {
        const pos = self.position();

        // Calculate view vectors
        const forward_x = -pos.x; // Looking at origin
        const forward_y = -pos.y;
        const forward_z = -pos.z;
        const forward_len = @sqrt(forward_x * forward_x + forward_y * forward_y + forward_z * forward_z);

        const fwd_x = forward_x / forward_len;
        const fwd_y = forward_y / forward_len;
        const fwd_z = forward_z / forward_len;

        // Right = cross(forward, world_up)
        const up_world_x: f32 = 0.0;
        const up_world_y: f32 = 1.0;
        const up_world_z: f32 = 0.0;

        var right_x = fwd_y * up_world_z - fwd_z * up_world_y;
        var right_y = fwd_z * up_world_x - fwd_x * up_world_z;
        var right_z = fwd_x * up_world_y - fwd_y * up_world_x;
        const right_len = @sqrt(right_x * right_x + right_y * right_y + right_z * right_z);

        if (right_len > 0.0) {
            right_x /= right_len;
            right_y /= right_len;
            right_z /= right_len;
        }

        // Up = cross(right, forward)
        const up_x = right_y * fwd_z - right_z * fwd_y;
        const up_y = right_z * fwd_x - right_x * fwd_z;
        const up_z = right_x * fwd_y - right_y * fwd_x;

        return .{
            .pos_x = pos.x,
            .pos_y = pos.y,
            .pos_z = pos.z,
            ._pad0 = 0.0,
            .right_x = right_x,
            .right_y = right_y,
            .right_z = right_z,
            ._pad1 = 0.0,
            .up_x = up_x,
            .up_y = up_y,
            .up_z = up_z,
            ._pad2 = 0.0,
            .forward_x = fwd_x,
            .forward_y = fwd_y,
            .forward_z = fwd_z,
            ._pad3 = 0.0,
            .tan_half_fov = @tan(math.degreesToRadians(60.0 * 0.5)),
            .aspect = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height)),
            .moving = if (self.moving) 1 else 0,
            ._pad4 = 0,
        };
    }
};

const GpuContext = struct {
    agent: hsa.hsa_agent_t,
    queue: ?*hsa.hsa_queue_t,
    kernarg_region: ?hsa.hsa_region_t,
    fine_grained_region: ?hsa.hsa_region_t,
    coarse_grained_region: ?hsa.hsa_region_t,

    const Self = @This();

    fn init() !Self {
        const status = hsa.hsa_init();
        if (status != hsa.HSA_STATUS_SUCCESS) {
            print("HSA init failed with status: {}\n", .{status});
            return HsaError.HsaInitFailed;
        }

        print("HSA initialized successfully\n", .{});

        var ctx = Self{
            .agent = undefined,
            .queue = null,
            .kernarg_region = null,
            .fine_grained_region = null,
            .coarse_grained_region = null,
        };

        print("Searching for GPU agents...\n", .{});

        const find_status = hsa.hsa_iterate_agents(findGpuAgent, &ctx.agent);
        print("Agent iteration returned: {}\n", .{find_status});

        if (find_status != hsa.HSA_STATUS_SUCCESS and find_status != hsa.HSA_STATUS_INFO_BREAK) {
            print("No GPU agent found. Status: {}\n", .{find_status});
            return HsaError.AgentNotFound;
        }

        print("Searching for memory regions...\n", .{});
        const region_status = hsa.hsa_agent_iterate_regions(ctx.agent, findMemoryRegions, &ctx);
        if (region_status != hsa.HSA_STATUS_SUCCESS) {
            return HsaError.AgentNotFound;
        }

        if (ctx.coarse_grained_region == null) {
            print("ERROR: Coarse-grained region not found!\n", .{});
            return HsaError.RequiredMemoryRegionNotFound;
        }

        if (ctx.fine_grained_region == null) {
            print("ERROR: Fine-grained region not found!\n", .{});
            return HsaError.RequiredMemoryRegionNotFound;
        }

        const queue_status = hsa.hsa_queue_create(ctx.agent, 1024, hsa.HSA_QUEUE_TYPE_MULTI, null, null, 0, 0, &ctx.queue);
        if (queue_status != hsa.HSA_STATUS_SUCCESS) {
            return HsaError.QueueCreationFailed;
        }

        return ctx;
    }

    fn deinit(self: *Self) void {
        if (self.queue) |queue| {
            _ = hsa.hsa_queue_destroy(queue);
        }
        _ = hsa.hsa_shut_down();
    }

    fn allocateMemory(self: *Self, size: usize, comptime T: type) ![]T {
        if (self.coarse_grained_region == null) {
            print("ERROR: No coarse-grained region available for memory allocation\n", .{});
            return HsaError.MemoryAllocationFailed;
        }

        print("Allocating {} bytes ({} elements of {s})\n", .{ size, size / @sizeOf(T), @typeName(T) });

        var ptr: ?*anyopaque = null;
        var status = hsa.hsa_memory_allocate(self.coarse_grained_region.?, size, &ptr);
        if (status != hsa.HSA_STATUS_SUCCESS) {
            print("Memory allocation failed with status: {}\n", .{status});
            return HsaError.MemoryAllocationFailed;
        }

        status = hsa.hsa_amd_agents_allow_access(1, &self.agent, null, ptr);
        if (status != hsa.HSA_STATUS_SUCCESS) {
            print("Failed to grant GPU access to memory: {}\n", .{status});
            _ = hsa.hsa_memory_free(ptr);
            return HsaError.MemoryAllocationFailed;
        }

        const typed_ptr: [*]T = @ptrCast(@alignCast(ptr));
        return typed_ptr[0 .. size / @sizeOf(T)];
    }

    fn allocateKernargs(self: *Self, size: usize) ![]u8 {
        const region = if (self.kernarg_region) |kr| kr else self.fine_grained_region.?;

        var ptr: ?*anyopaque = null;
        const status = hsa.hsa_memory_allocate(region, size, &ptr);
        if (status != hsa.HSA_STATUS_SUCCESS) {
            return HsaError.MemoryAllocationFailed;
        }

        const byte_ptr: [*]u8 = @ptrCast(ptr);
        return byte_ptr[0..size];
    }

    fn freeMemory(self: *Self, ptr: anytype) void {
        _ = self;
        _ = hsa.hsa_memory_free(@ptrCast(ptr.ptr));
    }
};

const KernelManager = struct {
    code_object_reader: hsa.hsa_code_object_reader_t,
    executable: hsa.hsa_executable_t,
    agent: hsa.hsa_agent_t,

    const Self = @This();

    fn init(ctx: *GpuContext) !Self {
        print("Loading embedded kernel binary ({} bytes)...\n", .{gpu_kernel_binary.len});

        var manager = Self{
            .code_object_reader = undefined,
            .executable = undefined,
            .agent = ctx.agent,
        };

        var status = hsa.hsa_code_object_reader_create_from_memory(gpu_kernel_binary.ptr, gpu_kernel_binary.len, &manager.code_object_reader);
        if (status != hsa.HSA_STATUS_SUCCESS) {
            print("Failed to create code object reader: {}\n", .{status});
            return HsaError.CodeObjectLoadFailed;
        }

        status = hsa.hsa_executable_create_alt(hsa.HSA_PROFILE_FULL, hsa.HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, null, &manager.executable);
        if (status != hsa.HSA_STATUS_SUCCESS) {
            _ = hsa.hsa_code_object_reader_destroy(manager.code_object_reader);
            return HsaError.CodeObjectLoadFailed;
        }

        status = hsa.hsa_executable_load_agent_code_object(manager.executable, ctx.agent, manager.code_object_reader, null, null);
        if (status != hsa.HSA_STATUS_SUCCESS) {
            _ = hsa.hsa_executable_destroy(manager.executable);
            _ = hsa.hsa_code_object_reader_destroy(manager.code_object_reader);
            return HsaError.CodeObjectLoadFailed;
        }

        status = hsa.hsa_executable_freeze(manager.executable, null);
        if (status != hsa.HSA_STATUS_SUCCESS) {
            _ = hsa.hsa_executable_destroy(manager.executable);
            _ = hsa.hsa_code_object_reader_destroy(manager.code_object_reader);
            return HsaError.CodeObjectLoadFailed;
        }

        return manager;
    }

    fn getKernelSymbol(self: *Self, kernel_name: []const u8) !hsa.hsa_executable_symbol_t {
        var symbol: hsa.hsa_executable_symbol_t = undefined;

        var kernel_name_buffer: [256]u8 = undefined;
        const full_name = std.fmt.bufPrintZ(&kernel_name_buffer, "{s}.kd", .{kernel_name}) catch {
            return HsaError.KernelNotFound;
        };

        print("Looking for kernel symbol: {s}\n", .{full_name});

        var status = hsa.hsa_executable_get_symbol_by_name(self.executable, full_name.ptr, &self.agent, &symbol);

        if (status != hsa.HSA_STATUS_SUCCESS) {
            const base_name = std.fmt.bufPrintZ(&kernel_name_buffer, "{s}", .{kernel_name}) catch {
                return HsaError.KernelNotFound;
            };

            status = hsa.hsa_executable_get_symbol_by_name(self.executable, base_name.ptr, &self.agent, &symbol);

            if (status != hsa.HSA_STATUS_SUCCESS) {
                return HsaError.KernelNotFound;
            }
        }

        return symbol;
    }

    fn deinit(self: *Self) void {
        _ = hsa.hsa_executable_destroy(self.executable);
        _ = hsa.hsa_code_object_reader_destroy(self.code_object_reader);
    }
};

fn findGpuAgent(agent: hsa.hsa_agent_t, data: ?*anyopaque) callconv(.C) hsa.hsa_status_t {
    var device_type: hsa.hsa_device_type_t = undefined;
    const status = hsa.hsa_agent_get_info(agent, hsa.HSA_AGENT_INFO_DEVICE, &device_type);
    if (status != hsa.HSA_STATUS_SUCCESS) {
        return status;
    }

    if (device_type == hsa.HSA_DEVICE_TYPE_GPU) {
        print("Found GPU agent!\n", .{});
        const agent_ptr: *hsa.hsa_agent_t = @ptrCast(@alignCast(data));
        agent_ptr.* = agent;
        return hsa.HSA_STATUS_INFO_BREAK;
    }

    return hsa.HSA_STATUS_SUCCESS;
}

fn findMemoryRegions(region: hsa.hsa_region_t, data: ?*anyopaque) callconv(.C) hsa.hsa_status_t {
    const ctx: *GpuContext = @ptrCast(@alignCast(data));

    var segment: hsa.hsa_region_segment_t = undefined;
    const status = hsa.hsa_region_get_info(region, hsa.HSA_REGION_INFO_SEGMENT, &segment);
    if (status != hsa.HSA_STATUS_SUCCESS) {
        return status;
    }

    if (segment == hsa.HSA_REGION_SEGMENT_KERNARG) {
        ctx.kernarg_region = region;
    } else if (segment == hsa.HSA_REGION_SEGMENT_GLOBAL) {
        var flags: hsa.hsa_region_global_flag_t = undefined;
        const flag_status = hsa.hsa_region_get_info(region, hsa.HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
        if (flag_status != hsa.HSA_STATUS_SUCCESS) {
            return flag_status;
        }

        if ((flags & hsa.HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) != 0) {
            ctx.fine_grained_region = region;
        } else {
            ctx.coarse_grained_region = region;
        }
    }

    return hsa.HSA_STATUS_SUCCESS;
}

fn executeKernel(
    ctx: *GpuContext,
    symbol: hsa.hsa_executable_symbol_t,
    kernargs: []const u8,
    grid_size: [3]u32,
    workgroup_size: [3]u32,
) !void {
    var kernel_object: u64 = undefined;
    var status = hsa.hsa_executable_symbol_get_info(symbol, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
    if (status != hsa.HSA_STATUS_SUCCESS) {
        return HsaError.ExecutionFailed;
    }

    const queue = ctx.queue.?;

    var completion_signal: hsa.hsa_signal_t = undefined;
    status = hsa.hsa_signal_create(1, 0, null, &completion_signal);
    if (status != hsa.HSA_STATUS_SUCCESS) {
        return HsaError.ExecutionFailed;
    }
    defer _ = hsa.hsa_signal_destroy(completion_signal);

    const packet_id = hsa.hsa_queue_add_write_index_relaxed(queue, 1);
    const base_packets = @as([*]hsa.hsa_kernel_dispatch_packet_t, @ptrCast(@alignCast(queue.base_address)));
    const packet_ptr = &base_packets[@mod(packet_id, queue.size)];

    @memset(@as([*]u8, @ptrCast(packet_ptr))[0..@sizeOf(hsa.hsa_kernel_dispatch_packet_t)], 0);

    packet_ptr.setup = @as(u16, 2) << hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    packet_ptr.header = @as(u16, hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH) << hsa.HSA_PACKET_HEADER_TYPE |
        @as(u16, hsa.HSA_FENCE_SCOPE_SYSTEM) << hsa.HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE |
        @as(u16, hsa.HSA_FENCE_SCOPE_SYSTEM) << hsa.HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    packet_ptr.workgroup_size_x = @intCast(workgroup_size[0]);
    packet_ptr.workgroup_size_y = @intCast(workgroup_size[1]);
    packet_ptr.workgroup_size_z = @intCast(workgroup_size[2]);
    packet_ptr.grid_size_x = @intCast(grid_size[0]);
    packet_ptr.grid_size_y = @intCast(grid_size[1]);
    packet_ptr.grid_size_z = @intCast(grid_size[2]);
    packet_ptr.kernel_object = kernel_object;
    packet_ptr.kernarg_address = @ptrCast(@constCast(kernargs.ptr));
    packet_ptr.private_segment_size = 0;
    packet_ptr.group_segment_size = 2048;
    packet_ptr.completion_signal = completion_signal;

    hsa.hsa_queue_store_write_index_relaxed(queue, packet_id + 1);
    hsa.hsa_signal_store_relaxed(queue.doorbell_signal, @intCast(packet_id));

    const wait_result = hsa.hsa_signal_wait_scacquire(completion_signal, hsa.HSA_SIGNAL_CONDITION_EQ, 0, std.math.maxInt(u64), hsa.HSA_WAIT_STATE_BLOCKED);

    if (wait_result != 0) {
        print("Kernel execution may have failed or timed out\n", .{});
        return HsaError.ExecutionFailed;
    }
}

fn simulateBlackHole(
    ctx: *GpuContext,
    kernel_manager: *KernelManager,
) !void {
    print("\n=== Black Hole Ray Tracing Simulation ===\n", .{});

    const width: u32 = 800;
    const height: u32 = 600;
    const max_steps: u32 = 60000;

    // Initialize black hole (Sagittarius A*)
    const saga = BlackHole.init(8.54e36);
    print("Black hole mass: {e} kg\n", .{saga.mass});
    print("Schwarzschild radius: {e} m\n", .{saga.r_s});

    // Initialize camera
    var camera = Camera{};

    // Allocate GPU memory for output image
    const output_buffer = try ctx.allocateMemory(width * height * @sizeOf(u32), u32);
    defer ctx.freeMemory(output_buffer);

    // Allocate camera data buffer
    const camera_buffer = try ctx.allocateMemory(@sizeOf(CameraData), CameraData);
    defer ctx.freeMemory(camera_buffer);

    // Allocate disk data buffer
    const disk_buffer = try ctx.allocateMemory(@sizeOf(DiskData), DiskData);
    defer ctx.freeMemory(disk_buffer);

    // Set up disk parameters (accretion disk)
    const disk_data = DiskData{
        .r1 = @floatCast(saga.r_s * 2.2), // Inner radius just outside event horizon
        .r2 = @floatCast(saga.r_s * 5.2), // Outer radius
        .num = 2.0,
        .thickness = 1e9,
    };
    disk_buffer[0] = disk_data;

    // Setup kernel arguments
    const kernargs = try ctx.allocateKernargs(48);
    defer ctx.freeMemory(kernargs);

    const arg_ptrs = std.mem.bytesAsSlice(u64, kernargs[0..40]);
    arg_ptrs[0] = @intFromPtr(output_buffer.ptr);
    arg_ptrs[1] = @intFromPtr(camera_buffer.ptr);
    arg_ptrs[2] = @intFromPtr(disk_buffer.ptr);

    const width_ptr = @as(*u32, @ptrCast(@alignCast(&kernargs[24])));
    width_ptr.* = width;

    const height_ptr = @as(*u32, @ptrCast(@alignCast(&kernargs[28])));
    height_ptr.* = height;

    const steps_ptr = @as(*u32, @ptrCast(@alignCast(&kernargs[32])));
    steps_ptr.* = max_steps;

    // Get kernel symbol
    const symbol = try kernel_manager.getKernelSymbol("trace_geodesics");

    // Simulate multiple frames with camera movement
    print("\nRunning simulation...\n", .{});

    var frame: u32 = 0;
    while (frame < 10) : (frame += 1) {
        // Update camera position (orbit around black hole)
        camera.azimuth += 0.1;
        camera_buffer[0] = camera.getCameraData(width, height);

        // Execute kernel
        const workgroup_size: u32 = 16;
        const grid_x = (width + workgroup_size - 1) / workgroup_size * workgroup_size;
        const grid_y = (height + workgroup_size - 1) / workgroup_size * workgroup_size;

        const start_time = std.time.nanoTimestamp();
        try executeKernel(ctx, symbol, kernargs, .{ grid_x, grid_y, 1 }, .{ workgroup_size, workgroup_size, 1 });
        const end_time = std.time.nanoTimestamp();
        const elapsed_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;

        print("Frame {}: {:.2} ms\n", .{ frame, elapsed_ms });

        // Optional: Save frame to file (PPM format)
        if (frame == 0) {
            try saveFrameAsPPM(output_buffer, width, height, "blackhole_frame.ppm");
            print("Saved first frame to blackhole_frame.ppm\n", .{});
        }
    }

    print("\nSimulation completed successfully!\n", .{});
}

fn saveFrameAsPPM(buffer: []u32, width: u32, height: u32, filename: []const u8) !void {
    const file = try std.fs.cwd().createFile(filename, .{});
    defer file.close();

    const writer = file.writer();
    try writer.print("P3\n{} {}\n255\n", .{ width, height });

    for (0..height) |y| {
        for (0..width) |x| {
            const pixel = buffer[y * width + x];
            const r = (pixel >> 0) & 0xFF;
            const g = (pixel >> 8) & 0xFF;
            const b = (pixel >> 16) & 0xFF;
            try writer.print("{} {} {} ", .{ r, g, b });
        }
        try writer.writeByte('\n');
    }
}

pub fn main() !void {
    print("Initializing Black Hole Ray Tracing Simulation with HSA runtime...\n", .{});

    var ctx = GpuContext.init() catch |err| {
        print("Failed to initialize GPU context: {}\n", .{err});
        return;
    };
    defer ctx.deinit();

    print("GPU context initialized successfully\n", .{});

    var kernel_manager = KernelManager.init(&ctx) catch |err| {
        print("Failed to load kernels: {}\n", .{err});
        return;
    };
    defer kernel_manager.deinit();

    print("Kernels loaded successfully\n", .{});

    try simulateBlackHole(&ctx, &kernel_manager);

    print("All tests completed successfully!\n", .{});
}
