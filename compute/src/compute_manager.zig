const std = @import("std");
const hsa_context = @import("hsa_context.zig");
const camera = @import("camera.zig");

// Import the embedded GPU kernel
const gpu_kernel_binary = @embedFile("gpu-kernel");

// HSA Runtime C bindings
const hsa = @cImport({
    @cInclude("hsa/hsa.h");
    @cInclude("hsa/hsa_ext_amd.h");
});

// Camera data structure matching kernel - make it public
pub const CameraData = struct {
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

// Disk parameters - make it public
pub const DiskData = struct {
    r1: f32,
    r2: f32,
    num: f32,
    thickness: f32,
};

pub const ComputeManager = struct {
    ctx: *hsa_context.HsaContext,
    code_object_reader: hsa.hsa_code_object_reader_t,
    executable: hsa.hsa_executable_t,
    kernel_symbol: hsa.hsa_executable_symbol_t,
    kernel_object: u64,

    // Buffers
    output_buffer: []u32,
    camera_buffer: []CameraData,
    disk_buffer: []DiskData,
    kernargs: []u8,

    // Dimensions
    width: u32,
    height: u32,
    max_steps: u32,

    // Black hole parameters (Sagittarius A*)
    saga_mass: f64 = 8.54e36,
    saga_rs: f32 = 1.269e10,

    const Self = @This();

    pub fn init(ctx: *hsa_context.HsaContext, width: u32, height: u32, max_steps: u32) !Self {
        var manager = Self{
            .ctx = ctx,
            .code_object_reader = undefined,
            .executable = undefined,
            .kernel_symbol = undefined,
            .kernel_object = undefined,
            .output_buffer = undefined,
            .camera_buffer = undefined,
            .disk_buffer = undefined,
            .kernargs = undefined,
            .width = width,
            .height = height,
            .max_steps = max_steps,
        };

        // Load kernel
        try manager.loadKernel();

        // Allocate buffers
        try manager.allocateBuffers();

        // Initialize disk parameters
        try manager.initializeDisk();

        // Setup kernel arguments
        try manager.setupKernargs();

        return manager;
    }

    fn loadKernel(self: *Self) !void {
        var status = hsa.hsa_code_object_reader_create_from_memory(
            gpu_kernel_binary.ptr,
            gpu_kernel_binary.len,
            &self.code_object_reader,
        );
        if (status != hsa.HSA_STATUS_SUCCESS) {
            return hsa_context.HsaError.CodeObjectLoadFailed;
        }

        status = hsa.hsa_executable_create_alt(
            hsa.HSA_PROFILE_FULL,
            hsa.HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
            null,
            &self.executable,
        );
        if (status != hsa.HSA_STATUS_SUCCESS) {
            _ = hsa.hsa_code_object_reader_destroy(self.code_object_reader);
            return hsa_context.HsaError.CodeObjectLoadFailed;
        }

        status = hsa.hsa_executable_load_agent_code_object(
            self.executable,
            self.ctx.agent,
            self.code_object_reader,
            null,
            null,
        );
        if (status != hsa.HSA_STATUS_SUCCESS) {
            _ = hsa.hsa_executable_destroy(self.executable);
            _ = hsa.hsa_code_object_reader_destroy(self.code_object_reader);
            return hsa_context.HsaError.CodeObjectLoadFailed;
        }

        status = hsa.hsa_executable_freeze(self.executable, null);
        if (status != hsa.HSA_STATUS_SUCCESS) {
            _ = hsa.hsa_executable_destroy(self.executable);
            _ = hsa.hsa_code_object_reader_destroy(self.code_object_reader);
            return hsa_context.HsaError.CodeObjectLoadFailed;
        }

        // Get kernel symbol
        var kernel_name_buffer: [256]u8 = undefined;
        const kernel_name = std.fmt.bufPrintZ(&kernel_name_buffer, "trace_geodesics.kd", .{}) catch {
            return hsa_context.HsaError.KernelNotFound;
        };

        status = hsa.hsa_executable_get_symbol_by_name(
            self.executable,
            kernel_name.ptr,
            &self.ctx.agent,
            &self.kernel_symbol,
        );

        if (status != hsa.HSA_STATUS_SUCCESS) {
            // Try without .kd suffix
            const base_name = std.fmt.bufPrintZ(&kernel_name_buffer, "trace_geodesics", .{}) catch {
                return hsa_context.HsaError.KernelNotFound;
            };

            status = hsa.hsa_executable_get_symbol_by_name(
                self.executable,
                base_name.ptr,
                &self.ctx.agent,
                &self.kernel_symbol,
            );

            if (status != hsa.HSA_STATUS_SUCCESS) {
                return hsa_context.HsaError.KernelNotFound;
            }
        }

        // Get kernel object
        status = hsa.hsa_executable_symbol_get_info(
            self.kernel_symbol,
            hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
            &self.kernel_object,
        );
        if (status != hsa.HSA_STATUS_SUCCESS) {
            return hsa_context.HsaError.KernelNotFound;
        }
    }

    fn allocateBuffers(self: *Self) !void {
        // Allocate output buffer
        self.output_buffer = try self.ctx.allocateMemory(
            self.width * self.height * @sizeOf(u32),
            u32,
        );

        // Allocate camera buffer
        self.camera_buffer = try self.ctx.allocateMemory(@sizeOf(CameraData), CameraData);

        // Allocate disk buffer
        self.disk_buffer = try self.ctx.allocateMemory(@sizeOf(DiskData), DiskData);

        // Allocate kernargs
        self.kernargs = try self.ctx.allocateKernargs(48);
    }

    fn initializeDisk(self: *Self) !void {
        self.disk_buffer[0] = DiskData{
            .r1 = self.saga_rs * 2.2, // Inner radius just outside event horizon
            .r2 = self.saga_rs * 5.2, // Outer radius
            .num = 2.0,
            .thickness = 1e9,
        };
    }

    fn setupKernargs(self: *Self) !void {
        const arg_ptrs = std.mem.bytesAsSlice(u64, self.kernargs[0..40]);
        arg_ptrs[0] = @intFromPtr(self.output_buffer.ptr);
        arg_ptrs[1] = @intFromPtr(self.camera_buffer.ptr);
        arg_ptrs[2] = @intFromPtr(self.disk_buffer.ptr);

        const width_ptr = @as(*u32, @ptrCast(@alignCast(&self.kernargs[24])));
        width_ptr.* = self.width;

        const height_ptr = @as(*u32, @ptrCast(@alignCast(&self.kernargs[28])));
        height_ptr.* = self.height;

        const steps_ptr = @as(*u32, @ptrCast(@alignCast(&self.kernargs[32])));
        steps_ptr.* = self.max_steps;
    }

    pub fn updateCamera(self: *Self, cam: *const camera.Camera) !void {
        self.camera_buffer[0] = cam.getCameraData(self.width, self.height);
    }

    pub fn updateDiskParams(self: *Self, inner_radius: f32, outer_radius: f32, thickness: f32) !void {
        self.disk_buffer[0] = DiskData{
            .r1 = inner_radius,
            .r2 = outer_radius,
            .num = 2.0,
            .thickness = thickness,
        };
    }

    pub fn executeKernel(self: *Self) !void {
        const queue = self.ctx.queue.?;

        // Update max_steps in kernargs
        const steps_ptr = @as(*u32, @ptrCast(@alignCast(&self.kernargs[32])));
        steps_ptr.* = self.max_steps;

        // Create completion signal
        var completion_signal: hsa.hsa_signal_t = undefined;
        const status = hsa.hsa_signal_create(1, 0, null, &completion_signal);
        if (status != hsa.HSA_STATUS_SUCCESS) {
            return hsa_context.HsaError.ExecutionFailed;
        }
        defer _ = hsa.hsa_signal_destroy(completion_signal);

        // Get packet
        const packet_id = hsa.hsa_queue_add_write_index_relaxed(queue, 1);
        const base_packets = @as([*]hsa.hsa_kernel_dispatch_packet_t, @ptrCast(@alignCast(queue.base_address)));
        const packet_ptr = &base_packets[@mod(packet_id, queue.size)];

        // Clear packet
        @memset(@as([*]u8, @ptrCast(packet_ptr))[0..@sizeOf(hsa.hsa_kernel_dispatch_packet_t)], 0);

        // Configure packet
        const workgroup_size: u32 = 16;
        const grid_x = (self.width + workgroup_size - 1) / workgroup_size * workgroup_size;
        const grid_y = (self.height + workgroup_size - 1) / workgroup_size * workgroup_size;

        packet_ptr.setup = @as(u16, 2) << hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
        packet_ptr.header = @as(u16, hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH) << hsa.HSA_PACKET_HEADER_TYPE |
            @as(u16, hsa.HSA_FENCE_SCOPE_SYSTEM) << hsa.HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE |
            @as(u16, hsa.HSA_FENCE_SCOPE_SYSTEM) << hsa.HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

        packet_ptr.workgroup_size_x = workgroup_size;
        packet_ptr.workgroup_size_y = workgroup_size;
        packet_ptr.workgroup_size_z = 1;
        packet_ptr.grid_size_x = grid_x;
        packet_ptr.grid_size_y = grid_y;
        packet_ptr.grid_size_z = 1;
        packet_ptr.kernel_object = self.kernel_object;
        packet_ptr.kernarg_address = @ptrCast(@constCast(self.kernargs.ptr));
        packet_ptr.private_segment_size = 0;
        packet_ptr.group_segment_size = 2048;
        packet_ptr.completion_signal = completion_signal;

        // Submit packet
        hsa.hsa_queue_store_write_index_relaxed(queue, packet_id + 1);
        hsa.hsa_signal_store_relaxed(queue.doorbell_signal, @intCast(packet_id));

        // Wait for completion
        const wait_result = hsa.hsa_signal_wait_scacquire(
            completion_signal,
            hsa.HSA_SIGNAL_CONDITION_EQ,
            0,
            std.math.maxInt(u64),
            hsa.HSA_WAIT_STATE_BLOCKED,
        );

        if (wait_result != 0) {
            return hsa_context.HsaError.ExecutionFailed;
        }
    }

    pub fn readOutput(self: *Self, output: []u8) !void {
        const src_bytes = std.mem.sliceAsBytes(self.output_buffer);
        @memcpy(output, src_bytes[0..output.len]);
    }

    pub fn deinit(self: *Self) void {
        self.ctx.freeMemory(self.output_buffer);
        self.ctx.freeMemory(self.camera_buffer);
        self.ctx.freeMemory(self.disk_buffer);
        self.ctx.freeMemory(self.kernargs);

        _ = hsa.hsa_executable_destroy(self.executable);
        _ = hsa.hsa_code_object_reader_destroy(self.code_object_reader);
    }
};
