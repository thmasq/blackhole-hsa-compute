const std = @import("std");

// HSA Runtime C bindings
const hsa = @cImport({
    @cInclude("hsa/hsa.h");
    @cInclude("hsa/hsa_ext_amd.h");
});

pub const HsaError = error{
    HsaInitFailed,
    AgentNotFound,
    QueueCreationFailed,
    MemoryAllocationFailed,
    CodeObjectLoadFailed,
    KernelNotFound,
    ExecutionFailed,
    RequiredMemoryRegionNotFound,
};

pub const HsaContext = struct {
    agent: hsa.hsa_agent_t,
    queue: ?*hsa.hsa_queue_t,
    kernarg_region: ?hsa.hsa_region_t,
    fine_grained_region: ?hsa.hsa_region_t,
    coarse_grained_region: ?hsa.hsa_region_t,
    initialized: bool,

    const Self = @This();

    pub fn init() !Self {
        const status = hsa.hsa_init();
        if (status != hsa.HSA_STATUS_SUCCESS) {
            return HsaError.HsaInitFailed;
        }

        var ctx = Self{
            .agent = undefined,
            .queue = null,
            .kernarg_region = null,
            .fine_grained_region = null,
            .coarse_grained_region = null,
            .initialized = true,
        };

        const find_status = hsa.hsa_iterate_agents(findGpuAgent, &ctx.agent);
        if (find_status != hsa.HSA_STATUS_SUCCESS and find_status != hsa.HSA_STATUS_INFO_BREAK) {
            _ = hsa.hsa_shut_down();
            return HsaError.AgentNotFound;
        }

        const region_status = hsa.hsa_agent_iterate_regions(ctx.agent, findMemoryRegions, &ctx);
        if (region_status != hsa.HSA_STATUS_SUCCESS) {
            _ = hsa.hsa_shut_down();
            return HsaError.AgentNotFound;
        }

        if (ctx.coarse_grained_region == null) {
            _ = hsa.hsa_shut_down();
            return HsaError.RequiredMemoryRegionNotFound;
        }

        if (ctx.fine_grained_region == null) {
            _ = hsa.hsa_shut_down();
            return HsaError.RequiredMemoryRegionNotFound;
        }

        const queue_status = hsa.hsa_queue_create(
            ctx.agent,
            1024,
            hsa.HSA_QUEUE_TYPE_MULTI,
            null,
            null,
            0,
            0,
            &ctx.queue,
        );
        if (queue_status != hsa.HSA_STATUS_SUCCESS) {
            _ = hsa.hsa_shut_down();
            return HsaError.QueueCreationFailed;
        }

        return ctx;
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;

        if (self.queue) |queue| {
            _ = hsa.hsa_queue_destroy(queue);
        }
        _ = hsa.hsa_shut_down();
        self.initialized = false;
    }

    pub fn allocateMemory(self: *Self, size: usize, comptime T: type) ![]T {
        if (self.coarse_grained_region == null) {
            return HsaError.MemoryAllocationFailed;
        }

        var ptr: ?*anyopaque = null;
        var status = hsa.hsa_memory_allocate(self.coarse_grained_region.?, size, &ptr);
        if (status != hsa.HSA_STATUS_SUCCESS) {
            return HsaError.MemoryAllocationFailed;
        }

        status = hsa.hsa_amd_agents_allow_access(1, &self.agent, null, ptr);
        if (status != hsa.HSA_STATUS_SUCCESS) {
            _ = hsa.hsa_memory_free(ptr);
            return HsaError.MemoryAllocationFailed;
        }

        const typed_ptr: [*]T = @ptrCast(@alignCast(ptr));
        return typed_ptr[0 .. size / @sizeOf(T)];
    }

    pub fn allocateKernargs(self: *Self, size: usize) ![]u8 {
        const region = if (self.kernarg_region) |kr| kr else self.fine_grained_region.?;

        var ptr: ?*anyopaque = null;
        const status = hsa.hsa_memory_allocate(region, size, &ptr);
        if (status != hsa.HSA_STATUS_SUCCESS) {
            return HsaError.MemoryAllocationFailed;
        }

        const byte_ptr: [*]u8 = @ptrCast(ptr);
        return byte_ptr[0..size];
    }

    pub fn freeMemory(self: *Self, ptr: anytype) void {
        _ = self;
        _ = hsa.hsa_memory_free(@ptrCast(ptr.ptr));
    }
};

fn findGpuAgent(agent: hsa.hsa_agent_t, data: ?*anyopaque) callconv(.C) hsa.hsa_status_t {
    var device_type: hsa.hsa_device_type_t = undefined;
    const status = hsa.hsa_agent_get_info(agent, hsa.HSA_AGENT_INFO_DEVICE, &device_type);
    if (status != hsa.HSA_STATUS_SUCCESS) {
        return status;
    }

    if (device_type == hsa.HSA_DEVICE_TYPE_GPU) {
        const agent_ptr: *hsa.hsa_agent_t = @ptrCast(@alignCast(data));
        agent_ptr.* = agent;
        return hsa.HSA_STATUS_INFO_BREAK;
    }

    return hsa.HSA_STATUS_SUCCESS;
}

fn findMemoryRegions(region: hsa.hsa_region_t, data: ?*anyopaque) callconv(.C) hsa.hsa_status_t {
    const ctx: *HsaContext = @ptrCast(@alignCast(data));

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
