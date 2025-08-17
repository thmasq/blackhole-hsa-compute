// This file contains the type definitions for the C API
// The actual exports are in lib.zig

pub const BlackholeError = enum(c_int) {
    SUCCESS = 0,
    NOT_INITIALIZED = -1,
    INVALID_PARAMETER = -2,
    GPU_ERROR = -3,
    MEMORY_ERROR = -4,
    KERNEL_ERROR = -5,
};
