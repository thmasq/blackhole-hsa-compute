const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // GPU target specification for AMD GPUs
    const amdgcn_mcpu = b.option([]const u8, "gpu", "Target GPU features") orelse "gfx1201";
    const amdgcn_target = b.resolveTargetQuery(std.Build.parseTargetQuery(.{
        .arch_os_abi = "amdgcn-amdhsa-none",
        .cpu_features = amdgcn_mcpu,
    }) catch unreachable);

    // Build GPU kernel library for black hole ray tracing
    const gpu_kernel = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "blackhole-kernel",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/geodesic_kernel.zig"),
            .target = amdgcn_target,
            .optimize = .ReleaseFast,
        }),
    });
    gpu_kernel.linker_allow_shlib_undefined = false;
    gpu_kernel.bundle_compiler_rt = false;

    // Get the compiled kernel as a binary module
    const kernel_binary = gpu_kernel.getEmittedBin();

    // Build the main library as a shared library
    const lib = b.addSharedLibrary(.{
        .name = "blackhole_compute",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lib.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
        .version = .{ .major = 1, .minor = 0, .patch = 0 },
    });

    // Link HSA runtime - check multiple possible paths
    const rocm_paths = [_][]const u8{
        "/opt/rocm/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
    };

    const rocm_include_paths = [_][]const u8{
        "/opt/rocm/include",
        "/usr/include/hsa",
        "/usr/include",
    };

    // Try to find ROCm installation
    var rocm_lib_found = false;
    var rocm_include_found = false;

    for (rocm_paths) |path| {
        if (std.fs.cwd().access(path, .{})) {
            lib.addLibraryPath(.{ .cwd_relative = path });
            rocm_lib_found = true;
            break;
        } else |_| {}
    }

    for (rocm_include_paths) |path| {
        if (std.fs.cwd().access(path, .{})) {
            lib.addIncludePath(.{ .cwd_relative = path });
            rocm_include_found = true;
            break;
        } else |_| {}
    }

    if (!rocm_lib_found) {
        std.log.warn("ROCm library path not found, you may need to set LD_LIBRARY_PATH\n", .{});
    }

    if (!rocm_include_found) {
        std.log.warn("ROCm include path not found, compilation may fail\n", .{});
    }

    lib.linkSystemLibrary("hsa-runtime64");
    lib.linkSystemLibrary("m"); // Math library

    // Embed the GPU kernel binary in the library
    lib.root_module.addAnonymousImport("gpu-kernel", .{
        .root_source_file = kernel_binary,
    });

    // Install the shared library
    b.installArtifact(lib);

    // Install the C header
    const install_header = b.addInstallHeaderFile(
        b.path("include/blackhole_compute.h"),
        "blackhole_compute.h",
    );
    b.getInstallStep().dependOn(&install_header.step);

    // Add a test executable for the library
    const test_exe = b.addExecutable(.{
        .name = "test_blackhole",
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_exe.linkLibrary(lib);
    test_exe.linkSystemLibrary("c");

    const run_test = b.addRunArtifact(test_exe);
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_test.step);
}
