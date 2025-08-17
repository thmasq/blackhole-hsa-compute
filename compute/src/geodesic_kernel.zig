const std = @import("std");
const builtin = @import("builtin");

// AMDGCN intrinsics
extern fn @"llvm.amdgcn.workitem.id.x"() u32;
extern fn @"llvm.amdgcn.workitem.id.y"() u32;
extern fn @"llvm.amdgcn.workgroup.id.x"() u32;
extern fn @"llvm.amdgcn.workgroup.id.y"() u32;
extern fn @"llvm.amdgcn.s.barrier"() void;

// Mathematical intrinsics
extern fn @"llvm.sqrt.f32"(f32) f32;
extern fn @"llvm.cos.f32"(f32) f32;
extern fn @"llvm.sin.f32"(f32) f32;

// Shared memory for tile-based computation
var shared_colors: [256]u32 addrspace(.shared) = undefined;

// Constants for black hole physics
const SAGA_RS: f32 = 1.269e10; // Schwarzschild radius of Sagittarius A*
const D_LAMBDA: f32 = 1e7; // Integration step size
const ESCAPE_R: f32 = 1e30; // Escape radius
const PI: f32 = 3.14159265359;

// Ray structure for geodesic integration
const Ray = struct {
    x: f32,
    y: f32,
    z: f32,
    r: f32,
    theta: f32,
    phi: f32,
    dr: f32,
    dtheta: f32,
    dphi: f32,
    E: f32,
    L: f32,
};

// Camera uniforms structure
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

// Helper math functions
inline fn length3(x: f32, y: f32, z: f32) f32 {
    return @"llvm.sqrt.f32"(x * x + y * y + z * z);
}

inline fn normalize3(x: *f32, y: *f32, z: *f32) void {
    const len = length3(x.*, y.*, z.*);
    if (len > 0.0) {
        x.* /= len;
        y.* /= len;
        z.* /= len;
    }
}

inline fn atan2_approx(y: f32, x: f32) f32 {
    // Approximate atan2 for GPU
    const abs_y = @abs(y);
    const abs_x = @abs(x);
    const max_val = @max(abs_x, abs_y);
    const min_val = @min(abs_x, abs_y);

    if (max_val == 0.0) return 0.0;

    const ratio = min_val / max_val;
    const angle = ratio * (PI / 4.0);

    if (abs_y > abs_x) {
        if (y >= 0) return PI / 2.0 - angle;
        return -PI / 2.0 + angle;
    }
    if (x >= 0) return angle;
    if (y >= 0) return PI - angle;
    return -PI + angle;
}

inline fn acos_approx(x: f32) f32 {
    // Approximate acos using polynomial
    const x_clamped = @min(@max(x, -1.0), 1.0);
    const x2 = x_clamped * x_clamped;
    const x3 = x2 * x_clamped;
    return PI / 2.0 - (x_clamped + x3 / 6.0 + 3.0 * x2 * x3 / 40.0);
}

// Initialize ray from position and direction
fn initRay(pos_x: f32, pos_y: f32, pos_z: f32, dir_x: f32, dir_y: f32, dir_z: f32) Ray {
    var ray: Ray = undefined;

    ray.x = pos_x;
    ray.y = pos_y;
    ray.z = pos_z;
    ray.r = length3(pos_x, pos_y, pos_z);

    if (ray.r > 0.0) {
        ray.theta = acos_approx(pos_z / ray.r);
        ray.phi = atan2_approx(pos_y, pos_x);
    } else {
        ray.theta = 0.0;
        ray.phi = 0.0;
    }

    const sin_theta = @"llvm.sin.f32"(ray.theta);
    const cos_theta = @"llvm.cos.f32"(ray.theta);
    const sin_phi = @"llvm.sin.f32"(ray.phi);
    const cos_phi = @"llvm.cos.f32"(ray.phi);

    ray.dr = sin_theta * cos_phi * dir_x + sin_theta * sin_phi * dir_y + cos_theta * dir_z;

    if (ray.r > 0.0) {
        ray.dtheta = (cos_theta * cos_phi * dir_x + cos_theta * sin_phi * dir_y - sin_theta * dir_z) / ray.r;
        if (sin_theta > 0.0) {
            ray.dphi = (-sin_phi * dir_x + cos_phi * dir_y) / (ray.r * sin_theta);
        } else {
            ray.dphi = 0.0;
        }
    } else {
        ray.dtheta = 0.0;
        ray.dphi = 0.0;
    }

    ray.L = ray.r * ray.r * sin_theta * ray.dphi;
    const f = 1.0 - SAGA_RS / ray.r;
    const dt_dL = @"llvm.sqrt.f32"((ray.dr * ray.dr) / f + ray.r * ray.r * (ray.dtheta * ray.dtheta + sin_theta * sin_theta * ray.dphi * ray.dphi));
    ray.E = f * dt_dL;

    return ray;
}

// Geodesic right-hand side for RK4 integration
fn geodesicRHS(ray: *const Ray, d1_r: *f32, d1_theta: *f32, d1_phi: *f32, d2_r: *f32, d2_theta: *f32, d2_phi: *f32) void {
    const r = ray.r;
    const theta = ray.theta;
    const dr = ray.dr;
    const dtheta = ray.dtheta;
    const dphi = ray.dphi;

    const f = 1.0 - SAGA_RS / r;
    const dt_dL = ray.E / f;
    const sin_theta = @"llvm.sin.f32"(theta);
    const cos_theta = @"llvm.cos.f32"(theta);

    d1_r.* = dr;
    d1_theta.* = dtheta;
    d1_phi.* = dphi;

    d2_r.* = -(SAGA_RS / (2.0 * r * r)) * f * dt_dL * dt_dL +
        (SAGA_RS / (2.0 * r * r * f)) * dr * dr +
        r * (dtheta * dtheta + sin_theta * sin_theta * dphi * dphi);

    d2_theta.* = -2.0 * dr * dtheta / r + sin_theta * cos_theta * dphi * dphi;

    if (sin_theta > 0.0) {
        d2_phi.* = -2.0 * dr * dphi / r - 2.0 * (cos_theta / sin_theta) * dtheta * dphi;
    } else {
        d2_phi.* = 0.0;
    }
}

// RK4 integration step
fn rk4Step(ray: *Ray, dL: f32) void {
    var k1a_r: f32 = undefined;
    var k1a_theta: f32 = undefined;
    var k1a_phi: f32 = undefined;
    var k1b_r: f32 = undefined;
    var k1b_theta: f32 = undefined;
    var k1b_phi: f32 = undefined;

    geodesicRHS(ray, &k1a_r, &k1a_theta, &k1a_phi, &k1b_r, &k1b_theta, &k1b_phi);

    ray.r += dL * k1a_r;
    ray.theta += dL * k1a_theta;
    ray.phi += dL * k1a_phi;
    ray.dr += dL * k1b_r;
    ray.dtheta += dL * k1b_theta;
    ray.dphi += dL * k1b_phi;

    const sin_theta = @"llvm.sin.f32"(ray.theta);
    const cos_theta = @"llvm.cos.f32"(ray.theta);
    const sin_phi = @"llvm.sin.f32"(ray.phi);
    const cos_phi = @"llvm.cos.f32"(ray.phi);

    ray.x = ray.r * sin_theta * cos_phi;
    ray.y = ray.r * sin_theta * sin_phi;
    ray.z = ray.r * cos_theta;
}

// Check if ray intercepts black hole
inline fn interceptBlackHole(ray: *const Ray) bool {
    return ray.r <= SAGA_RS;
}

// Check if ray crosses accretion disk
inline fn crossesEquatorialPlane(old_y: f32, new_y: f32, new_x: f32, new_z: f32, disk: *const DiskData) bool {
    const crossed = (old_y * new_y < 0.0);
    const r = @"llvm.sqrt.f32"(new_x * new_x + new_z * new_z);
    return crossed and (r >= disk.r1 and r <= disk.r2);
}

// Pack color into u32 for output
inline fn packColor(r: f32, g: f32, b: f32, a: f32) u32 {
    const r8: u32 = @intFromFloat(@min(@max(r * 255.0, 0.0), 255.0));
    const g8: u32 = @intFromFloat(@min(@max(g * 255.0, 0.0), 255.0));
    const b8: u32 = @intFromFloat(@min(@max(b * 255.0, 0.0), 255.0));
    const a8: u32 = @intFromFloat(@min(@max(a * 255.0, 0.0), 255.0));
    return (a8 << 24) | (b8 << 16) | (g8 << 8) | r8;
}

// Main geodesic ray tracing kernel
export fn trace_geodesics(output: [*]addrspace(.global) u32, camera: [*]addrspace(.global) const CameraData, disk: [*]addrspace(.global) const DiskData, width: u32, height: u32, max_steps: u32) callconv(.Kernel) void {
    const local_x = @"llvm.amdgcn.workitem.id.x"();
    const local_y = @"llvm.amdgcn.workitem.id.y"();
    const group_x = @"llvm.amdgcn.workgroup.id.x"();
    const group_y = @"llvm.amdgcn.workgroup.id.y"();

    const workgroup_size_x: u32 = 16;
    const workgroup_size_y: u32 = 16;

    const global_x = group_x * workgroup_size_x + local_x;
    const global_y = group_y * workgroup_size_y + local_y;

    if (global_x >= width or global_y >= height) return;

    // Load camera data
    const cam = camera[0];
    const disk_data = disk[0];

    // Calculate ray direction
    const u = (2.0 * (@as(f32, @floatFromInt(global_x)) + 0.5) / @as(f32, @floatFromInt(width)) - 1.0) *
        cam.aspect * cam.tan_half_fov;
    const v = (1.0 - 2.0 * (@as(f32, @floatFromInt(global_y)) + 0.5) / @as(f32, @floatFromInt(height))) *
        cam.tan_half_fov;

    var dir_x = u * cam.right_x - v * cam.up_x + cam.forward_x;
    var dir_y = u * cam.right_y - v * cam.up_y + cam.forward_y;
    var dir_z = u * cam.right_z - v * cam.up_z + cam.forward_z;
    normalize3(&dir_x, &dir_y, &dir_z);

    // Initialize ray
    var ray = initRay(cam.pos_x, cam.pos_y, cam.pos_z, dir_x, dir_y, dir_z);

    var color_r: f32 = 0.0;
    var color_g: f32 = 0.0;
    var color_b: f32 = 0.0;
    var color_a: f32 = 1.0;

    var prev_y = ray.y;
    var hit_black_hole = false;
    var hit_disk = false;

    // Integrate geodesic
    var step: u32 = 0;
    const steps = if (cam.moving != 0) max_steps / 2 else max_steps;

    while (step < steps) : (step += 1) {
        if (interceptBlackHole(&ray)) {
            hit_black_hole = true;
            break;
        }

        rk4Step(&ray, D_LAMBDA);

        if (crossesEquatorialPlane(prev_y, ray.y, ray.x, ray.z, &disk_data)) {
            hit_disk = true;
            break;
        }

        prev_y = ray.y;

        if (ray.r > ESCAPE_R) break;
    }

    // Color based on hit
    if (hit_disk) {
        const r_normalized = length3(ray.x, ray.y, ray.z) / disk_data.r2;
        color_r = 1.0;
        color_g = r_normalized;
        color_b = 0.2;
        color_a = r_normalized;
    } else if (hit_black_hole) {
        color_r = 0.0;
        color_g = 0.0;
        color_b = 0.0;
        color_a = 1.0;
    } else {
        // Background (starfield or void)
        color_r = 0.0;
        color_g = 0.0;
        color_b = 0.0;
        color_a = 1.0;
    }

    // Write output
    const pixel_index = global_y * width + global_x;
    output[pixel_index] = packColor(color_r, color_g, color_b, color_a);
}
