const std = @import("std");
const math = std.math;

// Import CameraData from compute_manager
const compute_manager = @import("compute_manager.zig");
const CameraData = compute_manager.CameraData;

pub const Camera = struct {
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

    const Self = @This();

    pub fn position(self: *const Self) struct { x: f32, y: f32, z: f32 } {
        const clamped_elevation = @min(@max(self.elevation, 0.01), math.pi - 0.01);
        return .{
            .x = self.radius * @sin(clamped_elevation) * @cos(self.azimuth) + self.target_x,
            .y = self.radius * @cos(clamped_elevation) + self.target_y,
            .z = self.radius * @sin(clamped_elevation) * @sin(self.azimuth) + self.target_z,
        };
    }

    pub fn getCameraData(self: *const Self, width: u32, height: u32) CameraData {
        const pos = self.position();

        // Calculate view vectors
        const forward_x = self.target_x - pos.x;
        const forward_y = self.target_y - pos.y;
        const forward_z = self.target_z - pos.z;
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
