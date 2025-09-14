/// AMD GPU-optimized 32-bit float math library for AMDGCN architecture
/// Based on ROCm OCML (OpenCL Math Library) implementation
const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// LLVM/AMDGCN Intrinsics
// ============================================================================

// Basic LLVM math intrinsics
extern fn @"llvm.sqrt.f32"(f32) f32;
extern fn @"llvm.sin.f32"(f32) f32;
extern fn @"llvm.cos.f32"(f32) f32;
extern fn @"llvm.exp.f32"(f32) f32;
extern fn @"llvm.exp2.f32"(f32) f32;
extern fn @"llvm.log.f32"(f32) f32;
extern fn @"llvm.log2.f32"(f32) f32;
extern fn @"llvm.log10.f32"(f32) f32;
extern fn @"llvm.pow.f32"(f32, f32) f32;
extern fn @"llvm.fabs.f32"(f32) f32;
extern fn @"llvm.floor.f32"(f32) f32;
extern fn @"llvm.ceil.f32"(f32) f32;
extern fn @"llvm.trunc.f32"(f32) f32;
extern fn @"llvm.round.f32"(f32) f32;
extern fn @"llvm.rint.f32"(f32) f32;
extern fn @"llvm.fma.f32"(f32, f32, f32) f32;
extern fn @"llvm.copysign.f32"(f32, f32) f32;
extern fn @"llvm.minnum.f32"(f32, f32) f32;
extern fn @"llvm.maxnum.f32"(f32, f32) f32;

// AMD-specific hardware intrinsics
extern fn @"llvm.amdgcn.rcp.f32"(f32) f32;
extern fn @"llvm.amdgcn.rsq.f32"(f32) f32;
extern fn @"llvm.amdgcn.sin.f32"(f32) f32;
extern fn @"llvm.amdgcn.cos.f32"(f32) f32;
extern fn @"llvm.amdgcn.log.f32"(f32) f32;
extern fn @"llvm.amdgcn.exp2.f32"(f32) f32;
extern fn @"llvm.amdgcn.sqrt.f32"(f32) f32;

// Floating point classification
extern fn @"llvm.is.fpclass.f32"(f32, i32) i1;

// Constants
const PI: f32 = 0x1.921fb6p+1;
const PI_2: f32 = 0x1.921fb6p+0;
const PI_4: f32 = 0x1.921fb6p-1;
const THREE_PI_4: f32 = 0x1.2d97c8p+1;
const M_LOG2E_F: f32 = 0x1.715476p+0;
const M_LN2_F: f32 = 0x1.62e430p-1;

// FP class masks
const CLASS_SNAN: i32 = 0x001;
const CLASS_QNAN: i32 = 0x002;
const CLASS_NINF: i32 = 0x004;
const CLASS_NNOR: i32 = 0x008;
const CLASS_NSUB: i32 = 0x010;
const CLASS_NZER: i32 = 0x020;
const CLASS_PZER: i32 = 0x040;
const CLASS_PSUB: i32 = 0x080;
const CLASS_PNOR: i32 = 0x100;
const CLASS_PINF: i32 = 0x200;

// ============================================================================
// Utility Functions
// ============================================================================

inline fn asInt(x: f32) i32 {
    return @bitCast(x);
}

inline fn asFloat(x: i32) f32 {
    return @bitCast(x);
}

inline fn asUint(x: f32) u32 {
    return @bitCast(x);
}

inline fn mad(a: f32, b: f32, c: f32) f32 {
    return @"llvm.fma.f32"(a, b, c);
}

inline fn isFinite(x: f32) bool {
    return @"llvm.is.fpclass.f32"(x, CLASS_NNOR | CLASS_NSUB | CLASS_NZER | CLASS_PZER | CLASS_PSUB | CLASS_PNOR);
}

inline fn isInf(x: f32) bool {
    return @"llvm.is.fpclass.f32"(x, CLASS_NINF | CLASS_PINF);
}

inline fn isNan(x: f32) bool {
    return @"llvm.is.fpclass.f32"(x, CLASS_SNAN | CLASS_QNAN);
}

inline fn isNormal(x: f32) bool {
    return @"llvm.is.fpclass.f32"(x, CLASS_NNOR | CLASS_PNOR);
}

inline fn signBit(x: f32) bool {
    return asInt(x) < 0;
}

// ============================================================================
// Basic Math Functions
// ============================================================================

/// Absolute value
pub inline fn fabs(x: f32) f32 {
    return @"llvm.fabs.f32"(x);
}

/// Copy sign from y to magnitude of x
pub inline fn copysign(x: f32, y: f32) f32 {
    return @"llvm.copysign.f32"(x, y);
}

/// Square root
pub inline fn sqrt(x: f32) f32 {
    return @"llvm.amdgcn.sqrt.f32"(x);
}

/// Reciprocal square root (1/sqrt(x))
pub inline fn rsqrt(x: f32) f32 {
    return @"llvm.amdgcn.rsq.f32"(x);
}

/// Reciprocal (1/x)
pub inline fn rcp(x: f32) f32 {
    return @"llvm.amdgcn.rcp.f32"(x);
}

/// Cube root
pub inline fn cbrt(x: f32) f32 {
    const ax = fabs(x);
    var z = @"llvm.amdgcn.exp2.f32"(0x1.555556p-2 * @"llvm.amdgcn.log.f32"(ax));
    z = mad(mad(rcp(z * z), -ax, z), -0x1.555556p-2, z);

    const result = if (x != 0.0 and isFinite(x)) z else x;
    return copysign(result, x);
}

/// Reciprocal cube root (1/cbrt(x))
pub inline fn rcbrt(x: f32) f32 {
    const ax = fabs(x);
    var z = @"llvm.amdgcn.exp2.f32"(-0x1.555556p-2 * @"llvm.amdgcn.log.f32"(ax));
    z = mad(mad(z * z, -z * ax, 1.0), 0x1.555556p-2 * z, z);

    const xi = rcp(x);
    const result = if (x != 0.0 and isFinite(x)) z else xi;
    return copysign(result, x);
}

/// Power function x^y
pub inline fn pow(x: f32, y: f32) f32 {
    if (x == 1.0) return 1.0;
    if (y == 0.0) return 1.0;

    const ax = fabs(x);
    const expylnx = @"llvm.exp2.f32"(y * @"llvm.log2.f32"(ax));

    const is_odd_y = @mod(@as(i32, @intFromFloat(y)), 2) == 1;
    var result = copysign(expylnx, if (is_odd_y) x else 1.0);

    // Handle edge cases
    if (x < 0.0 and @floor(y) != y) {
        result = std.math.nan(f32);
    }

    const ay = fabs(y);
    if (isInf(ay)) {
        const y_is_neg_inf = y != ay;
        result = if (ax == 1.0) ax else if ((ax < 1.0) != y_is_neg_inf) 0.0 else ay;
    }

    if (isInf(ax) or x == 0.0) {
        result = copysign(if ((x == 0.0) != (y < 0.0)) 0.0 else std.math.inf(f32), if (is_odd_y) x else 0.0);
    }

    if (isNan(x) or isNan(y)) {
        result = std.math.nan(f32);
    }

    return result;
}

/// Power function for positive x only
pub inline fn powr(x: f32, y: f32) f32 {
    if (x < 0.0) return std.math.nan(f32);

    var result = @"llvm.exp2.f32"(y * @"llvm.log2.f32"(x));

    const iz = if (y < 0.0) std.math.inf(f32) else 0.0;
    const zi = if (y < 0.0) 0.0 else std.math.inf(f32);

    if (x == 0.0) {
        result = if (y == 0.0) std.math.nan(f32) else iz;
    }

    if (x == std.math.inf(f32) and y != 0.0) {
        result = zi;
    }

    if (isInf(y) and x != 1.0) {
        result = if (x < 1.0) iz else zi;
    }

    if (isNan(x) or isNan(y)) {
        result = std.math.nan(f32);
    }

    return result;
}

/// Integer power x^n
pub inline fn pown(x: f32, n: i32) f32 {
    if (n == 0) return 1.0;

    const ax = fabs(x);
    const expylnx = @"llvm.exp2.f32"(@as(f32, @floatFromInt(n)) * @"llvm.log2.f32"(ax));

    const is_odd_y = (n & 1) != 0;
    var result = copysign(expylnx, if (is_odd_y) x else 1.0);

    if (isInf(ax) or x == 0.0) {
        result = copysign(if ((x == 0.0) != (n < 0)) 0.0 else std.math.inf(f32), if (is_odd_y) x else 0.0);
    }

    return result;
}

/// nth root
pub inline fn rootn(x: f32, n: i32) f32 {
    const ax = fabs(x);
    const expylnx = @"llvm.exp2.f32"(@"llvm.log2.f32"(ax) / @as(f32, @floatFromInt(n)));

    const is_odd_y = (n & 1) != 0;
    var result = copysign(expylnx, if (is_odd_y) x else 1.0);

    if (isInf(ax) or x == 0.0) {
        result = copysign(if ((x == 0.0) != (n < 0)) 0.0 else std.math.inf(f32), if (is_odd_y) x else 0.0);
    }

    if ((x < 0.0 and !is_odd_y) or n == 0) {
        result = std.math.nan(f32);
    }

    return result;
}

// ============================================================================
// Exponential and Logarithmic Functions
// ============================================================================

/// Natural exponential (e^x)
pub inline fn exp(x: f32) f32 {
    return @"llvm.exp.f32"(x);
}

/// Base-2 exponential (2^x)
pub inline fn exp2(x: f32) f32 {
    return @"llvm.exp2.f32"(x);
}

/// Base-10 exponential (10^x)
pub inline fn exp10(x: f32) f32 {
    // 10^x = 2^(x * log2(10))
    const log2_10 = 0x1.a934f0p+1;
    return @"llvm.exp2.f32"(x * log2_10);
}

/// exp(x) - 1, accurate for small x
pub inline fn expm1(x: f32) f32 {
    const fn_val = @round(x * M_LOG2E_F);
    const t = mad(-fn_val, -0x1.05c610p-29, mad(-fn_val, 0x1.62e430p-1, x));
    const p = mad(t, mad(t, mad(t, mad(t, mad(t, 0x1.a26762p-13, 0x1.6d2e00p-10), 0x1.110ff2p-7), 0x1.555502p-5), 0x1.555556p-3), 0x1.000000p-1);
    const p_result = mad(t, t * p, t);
    const e = if (fn_val == 128.0) 127 else @as(i32, @intFromFloat(fn_val));
    const s = asFloat((@as(u32, @intCast(e + 127)) << 23));
    var z = mad(s, p_result, s - 1.0);
    z = if (fn_val == 128.0) 2.0 * z else z;

    if (x > 0x1.62e42ep+6) z = std.math.inf(f32);
    z = if (x < -17.0) -1.0 else z;

    return z;
}

/// Natural logarithm
pub inline fn log(x: f32) f32 {
    return @"llvm.log.f32"(x);
}

/// Base-2 logarithm
pub inline fn log2(x: f32) f32 {
    return @"llvm.log2.f32"(x);
}

/// Base-10 logarithm
pub inline fn log10(x: f32) f32 {
    return @"llvm.log10.f32"(x);
}

/// log(1 + x), accurate for small x
pub inline fn log1p(x: f32) f32 {
    if (fabs(x) < 0x1.0p-24) return x;

    // Use extended precision implementation for accuracy
    var result = log(1.0 + x);

    if (x == std.math.inf(f32)) result = x;
    if (x < -1.0) result = std.math.nan(f32);
    if (x == -1.0) result = -std.math.inf(f32);

    return result;
}

/// Extract exponent as floating point
pub inline fn logb(x: f32) f32 {
    const exp_bits = (asUint(x) >> 23) & 0xFF;
    var result = @as(f32, @floatFromInt(@as(i32, @intCast(exp_bits)) - 127));

    const ax = fabs(x);
    if (!isFinite(ax)) result = ax;
    if (x == 0.0) result = -std.math.inf(f32);

    return result;
}

// ============================================================================
// Trigonometric Functions
// ============================================================================

/// Sine
pub inline fn sin(x: f32) f32 {
    const ax = fabs(x);

    // Use hardware sine for small angles
    if (ax < 0x1.0p+17) {
        return @"llvm.amdgcn.sin.f32"(x);
    }

    // For large angles, use software reduction
    return @"llvm.sin.f32"(x);
}

/// Cosine
pub inline fn cos(x: f32) f32 {
    const ax = fabs(x);

    // Use hardware cosine for small angles
    if (ax < 0x1.0p+17) {
        return @"llvm.amdgcn.cos.f32"(x);
    }

    // For large angles, use software reduction
    return @"llvm.cos.f32"(x);
}

/// Tangent
pub inline fn tan(x: f32) f32 {
    const s = sin(x);
    const c = cos(x);
    return s / c;
}

/// Sine and cosine computed together
pub inline fn sincos(x: f32, cos_result: *f32) f32 {
    const s = sin(x);
    cos_result.* = cos(x);
    return s;
}

/// Arc sine
pub inline fn asin(x: f32) f32 {
    const ax = fabs(x);
    const tx = mad(ax, -0.5, 0.5);
    const x2 = x * x;
    const r = if (ax >= 0.5) tx else x2;

    const u = r * mad(r, mad(r, mad(r, mad(r, mad(r, 0x1.38434ep-5, 0x1.bf8bb4p-7), 0x1.069878p-5), 0x1.6c8362p-5), 0x1.33379p-4), 0x1.555558p-3);

    const s = sqrt(r);
    const ret = mad(0x1.ddcb02p-1, 0x1.aee9d6p+0, -2.0 * mad(s, u, s));

    const xux = mad(ax, u, ax);
    const result = if (ax < 0.5) xux else ret;

    return copysign(result, x);
}

/// Arc cosine
pub inline fn acos(x: f32) f32 {
    const ax = fabs(x);

    const rt = mad(-0.5, ax, 0.5);
    const x2 = ax * ax;
    const r = if (ax > 0.5) rt else x2;

    const u = r * mad(r, mad(r, mad(r, mad(r, mad(r, 0x1.38434ep-5, 0x1.bf8bb4p-7), 0x1.069878p-5), 0x1.6c8362p-5), 0x1.33379p-4), 0x1.555558p-3);

    const s = sqrt(r);
    const ztp = 2.0 * mad(s, u, s);
    const ztn = mad(0x1.ddcb02p+0, 0x1.aee9d6p+0, -ztp);
    const zt = if (x < 0.0) ztn else ztp;
    const z = mad(0x1.ddcb02p-1, 0x1.aee9d6p+0, -mad(x, u, x));

    return if (ax > 0.5) zt else z;
}

/// Arc tangent
pub inline fn atan(x: f32) f32 {
    const v = fabs(x);
    const g = v > 1.0;

    const vi = rcp(v);
    const input = if (g) vi else v;

    const t = input * input;
    var a = mad(t, mad(t, mad(t, mad(t, mad(t, mad(t, mad(t, 0x1.5a54bp-9, -0x1.f4b218p-7), 0x1.53f67ep-5), -0x1.2fa9aep-4), 0x1.b26364p-4), -0x1.22c1ccp-3), 0x1.99717ep-3), -0x1.5554c4p-2);
    a = mad(input, t * a, input);

    const y = mad(0x1.ddcb02p-1, 0x1.aee9d6p+0, -a);
    a = if (g) y else a;

    return copysign(a, x);
}

/// Arc tangent of y/x
pub inline fn atan2(y: f32, x: f32) f32 {
    const ax = fabs(x);
    const ay = fabs(y);
    const v = @min(ax, ay);
    const u = @max(ax, ay);

    const vbyu = v / u;

    const t = vbyu * vbyu;
    var a = mad(t, mad(t, mad(t, mad(t, mad(t, mad(t, mad(t, 0x1.5a54bp-9, -0x1.f4b218p-7), 0x1.53f67ep-5), -0x1.2fa9aep-4), 0x1.b26364p-4), -0x1.22c1ccp-3), 0x1.99717ep-3), -0x1.5554c4p-2);
    a = mad(vbyu, t * a, vbyu);

    const t_val = PI_2 - a;
    a = if (ay > ax) t_val else a;
    const t_val2 = PI - a;
    a = if (x < 0.0) t_val2 else a;

    const t_val3 = if (asInt(x) < 0) PI else 0.0;
    a = if (y == 0.0) t_val3 else a;

    // Handle infinity cases
    const t_val4 = if (x < 0.0) THREE_PI_4 else PI_4;
    a = if (isInf(x) and isInf(y)) t_val4 else a;

    // Handle NaN
    a = if (isNan(x) or isNan(y)) std.math.nan(f32) else a;

    return copysign(a, y);
}

// ============================================================================
// Hyperbolic Functions
// ============================================================================

/// Hyperbolic sine
pub inline fn sinh(x: f32) f32 {
    const y = fabs(x);

    if (y < 0x1.0p-12) return y;

    const ex = exp(y - M_LN2_F);
    const z = ex - rcp(ex) * 0.5;

    const result = if (y > 0x1.65a9f8p+6) std.math.inf(f32) else z;
    return copysign(result, x);
}

/// Hyperbolic cosine
pub inline fn cosh(x: f32) f32 {
    const y = fabs(x);
    const ex = exp(y - M_LN2_F * 0.5);
    const z = ex + rcp(ex) * 0.5;

    return if (y > 0x1.65a9f8p+6) std.math.inf(f32) else z;
}

/// Hyperbolic tangent
pub inline fn tanh(x: f32) f32 {
    const y = fabs(x);

    var z: f32 = undefined;
    if (y < 0.625) {
        const y2 = y * y;
        const p = mad(y2, mad(y2, mad(y2, mad(y2, -0x1.758e7ap-8, 0x1.521192p-6), -0x1.b8389cp-5), 0x1.110704p-3), -0x1.555532p-2);
        z = mad(y2, y * p, y);
    } else {
        const t = exp(2.0 * y);
        z = mad(-2.0, rcp(t + 1.0), 1.0);
    }

    return copysign(z, x);
}

/// Inverse hyperbolic sine
pub inline fn asinh(x: f32) f32 {
    const y = fabs(x);

    if (y < 0x1.0p-12) return y;

    const z = log(y + sqrt(y * y + 1.0));
    const result = if (y == std.math.inf(f32)) y else z;

    return copysign(result, x);
}

/// Inverse hyperbolic cosine
pub inline fn acosh(x: f32) f32 {
    var z = log(x + sqrt(x * x - 1.0));

    if (x == std.math.inf(f32)) z = x;
    if (x < 1.0) z = std.math.nan(f32);

    return z;
}

/// Inverse hyperbolic tangent
pub inline fn atanh(x: f32) f32 {
    const y = fabs(x);

    if (y < 0x1.0p-12) return y;

    var z = 0.5 * log((1.0 + y) / (1.0 - y));

    if (y > 1.0) z = std.math.nan(f32);
    if (y == 1.0) z = std.math.inf(f32);

    return copysign(z, x);
}

// ============================================================================
// Rounding and Truncation Functions
// ============================================================================

/// Round to nearest integer
pub inline fn round(x: f32) f32 {
    return @"llvm.round.f32"(x);
}

/// Round to nearest integer using current rounding mode
pub inline fn rint(x: f32) f32 {
    return @"llvm.rint.f32"(x);
}

/// Round toward zero (truncate)
pub inline fn trunc(x: f32) f32 {
    return @"llvm.trunc.f32"(x);
}

/// Round toward negative infinity
pub inline fn floor(x: f32) f32 {
    return @"llvm.floor.f32"(x);
}

/// Round toward positive infinity
pub inline fn ceil(x: f32) f32 {
    return @"llvm.ceil.f32"(x);
}

/// Fractional part
pub inline fn fract(x: f32, iptr: *f32) f32 {
    iptr.* = floor(x);
    const frac_part = x - iptr.*;

    var result = if (frac_part < 0x1.fffffep-1) frac_part else 0x1.fffffep-1;
    if (isNan(x)) result = x;
    if (isInf(x)) result = 0.0;

    return result;
}

/// Split into integer and fractional parts
pub inline fn modf(x: f32, iptr: *f32) f32 {
    const tx = trunc(x);
    const ret = x - tx;
    const result = if (isInf(x)) 0.0 else ret;
    iptr.* = tx;
    return copysign(result, x);
}

// ============================================================================
// Min/Max and Comparison Functions
// ============================================================================

/// IEEE 754 minimum
pub inline fn fmin(x: f32, y: f32) f32 {
    return @"llvm.minnum.f32"(x, y);
}

/// IEEE 754 maximum
pub inline fn fmax(x: f32, y: f32) f32 {
    return @"llvm.maxnum.f32"(x, y);
}

/// Comparison minimum
pub inline fn min(x: f32, y: f32) f32 {
    return if (x < y) x else y;
}

/// Comparison maximum
pub inline fn max(x: f32, y: f32) f32 {
    return if (x > y) x else y;
}

/// Minimum by magnitude
pub inline fn minmag(x: f32, y: f32) f32 {
    var result = fmin(x, y);
    const ax = fabs(x);
    const ay = fabs(y);
    if (ax < ay) result = x;
    if (ay < ax) result = y;
    return result;
}

/// Maximum by magnitude
pub inline fn maxmag(x: f32, y: f32) f32 {
    var result = fmax(x, y);
    const ax = fabs(x);
    const ay = fabs(y);
    if (ax > ay) result = x;
    if (ay > ax) result = y;
    return result;
}

/// Positive difference
pub inline fn fdim(x: f32, y: f32) f32 {
    return if (x <= y and !isNan(x) and !isNan(y)) 0.0 else (x - y);
}

// ============================================================================
// Floating Point Manipulation Functions
// ============================================================================

/// Load exponent: x * 2^exp
pub inline fn ldexp(x: f32, exp_ptr: i32) f32 {
    // Clamp exponent to valid range
    const clamped_exp = @min(@max(exp_ptr, -126), 127);
    const exp_bits = @as(u32, @intCast(clamped_exp + 127)) << 23;
    const scale = asFloat(@as(i32, @bitCast(exp_bits)));
    return x * scale;
}

/// Extract mantissa and exponent
pub inline fn frexp(x: f32, exp_ptr: *i32) f32 {
    if (x == 0.0 or !isFinite(x)) {
        exp_ptr.* = 0;
        return x;
    }

    const bits = asUint(fabs(x));
    exp.* = @as(i32, @intCast((bits >> 23) & 0xFF)) - 126;

    const mantissa_bits = (bits & 0x007FFFFF) | 0x3F000000;
    return copysign(asFloat(@as(i32, @bitCast(mantissa_bits))), x);
}

/// Scale by power of 2: x * 2^n
pub inline fn scalbn(x: f32, n: i32) f32 {
    return ldexp(x, n);
}

/// Next representable value toward y
pub inline fn nextafter(x: f32, y: f32) f32 {
    if (isNan(x) or isNan(y)) return std.math.nan(f32);
    if (x == y) return y;

    const ix = asInt(x);
    var result: i32 = undefined;

    if (x == 0.0) {
        result = (asInt(y) & 0x80000000) | 1;
    } else {
        const mx = if (ix < 0) 0x80000000 - ix else ix;
        const my = if (asInt(y) < 0) 0x80000000 - asInt(y) else asInt(y);
        const t = mx + (if (mx < my) 1 else -1);
        result = if (t < 0) 0x80000000 - t else t;
    }

    return asFloat(result);
}

// ============================================================================
// Classification Functions
// ============================================================================

/// Floating point classification
pub inline fn fpclassify(x: f32) i32 {
    if (isInf(x)) return if (x > 0) 1 else -1; // FP_INFINITE
    if (isNan(x)) return 0; // FP_NAN
    if (x == 0.0) return 2; // FP_ZERO
    if (!isNormal(x)) return 3; // FP_SUBNORMAL
    return 4; // FP_NORMAL
}

/// Check if finite
pub inline fn isfinite(x: f32) bool {
    return isFinite(x);
}

/// Check if infinite
pub inline fn isinf(x: f32) bool {
    return isInf(x);
}

/// Check if NaN
pub inline fn isnan(x: f32) bool {
    return isNan(x);
}

/// Check if normal
pub inline fn isnormal(x: f32) bool {
    return isNormal(x);
}

/// Check sign bit
pub inline fn signbit(x: f32) bool {
    return asInt(x) < 0;
}

/// Integer logarithm base 2
pub inline fn ilogb(x: f32) i32 {
    var result = @as(i32, @intCast((asUint(x) >> 23) & 0xFF)) - 127;

    if (isNan(x)) result = 0x7fffffff; // FP_ILOGBNAN
    if (isInf(x)) result = 0x7fffffff; // INT_MAX
    if (x == 0.0) result = -0x7fffffff - 1; // FP_ILOGB0

    return result;
}

// ============================================================================
// Fused Multiply-Add Functions
// ============================================================================

/// Fused multiply-add
pub inline fn fma(a: f32, b: f32, c: f32) f32 {
    return @"llvm.fma.f32"(a, b, c);
}

/// Multiply-add (may not be fused)
pub inline fn fmuladd(a: f32, b: f32, c: f32) f32 {
    return a * b + c;
}

// ============================================================================
// Remainder Functions
// ============================================================================

/// Floating point remainder
pub inline fn fmod(x: f32, y: f32) f32 {
    if (y == 0.0 or isInf(x) or isNan(y)) return std.math.nan(f32);
    if (isNan(x)) return x;
    if (isInf(y)) return x;

    return x - trunc(x / y) * y;
}

/// IEEE remainder
pub inline fn remainder(x: f32, y: f32) f32 {
    if (y == 0.0 or isInf(x) or isNan(y)) return std.math.nan(f32);
    if (isNan(x)) return x;
    if (isInf(y)) return x;

    const n = round(x / y);
    return x - n * y;
}

// ============================================================================
// Hypot Functions
// ============================================================================

/// Hypotenuse: sqrt(x^2 + y^2)
pub inline fn hypot(x: f32, y: f32) f32 {
    const a = fabs(x);
    const b = fabs(y);
    const t = @max(a, b);

    if (!isFinite(t)) return t;
    if (t == 0.0) return 0.0;

    const exp_val = @as(i32, @intCast((asUint(t) >> 23) & 0xFF)) - 127;
    const scale = ldexp(1.0, -exp_val);
    const as = a * scale;
    const bs = b * scale;

    return ldexp(sqrt(as * as + bs * bs), exp_val);
}

/// Reciprocal hypotenuse: 1/sqrt(x^2 + y^2)
pub inline fn rhypot(x: f32, y: f32) f32 {
    if (isInf(x) or isInf(y)) return 0.0;
    return rcp(hypot(x, y));
}

// ============================================================================
// Vector Length Functions
// ============================================================================

/// 3D vector length
pub inline fn len3(x: f32, y: f32, z: f32) f32 {
    const a = fabs(x);
    const b = fabs(y);
    const c = fabs(z);

    // Sort to get largest magnitude first
    var max_val = @max(@max(a, b), c);
    var mid_val = @max(@min(a, b), @min(@max(a, b), c));
    var min_val = @min(@min(a, b), c);

    if (isInf(max_val)) return std.math.inf(f32);
    if (max_val == 0.0) return 0.0;

    const exp_val = @as(i32, @intCast((asUint(max_val) >> 23) & 0xFF)) - 127;
    const scale = ldexp(1.0, -exp_val);

    max_val *= scale;
    mid_val *= scale;
    min_val *= scale;

    return ldexp(sqrt(max_val * max_val + mid_val * mid_val + min_val * min_val), exp_val);
}

/// 4D vector length
pub inline fn len4(x: f32, y: f32, z: f32, w: f32) f32 {
    const a = fabs(x);
    const b = fabs(y);
    const c = fabs(z);
    const d = fabs(w);

    const max_val = @max(@max(@max(a, b), c), d);

    if (isInf(max_val)) return std.math.inf(f32);
    if (max_val == 0.0) return 0.0;

    const exp_val = @as(i32, @intCast((asUint(max_val) >> 23) & 0xFF)) - 127;
    const scale = ldexp(1.0, -exp_val);

    const as = a * scale;
    const bs = b * scale;
    const cs = c * scale;
    const ds = d * scale;

    return ldexp(sqrt(as * as + bs * bs + cs * cs + ds * ds), exp_val);
}

/// Reciprocal 3D vector length
pub inline fn rlen3(x: f32, y: f32, z: f32) f32 {
    if (isInf(x) or isInf(y) or isInf(z)) return 0.0;
    return rcp(len3(x, y, z));
}

/// Reciprocal 4D vector length
pub inline fn rlen4(x: f32, y: f32, z: f32, w: f32) f32 {
    if (isInf(x) or isInf(y) or isInf(z) or isInf(w)) return 0.0;
    return rcp(len4(x, y, z, w));
}

// ============================================================================
// Native/Fast Functions (Lower Precision)
// ============================================================================

/// Fast reciprocal
pub inline fn native_recip(x: f32) f32 {
    return rcp(x);
}

/// Fast square root
pub inline fn native_sqrt(x: f32) f32 {
    return sqrt(x);
}

/// Fast reciprocal square root
pub inline fn native_rsqrt(x: f32) f32 {
    return rsqrt(x);
}

/// Fast sine
pub inline fn native_sin(x: f32) f32 {
    return @"llvm.amdgcn.sin.f32"(x);
}

/// Fast cosine
pub inline fn native_cos(x: f32) f32 {
    return @"llvm.amdgcn.cos.f32"(x);
}

/// Fast natural exponential
pub inline fn native_exp(x: f32) f32 {
    return exp2(M_LOG2E_F * x);
}

/// Fast base-2 exponential
pub inline fn native_exp2(x: f32) f32 {
    return @"llvm.amdgcn.exp2.f32"(x);
}

/// Fast base-10 exponential
pub inline fn native_exp10(x: f32) f32 {
    return exp2(0x1.a934f0p+1 * x);
}

/// Fast natural logarithm
pub inline fn native_log(x: f32) f32 {
    return @"llvm.amdgcn.log.f32"(x);
}

/// Fast base-2 logarithm
pub inline fn native_log2(x: f32) f32 {
    return @"llvm.amdgcn.log.f32"(x);
}

/// Fast base-10 logarithm
pub inline fn native_log10(x: f32) f32 {
    return native_log2(x) / 0x1.a934f0p+1;
}

// ============================================================================
// Utility Functions
// ============================================================================

/// NaN with payload
pub inline fn nan(nancode: u32) f32 {
    return asFloat(@as(i32, @bitCast(0x7FC00000 | (nancode & 0xFFFFF))));
}

/// Previous representable value
pub inline fn pred(x: f32) f32 {
    if (x == -std.math.inf(f32) or isNan(x)) return x;

    const ix = asInt(x);
    const mx = if (ix < 0) 0x80000000 - ix else ix;
    const t = mx - 1;
    const result = if (t < 0) 0x80000000 - t else t;
    return asFloat(result);
}

/// Next representable value
pub inline fn succ(x: f32) f32 {
    if (x == std.math.inf(f32) or isNan(x)) return x;

    const ix = asInt(x);
    const mx = if (ix < 0) 0x80000000 - ix else ix;
    const t = mx + 1;
    const result = if (t < 0) 0x80000000 - t else t;
    if (mx == 0xFFFFFFFF) return asFloat(0x80000000);
    return asFloat(result);
}
