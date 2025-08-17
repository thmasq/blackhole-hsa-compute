// Vertex shader - simple fullscreen quad
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    // Generate a fullscreen triangle pair
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0)
    );
    
    return vec4<f32>(pos[vertex_index], 0.0, 1.0);
}

// Fragment shader - display texture
@group(0) @binding(0) var frame_texture: texture_2d<f32>;
@group(0) @binding(1) var frame_sampler: sampler;

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    // Convert fragment position to UV coordinates
    let dimensions = textureDimensions(frame_texture);
    let uv = position.xy / vec2<f32>(f32(dimensions.x), f32(dimensions.y));
    
    // Sample the texture
    return textureSample(frame_texture, frame_sampler, uv);
}
