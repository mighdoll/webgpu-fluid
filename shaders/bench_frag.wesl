// Jacobi pattern: ONE bind group (fair comparison with compute)
// Fragment uses 2 bindings, compute uses 3 (adds storage output)
@group(0) @binding(0) var divergence : texture_2d<f32>;
@group(0) @binding(1) var pressure : texture_2d<f32>;

@fragment
fn main(@builtin(position) coords: vec4<f32>) -> @location(0) f32 {
    let c = vec2<i32>(coords.xy);
    let dims = vec2<i32>(textureDimensions(pressure));
    let L = textureLoad(pressure, clamp(c - vec2(1, 0), vec2(0), dims - 1), 0).x;
    let R = textureLoad(pressure, clamp(c + vec2(1, 0), vec2(0), dims - 1), 0).x;
    let B = textureLoad(pressure, clamp(c + vec2(0, 1), vec2(0), dims - 1), 0).x;
    let T = textureLoad(pressure, clamp(c - vec2(0, 1), vec2(0), dims - 1), 0).x;
    let bC = textureLoad(divergence, c, 0).x;
    return (L + R + B + T - bC) / 4.0;
}
