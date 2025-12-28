// Jacobi pattern: ONE bind group with 3 bindings (like original)
@group(0) @binding(0) var divergence : texture_2d<f32>;
@group(0) @binding(1) var pressure : texture_2d<f32>;
@group(0) @binding(2) var new_pressure : texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = vec2<i32>(id.xy);
    let dims = vec2<i32>(textureDimensions(divergence));
    if (c.x >= dims.x || c.y >= dims.y) {
        return;
    }
    let L = textureLoad(pressure, clamp(c - vec2(1, 0), vec2(0), dims - 1), 0).x;
    let R = textureLoad(pressure, clamp(c + vec2(1, 0), vec2(0), dims - 1), 0).x;
    let B = textureLoad(pressure, clamp(c + vec2(0, 1), vec2(0), dims - 1), 0).x;
    let T = textureLoad(pressure, clamp(c - vec2(0, 1), vec2(0), dims - 1), 0).x;
    let bC = textureLoad(divergence, c, 0).x;
    textureStore(new_pressure, c, vec4((L + R + B + T - bC) / 4.0, 0, 0, 1));
}
