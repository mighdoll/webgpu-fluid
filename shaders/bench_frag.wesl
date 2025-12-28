@group(0) @binding(0) var input : texture_2d<f32>;

@fragment
fn main(@builtin(position) coords: vec4<f32>) -> @location(0) f32 {
    return textureLoad(input, vec2<i32>(coords.xy), 0).x;
}
