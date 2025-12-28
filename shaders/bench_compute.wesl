@group(0) @binding(0) var input : texture_2d<f32>;
@group(0) @binding(1) var output : texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let coords = vec2<i32>(id.xy);
    let dims = vec2<i32>(textureDimensions(input));
    if (coords.x >= dims.x || coords.y >= dims.y) {
        return;
    }
    textureStore(output, coords, textureLoad(input, coords, 0));
}
