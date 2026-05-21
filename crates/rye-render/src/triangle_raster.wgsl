// Triangle rasterizer pipeline. Per-vertex projected position + per-vertex color; fragment
// shader passes color through. Depth-test / depth-write are configured on the host pipeline
// when a depth attachment is enabled, not in WGSL.
//
// Camera uniform matches `TriangleRasterUniforms` on the host side (Rust); the mat4x4 std140
// layout puts the matrix at offset 0 with 16-byte alignment, no padding needed.

struct CameraUniform {
    view_projection: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    out.clip_pos = camera.view_projection * vec4<f32>(position, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
