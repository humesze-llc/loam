// Triangle rasterizer pipeline. Per-vertex projected position + color.
//
// Two fragment entry points, host-selected via `FragmentShading`:
// - `fs_flat`: returns the interpolated per-vertex color unmodified.
// - `fs_lambert`: face-normal Lambert from `dpdx`/`dpdy` of world position,
//   so faceted meshes (polychoral cross-section cell caps) need no normal
//   attribute.
//
// Camera uniform matches `TriangleRasterUniforms` on the host side.

struct CameraUniform {
    view_projection: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    // Read only by `fs_lambert` (face normals via `dpdx`/`dpdy`).
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    out.clip_pos = camera.view_projection * vec4<f32>(position, 1.0);
    out.color = color;
    out.world_pos = position;
    return out;
}

@fragment
fn fs_flat(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}

// Face-normal Lambert: diffuse only, fixed key light + ambient floor.
// `cross(dpdx, dpdy)` of world position is the exact face normal for a flat
// triangle regardless of winding.
@fragment
fn fs_lambert(in: VsOut) -> @location(0) vec4<f32> {
    let dpdx_p = dpdx(in.world_pos);
    let dpdy_p = dpdy(in.world_pos);
    let n = normalize(cross(dpdx_p, dpdy_p));

    // Upper-front-right so every polychoron has a well-lit face at the
    // default orbit angle.
    let key_dir = normalize(vec3<f32>(0.55, 0.85, 0.45));

    // Two-sided (`abs`, not `max(.,0)`): the cross-section mesh winding is
    // arbitrary, so one-sided Lambert would blacken half the polytope.
    let intensity = abs(dot(n, key_dir));

    let ambient = 0.30;
    let lambert = ambient + (1.0 - ambient) * intensity;

    return vec4<f32>(in.color.rgb * lambert, in.color.a);
}
