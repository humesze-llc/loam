// Triangle rasterizer pipeline. Per-vertex projected position + per-vertex color.
//
// Two fragment entry points; the host picks one at pipeline construction via
// `FragmentShading`:
//
// - `fs_flat`: pass-through, returns the interpolated per-vertex color unmodified. The
//   v1 behavior. Suitable for unlit overlays, debug fills, and meshes that already carry
//   their shading baked into per-vertex color.
// - `fs_lambert`: face-normal Lambert. Computes the normal from screen-space derivatives
//   of the interpolated world-space position (`dpdx`/`dpdy`), so the mesh doesn't need a
//   normal attribute. The geometry is assumed to be faceted (each triangle = one flat
//   face); derivative-based normals are exact for that case. Used for polychoral
//   cross-section cell caps, where the cross-section is a polyhedron and faceted shading
//   is honest to the geometry.
//
// Depth-test / depth-write are configured on the host pipeline when a depth attachment
// is enabled, not in WGSL.
//
// Camera uniform matches `TriangleRasterUniforms` on the host side (Rust); the mat4x4
// std140 layout puts the matrix at offset 0 with 16-byte alignment, no padding needed.

struct CameraUniform {
    view_projection: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    // World-space position, interpolated per-fragment. Only `fs_lambert` reads it (to
    // derive face normals via `dpdx`/`dpdy`); `fs_flat` ignores it. The varying costs
    // ~12 bytes per vertex of interpolator bandwidth and is free for `fs_flat` callers
    // since their VS work doesn't change.
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

// Flat pass-through. The vertex shader's interpolated per-vertex color reaches the
// framebuffer unmodified.
@fragment
fn fs_flat(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}

// Face-normal Lambert with a fixed key light + ambient floor.
//
// Normal comes from `cross(dpdx(p), dpdy(p))` on the interpolated world-space position;
// for a flat triangle this is exactly the face normal regardless of vertex ordering
// (the `normalize` handles the sign). Faceted shading shows cell boundaries as visible
// creases, which matches what a polychoral cross-section actually looks like.
//
// Light direction is hardcoded in world space; tunable visual constants only. Diffuse
// only (no specular). Polychoral caps are flat-faced and specular highlights would
// emphasize that they're rasterized rather than smooth, which we don't want.
@fragment
fn fs_lambert(in: VsOut) -> @location(0) vec4<f32> {
    let dpdx_p = dpdx(in.world_pos);
    let dpdy_p = dpdy(in.world_pos);
    let n = normalize(cross(dpdx_p, dpdy_p));

    // Key light direction (world space, normalised). Comes from upper-front-right;
    // chosen so all 6 polychora have at least one face well-lit from the default
    // camera orbit angle. Not user-configurable at v1.
    let key_dir = normalize(vec3<f32>(0.55, 0.85, 0.45));

    // Two-sided Lambert: `abs(dot)` rather than `max(dot, 0)` because the cross-section
    // mesh is single-sided geometry whose outward face direction is arbitrary (the
    // triangulator doesn't reason about which side of the cell cap faces the inhabitant).
    // Two-sided avoids one half of the polytope going pitch-black depending on which
    // way we happened to wind the triangles.
    let intensity = abs(dot(n, key_dir));

    // Ambient floor keeps shadowed faces visible without washing out the lit ones.
    let ambient = 0.30;
    let lambert = ambient + (1.0 - ambient) * intensity;

    return vec4<f32>(in.color.rgb * lambert, in.color.a);
}
