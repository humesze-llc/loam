//! `impl PhysicsSpace for EuclideanR2`, 2D Euclidean rigid-body physics.
//!
//! Angular velocity is a scalar ([`Bivector2`]); inertia is the scalar moment of inertia.
//! Orientation integrates by composing a rotor derived from `ω·dt`.

use glam::Vec2;

use rye_math::{Bivector, Bivector2, EuclideanR2, Iso2};

use crate::body::RigidBody;
use crate::collider::{Collider, ColliderKind};
use crate::integrator::PhysicsSpace;
use crate::narrowphase::Narrowphase;
use crate::response::Contact;

/// 2D scalar cross product: `u.x*v.y − u.y*v.x`. The "scalar bivector component" of `u ∧ v`
/// in R².
fn cross2d(u: Vec2, v: Vec2) -> f32 {
    u.x * v.y - u.y * v.x
}

/// Inverse moment-of-inertia, treating static or zero-inertia bodies as having infinite
/// inertia (returns 0).
fn inv_inertia(body: &RigidBody<EuclideanR2>) -> f32 {
    if body.inv_mass > 0.0 && body.inertia > 0.0 {
        1.0 / body.inertia
    } else {
        0.0
    }
}

impl PhysicsSpace for EuclideanR2 {
    type AngVel = Bivector2;
    /// Scalar moment of inertia I about the body's center.
    type Inertia = f32;

    fn integrate_orientation(&self, iso: Iso2, omega: Bivector2, dt: f32) -> Iso2 {
        // θ_new = θ_old + ω·dt. Compose existing rotor with the rotor built from the
        // incremental bivector.
        let delta = (omega * dt).exp();
        Iso2 {
            rotation: iso.rotation * delta,
            translation: iso.translation,
        }
    }

    fn apply_inv_inertia(&self, inertia: f32, torque: Bivector2) -> Bivector2 {
        if inertia > 0.0 {
            torque * (1.0 / inertia)
        } else {
            Bivector2::zero()
        }
    }

    fn velocity_at_point(&self, body: &RigidBody<EuclideanR2>, p: Vec2) -> Vec2 {
        // v(r) = v_lin + ω × r, where in 2D ω is a scalar (the bivector component) and ω × r =
        // (−ω·r.y, ω·r.x).
        let r = p - body.position;
        let w = body.angular_velocity.0;
        body.velocity + Vec2::new(-w * r.y, w * r.x)
    }

    fn effective_mass_inv(
        &self,
        a: &RigidBody<EuclideanR2>,
        b: &RigidBody<EuclideanR2>,
        contact_point: Vec2,
        direction: Vec2,
    ) -> f32 {
        let ra = contact_point - a.position;
        let rb = contact_point - b.position;
        let ra_cross = cross2d(ra, direction);
        let rb_cross = cross2d(rb, direction);
        a.inv_mass
            + b.inv_mass
            + ra_cross * ra_cross * inv_inertia(a)
            + rb_cross * rb_cross * inv_inertia(b)
    }

    fn apply_contact_impulse(
        &self,
        a: &mut RigidBody<EuclideanR2>,
        b: &mut RigidBody<EuclideanR2>,
        contact_point: Vec2,
        direction: Vec2,
        magnitude: f32,
    ) {
        let ra = contact_point - a.position;
        let rb = contact_point - b.position;
        let lin = direction * magnitude;
        a.velocity -= lin * a.inv_mass;
        b.velocity += lin * b.inv_mass;

        let inv_i_a = inv_inertia(a);
        let inv_i_b = inv_inertia(b);
        // τ = r × (j·dir), with sign convention "subtract from A".
        let torque_a = -cross2d(ra, lin);
        let torque_b = cross2d(rb, lin);
        a.angular_velocity = Bivector2(a.angular_velocity.0 + torque_a * inv_i_a);
        b.angular_velocity = Bivector2(b.angular_velocity.0 + torque_b * inv_i_b);
    }
}

/// Moment of inertia for a solid disk of radius `r` and mass `m`: `I = ½·m·r²`. Used as the
/// default for `Collider::Sphere` in 2D.
pub fn disk_inertia(mass: f32, radius: f32) -> f32 {
    0.5 * mass * radius * radius
}

/// Convenience: build a dynamic circular body in R².
pub fn sphere_body(
    position: Vec2,
    velocity: Vec2,
    radius: f32,
    mass: f32,
) -> RigidBody<EuclideanR2> {
    RigidBody::new(
        position,
        velocity,
        Collider::sphere_at_origin(radius),
        mass,
        disk_inertia(mass, radius),
        &EuclideanR2,
    )
}

/// Register all 2D Euclidean narrowphase functions: sphere-sphere, polygon-polygon,
/// sphere-polygon (reversed pair handled by the dispatch table's auto-flip).
pub fn register_default_narrowphase(np: &mut Narrowphase<EuclideanR2>) {
    np.register(ColliderKind::Sphere, ColliderKind::Sphere, sphere_sphere_r2);
    np.register(
        ColliderKind::Polygon2D,
        ColliderKind::Polygon2D,
        polygon_polygon_r2,
    );
    np.register(
        ColliderKind::Sphere,
        ColliderKind::Polygon2D,
        sphere_polygon_r2,
    );
}

fn sphere_sphere_r2(
    a: &RigidBody<EuclideanR2>,
    b: &RigidBody<EuclideanR2>,
    space: &EuclideanR2,
) -> Option<Contact<EuclideanR2>> {
    let Collider::Sphere { radius: ra, .. } = a.collider else {
        return None;
    };
    let Collider::Sphere { radius: rb, .. } = b.collider else {
        return None;
    };

    use rye_math::Space;
    let d = space.distance(a.position, b.position);
    let combined = ra + rb;
    if d >= combined {
        return None;
    }
    let log = space.log(a.position, b.position);
    let len = log.length();
    let normal = if len > 1e-8 { log / len } else { Vec2::Y };

    // Contact point: midpoint of the two surface points along the line between centers. For
    // equal-radius spheres this is just the midpoint; for unequal radii it's biased toward the
    // smaller one, which is what impulse response wants.
    let surface_a = a.position + normal * ra;
    let surface_b = b.position - normal * rb;
    let point = (surface_a + surface_b) * 0.5;

    Some(Contact {
        normal,
        point,
        penetration: combined - d,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

// ---------------------------------------------------------------------------
// Polygon-polygon via SAT (Separating Axis Theorem).
//
// For two convex polygons, the pair is non-overlapping iff some edge normal of A or B fully
// separates them. When they do overlap, the axis of minimum overlap gives the contact normal
// and penetration depth.
//
// Polygons must have their local vertices in counter-clockwise order; outward edge normals are
// `(edge.y, -edge.x) / |edge|`.
// ---------------------------------------------------------------------------

use rye_math::Rotor;

fn world_vertices(local: &[Vec2], pos: Vec2, rot: rye_math::Rotor2) -> Vec<Vec2> {
    local.iter().map(|&v| rot.apply(v) + pos).collect()
}

fn project_onto(axis: Vec2, verts: &[Vec2]) -> (f32, f32) {
    let first = verts[0].dot(axis);
    let (mut lo, mut hi) = (first, first);
    for &v in &verts[1..] {
        let p = v.dot(axis);
        if p < lo {
            lo = p;
        }
        if p > hi {
            hi = p;
        }
    }
    (lo, hi)
}

/// Best (smallest-overlap) axis found by iterating `sides`'s edge normals and projecting both
/// polygons. Returns `None` when any axis shows separation (i.e. no collision).
fn best_axis_from(sides: &[Vec2], other: &[Vec2]) -> Option<(Vec2, f32)> {
    let n = sides.len();
    let mut best: Option<(Vec2, f32)> = None;
    for i in 0..n {
        let v0 = sides[i];
        let v1 = sides[(i + 1) % n];
        let edge = v1 - v0;
        let len = edge.length();
        if len < 1e-8 {
            continue;
        }
        // Outward normal for CCW winding.
        let normal = Vec2::new(edge.y, -edge.x) / len;
        let (lo_a, hi_a) = project_onto(normal, sides);
        let (lo_b, hi_b) = project_onto(normal, other);
        let overlap = hi_a.min(hi_b) - lo_a.max(lo_b);
        if overlap <= 0.0 {
            return None;
        }
        match best {
            None => best = Some((normal, overlap)),
            Some((_, prev)) if overlap < prev => best = Some((normal, overlap)),
            _ => {}
        }
    }
    best
}

fn polygon_polygon_r2(
    a: &RigidBody<EuclideanR2>,
    b: &RigidBody<EuclideanR2>,
    _space: &EuclideanR2,
) -> Option<Contact<EuclideanR2>> {
    let Collider::Polygon2D { vertices: a_local } = &a.collider else {
        return None;
    };
    let Collider::Polygon2D { vertices: b_local } = &b.collider else {
        return None;
    };
    if a_local.len() < 3 || b_local.len() < 3 {
        return None;
    }

    let va = world_vertices(a_local, a.position, a.orientation.rotation);
    let vb = world_vertices(b_local, b.position, b.orientation.rotation);

    // Take the minimum across both polygons' edge normals. If either polygon's axes find full
    // separation, there is no collision.
    let mut best = best_axis_from(&va, &vb)?;
    if let Some((n, o)) = best_axis_from(&vb, &va) {
        if o < best.1 {
            best = (n, o);
        }
    } else {
        return None;
    }

    let (mut normal, penetration) = best;

    // Ensure the normal points from A's center toward B's center (the `Contact` convention).
    let ab = b.position - a.position;
    if normal.dot(ab) < 0.0 {
        normal = -normal;
    }

    // Contact-point heuristic: find the deepest-penetrating vertex of each polygon (projected
    // along the contact normal), then pick whichever actually lies inside the other polygon.
    // In a vertex-face contact only one side has a penetrating vertex, and that vertex IS the
    // contact point. For edge-edge (both inside) or grazing (neither strictly inside), fall
    // back to the midpoint of the two candidates. Imperfect; replace with a full
    // Sutherland-Hodgman manifold when stability demands it.
    let mut deepest_a = va[0];
    let mut max_proj = va[0].dot(normal);
    for &v in &va[1..] {
        let p = v.dot(normal);
        if p > max_proj {
            max_proj = p;
            deepest_a = v;
        }
    }
    let mut deepest_b = vb[0];
    let mut min_proj = vb[0].dot(normal);
    for &v in &vb[1..] {
        let p = v.dot(normal);
        if p < min_proj {
            min_proj = p;
            deepest_b = v;
        }
    }

    let a_inside_b = point_in_convex_ccw(&vb, deepest_a);
    let b_inside_a = point_in_convex_ccw(&va, deepest_b);
    let point = match (a_inside_b, b_inside_a) {
        (true, false) => deepest_a,
        (false, true) => deepest_b,
        _ => (deepest_a + deepest_b) * 0.5,
    };

    Some(Contact {
        normal,
        point,
        penetration,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

/// True if `p` lies inside the convex polygon given by CCW vertices `poly`. Used by
/// sphere-polygon to detect the "sphere center has tunneled inside" case and flip the normal
/// accordingly.
fn point_in_convex_ccw(poly: &[Vec2], p: Vec2) -> bool {
    for i in 0..poly.len() {
        let v0 = poly[i];
        let v1 = poly[(i + 1) % poly.len()];
        let edge = v1 - v0;
        let outward = Vec2::new(edge.y, -edge.x);
        if (p - v0).dot(outward) > 0.0 {
            return false;
        }
    }
    true
}

/// Closest point on the polygon boundary (edges) to an external point, plus its distance.
fn closest_on_polygon_boundary(poly: &[Vec2], p: Vec2) -> (Vec2, f32) {
    let mut best = poly[0];
    let mut best_d2 = (best - p).length_squared();
    for i in 0..poly.len() {
        let v0 = poly[i];
        let v1 = poly[(i + 1) % poly.len()];
        let edge = v1 - v0;
        let e2 = edge.length_squared();
        let t = if e2 > 1e-12 {
            ((p - v0).dot(edge) / e2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let q = v0 + edge * t;
        let d2 = (q - p).length_squared();
        if d2 < best_d2 {
            best_d2 = d2;
            best = q;
        }
    }
    (best, best_d2.sqrt())
}

fn sphere_polygon_r2(
    a: &RigidBody<EuclideanR2>,
    b: &RigidBody<EuclideanR2>,
    _space: &EuclideanR2,
) -> Option<Contact<EuclideanR2>> {
    let Collider::Sphere { radius, .. } = a.collider else {
        return None;
    };
    let Collider::Polygon2D { vertices: b_local } = &b.collider else {
        return None;
    };
    if b_local.len() < 3 {
        return None;
    }

    let vb = world_vertices(b_local, b.position, b.orientation.rotation);
    let center = a.position;
    let (closest, dist) = closest_on_polygon_boundary(&vb, center);

    if point_in_convex_ccw(&vb, center) {
        // Sphere center is inside the polygon, maximal penetration. Push the sphere out along
        // (center - closest) toward the nearest edge. Normal A->B is from sphere toward polygon
        // = (closest - center) direction, but since the center is inside we flip to push it
        // out.
        let dir = (center - closest).try_normalize().unwrap_or(Vec2::Y);
        return Some(Contact {
            normal: -dir, // from sphere (A) toward polygon (B)
            point: closest,
            penetration: dist + radius,
            restitution: (a.restitution + b.restitution) * 0.5,
        });
    }

    if dist >= radius {
        return None;
    }

    let normal = if dist > 1e-8 {
        (closest - center) / dist
    } else {
        Vec2::Y
    };

    Some(Contact {
        normal,
        point: closest,
        penetration: radius - dist,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

// ---------------------------------------------------------------------------
// Regular polygon builders.
// ---------------------------------------------------------------------------

/// Return CCW-ordered vertices of a regular n-gon with circumradius `r`. First vertex is at
/// angle 0 (on +X axis).
pub fn regular_polygon_vertices(n: u32, r: f32) -> Vec<Vec2> {
    use std::f32::consts::TAU;
    (0..n)
        .map(|k| {
            let theta = TAU * (k as f32) / (n as f32);
            Vec2::new(theta.cos(), theta.sin()) * r
        })
        .collect()
}

/// Moment of inertia of a solid regular n-gon of mass `m` and circumradius `r`
/// about its centroid:
///
/// `I = (m·r²/6) · (1 + 2·cos²(π/n))`
///
/// Reduces to `m·r²/4` for n=3, `m·r²/3` for n=4, and `m·r²/2` in the disk limit as n->∞.
pub fn regular_polygon_inertia(mass: f32, n: u32, r: f32) -> f32 {
    use std::f32::consts::PI;
    let c = (PI / n as f32).cos();
    (mass * r * r / 6.0) * (1.0 + 2.0 * c * c)
}

/// Convenience: build a dynamic regular n-gon body in R².
pub fn polygon_body(
    position: Vec2,
    velocity: Vec2,
    n: u32,
    circumradius: f32,
    mass: f32,
) -> RigidBody<EuclideanR2> {
    RigidBody::new(
        position,
        velocity,
        Collider::Polygon2D {
            vertices: regular_polygon_vertices(n, circumradius),
        },
        mass,
        regular_polygon_inertia(mass, n, circumradius),
        &EuclideanR2,
    )
}

/// CCW-wound corners of an axis-aligned rectangle centered at origin. Matches the winding
/// `polygon_polygon_r2` / `sphere_polygon_r2` expect.
fn rectangle_vertices(half_extents: Vec2) -> Vec<Vec2> {
    let (hx, hy) = (half_extents.x, half_extents.y);
    vec![
        Vec2::new(hx, -hy),
        Vec2::new(hx, hy),
        Vec2::new(-hx, hy),
        Vec2::new(-hx, -hy),
    ]
}

/// Build a dynamic axis-aligned rectangular body.
pub fn rectangle_body(
    center: Vec2,
    velocity: Vec2,
    half_extents: Vec2,
    mass: f32,
) -> RigidBody<EuclideanR2> {
    // Rectangle moment of inertia about its center: m·(w² + h²)/12, where w = 2·half.x, h =
    // 2·half.y, so I = m·(4·hx² + 4·hy²)/12 = m·(hx² + hy²)/3.
    let inertia = mass * (half_extents.x * half_extents.x + half_extents.y * half_extents.y) / 3.0;
    RigidBody::new(
        center,
        velocity,
        Collider::Polygon2D {
            vertices: rectangle_vertices(half_extents),
        },
        mass,
        inertia,
        &EuclideanR2,
    )
}

/// Build a static (infinite-mass) rectangular wall with CCW-wound corners. `half_extents` is
/// (width/2, height/2).
pub fn static_wall(center: Vec2, half_extents: Vec2) -> RigidBody<EuclideanR2> {
    RigidBody::fixed(
        center,
        Collider::Polygon2D {
            vertices: rectangle_vertices(half_extents),
        },
        // Any finite value is fine — the solver gates angular response on `inv_mass > 0`, so
        // static walls never actually rotate.
        1.0,
        &EuclideanR2,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Gravity;
    use crate::world::World;

    #[test]
    fn falling_body_accelerates_under_gravity() {
        let mut world = World::new(EuclideanR2);
        let id = world.push_body(sphere_body(Vec2::new(0.0, 5.0), Vec2::ZERO, 0.5, 1.0));
        world.push_field(Box::new(Gravity::new(Vec2::new(0.0, -9.8))));

        // One tick of dt = 1/60.
        world.step(1.0 / 60.0);

        let body = &world.bodies[id];
        // After one tick: v_y ≈ −9.8/60 ≈ −0.163.
        assert!(body.velocity.y < -0.1 && body.velocity.y > -0.2);
        // Position moved down by v·dt (velocity sampled post-gravity): y ≈ 5 +
        // (−0.163)·(1/60) ≈ 4.9973.
        assert!(body.position.y < 5.0 && body.position.y > 4.99);
    }

    #[test]
    fn static_body_ignores_gravity() {
        let mut world = World::new(EuclideanR2);
        let id = world.push_body(RigidBody::fixed(
            Vec2::new(0.0, 0.0),
            Collider::sphere_at_origin(1.0),
            disk_inertia(0.0, 1.0),
            &EuclideanR2,
        ));
        world.push_field(Box::new(Gravity::new(Vec2::new(0.0, -9.8))));

        for _ in 0..10 {
            world.step(1.0 / 60.0);
        }

        let body = &world.bodies[id];
        assert_eq!(body.position, Vec2::ZERO);
        assert_eq!(body.velocity, Vec2::ZERO);
    }

    #[test]
    fn sphere_sphere_contact_detected() {
        let mut np = Narrowphase::<EuclideanR2>::new();
        register_default_narrowphase(&mut np);

        let a = sphere_body(Vec2::ZERO, Vec2::ZERO, 1.0, 1.0);
        let b = sphere_body(Vec2::new(1.5, 0.0), Vec2::ZERO, 1.0, 1.0);
        let contact = np.test(&a, &b, &EuclideanR2).expect("should collide");
        assert!((contact.normal - Vec2::X).length() < 1e-5);
        assert!((contact.penetration - 0.5).abs() < 1e-5);
    }

    #[test]
    fn separating_spheres_produce_no_contact() {
        let mut np = Narrowphase::<EuclideanR2>::new();
        register_default_narrowphase(&mut np);

        let a = sphere_body(Vec2::ZERO, Vec2::ZERO, 0.4, 1.0);
        let b = sphere_body(Vec2::new(2.0, 0.0), Vec2::ZERO, 0.4, 1.0);
        assert!(np.test(&a, &b, &EuclideanR2).is_none());
    }

    #[test]
    fn regular_polygon_vertices_are_ccw() {
        // CCW square at the origin: cross product of successive edges should be positive.
        let verts = regular_polygon_vertices(4, 1.0);
        assert_eq!(verts.len(), 4);
        for i in 0..4 {
            let e0 = verts[(i + 1) % 4] - verts[i];
            let e1 = verts[(i + 2) % 4] - verts[(i + 1) % 4];
            let cross = e0.x * e1.y - e0.y * e1.x;
            assert!(cross > 0.0, "winding not CCW at edge {i}: cross={cross}");
        }
    }

    /// Test-local wrapper: axis-aligned box at rest.
    fn aa_box(center: Vec2, half: Vec2, mass: f32) -> RigidBody<EuclideanR2> {
        rectangle_body(center, Vec2::ZERO, half, mass)
    }

    #[test]
    fn polygon_polygon_detects_overlap() {
        let mut np = Narrowphase::<EuclideanR2>::new();
        register_default_narrowphase(&mut np);

        // Two axis-aligned unit squares (half-extent 1), centers 1.5 apart along X ->
        // x-extents overlap by 0.5.
        let a = aa_box(Vec2::ZERO, Vec2::ONE, 1.0);
        let b = aa_box(Vec2::new(1.5, 0.0), Vec2::ONE, 1.0);

        let c = np.test(&a, &b, &EuclideanR2).expect("should collide");
        // Normal A->B should point along ±X; minimum overlap 0.5.
        assert!(
            c.normal.dot(Vec2::X).abs() > 0.99,
            "normal not ±X: {:?}",
            c.normal
        );
        assert!(
            c.normal.dot(Vec2::X) > 0.0,
            "normal not A->B: {:?}",
            c.normal
        );
        assert!(
            (c.penetration - 0.5).abs() < 1e-4,
            "penetration: {}",
            c.penetration
        );
    }

    #[test]
    fn polygon_polygon_separating_produces_no_contact() {
        let mut np = Narrowphase::<EuclideanR2>::new();
        register_default_narrowphase(&mut np);

        let a = polygon_body(Vec2::ZERO, Vec2::ZERO, 4, 1.0, 1.0);
        let b = polygon_body(Vec2::new(3.0, 0.0), Vec2::ZERO, 4, 1.0, 1.0);
        assert!(np.test(&a, &b, &EuclideanR2).is_none());
    }

    #[test]
    fn polygon_rotation_affects_collision() {
        let mut np = Narrowphase::<EuclideanR2>::new();
        register_default_narrowphase(&mut np);

        // Two squares (circumradius 1) centers 1.9 apart. When unrotated their x-extent is ±1,
        // so they overlap.
        let a = polygon_body(Vec2::ZERO, Vec2::ZERO, 4, 1.0, 1.0);
        let b = polygon_body(Vec2::new(1.9, 0.0), Vec2::ZERO, 4, 1.0, 1.0);
        assert!(
            np.test(&a, &b, &EuclideanR2).is_some(),
            "unrotated squares at 1.9 should overlap"
        );

        // Rotate B by 45°. Its x-extent becomes ±√2/2 ≈ ±0.707, so the gap between A's right
        // edge (x=1) and B's left edge (x=1.9−0.707=1.193) is positive -> no collision.
        let mut b = polygon_body(Vec2::new(1.9, 0.0), Vec2::ZERO, 4, 1.0, 1.0);
        b.orientation = Iso2 {
            rotation: rye_math::Bivector2(std::f32::consts::FRAC_PI_4).exp(),
            translation: Vec2::ZERO,
        };
        assert!(np.test(&a, &b, &EuclideanR2).is_none());
    }

    #[test]
    fn sphere_polygon_edge_contact() {
        let mut np = Narrowphase::<EuclideanR2>::new();
        register_default_narrowphase(&mut np);

        // Square circumradius 1 at origin. Its right edge is at x=1. Sphere radius 0.5 at
        // (1.3, 0) -> distance from center to edge is 0.3, penetration = 0.5−0.3 = 0.2.
        let square = polygon_body(Vec2::ZERO, Vec2::ZERO, 4, 1.0, 1.0);
        let sphere = sphere_body(Vec2::new(1.3, 0.0), Vec2::ZERO, 0.5, 1.0);

        let c = np
            .test(&sphere, &square, &EuclideanR2)
            .expect("should collide");
        // Normal sphere->square (A->B): points from sphere toward polygon = −X.
        assert!(c.normal.dot(-Vec2::X) > 0.99, "normal: {:?}", c.normal);
        assert!(
            (c.penetration - 0.2).abs() < 1e-4,
            "penetration: {}",
            c.penetration
        );
    }

    #[test]
    fn sphere_polygon_no_contact_when_separated() {
        let mut np = Narrowphase::<EuclideanR2>::new();
        register_default_narrowphase(&mut np);

        let square = polygon_body(Vec2::ZERO, Vec2::ZERO, 4, 1.0, 1.0);
        let sphere = sphere_body(Vec2::new(2.5, 0.0), Vec2::ZERO, 0.5, 1.0);
        assert!(np.test(&sphere, &square, &EuclideanR2).is_none());
    }

    #[test]
    fn sphere_polygon_reverse_pair_handled() {
        // Registered as (Sphere, Polygon2D). When bodies come in as (Polygon2D, Sphere), the
        // dispatch table should flip and negate the normal.
        let mut np = Narrowphase::<EuclideanR2>::new();
        register_default_narrowphase(&mut np);

        let square = polygon_body(Vec2::ZERO, Vec2::ZERO, 4, 1.0, 1.0);
        let sphere = sphere_body(Vec2::new(1.3, 0.0), Vec2::ZERO, 0.5, 1.0);

        // polygon first, sphere second -> dispatch flips.
        let c = np
            .test(&square, &sphere, &EuclideanR2)
            .expect("should collide");
        // Normal polygon->sphere (A->B): now points from polygon toward sphere = +X.
        assert!(c.normal.dot(Vec2::X) > 0.99, "normal: {:?}", c.normal);
    }

    #[test]
    fn off_center_impact_produces_angular_velocity() {
        // A stationary square hit by a sphere falling onto its top-right corner should acquire
        // clockwise (negative) angular velocity. This is the core bug the angular-response fix
        // addresses: without torque from off-center contact, the square would just translate.
        let mut world = World::new(EuclideanR2);
        register_default_narrowphase(&mut world.narrowphase);

        let square_id = world.push_body(aa_box(Vec2::ZERO, Vec2::ONE, 1.0));
        // Sphere above the right half of the square, moving down fast.
        let _sphere_id = world.push_body(sphere_body(
            Vec2::new(0.6, 3.0),
            Vec2::new(0.0, -10.0),
            0.3,
            1.0,
        ));

        // Simulate until the sphere has landed and resolved.
        for _ in 0..60 {
            world.step(1.0 / 120.0);
        }

        let omega = world.bodies[square_id].angular_velocity.0;
        assert!(
            omega < -0.05,
            "square failed to rotate clockwise from off-center hit: ω = {omega}"
        );
    }

    #[test]
    fn head_on_contact_produces_no_rotation() {
        // Symmetrically aligned sphere-sphere head-on collision should only produce linear
        // response, no spin. Sanity-check that angular response doesn't leak into axis-aligned
        // cases.
        let mut world = World::new(EuclideanR2);
        register_default_narrowphase(&mut world.narrowphase);

        let a = world.push_body(sphere_body(
            Vec2::new(-1.0, 0.0),
            Vec2::new(2.0, 0.0),
            0.5,
            1.0,
        ));
        let b = world.push_body(sphere_body(
            Vec2::new(1.0, 0.0),
            Vec2::new(-2.0, 0.0),
            0.5,
            1.0,
        ));
        for _ in 0..30 {
            world.step(1.0 / 120.0);
        }
        assert!(world.bodies[a].angular_velocity.0.abs() < 0.01);
        assert!(world.bodies[b].angular_velocity.0.abs() < 0.01);
    }

    #[test]
    fn polygons_settle_on_floor_without_penetration() {
        // Drop a ring of polygons onto a static floor; after enough time they should all rest
        // above the floor surface.
        let mut world = World::new(EuclideanR2);
        register_default_narrowphase(&mut world.narrowphase);

        // Floor at y = 0, top surface at y = 0.5.
        let floor_top = 0.5;
        world.push_body(static_wall(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.5)));

        // Drop five hexagons onto the floor.
        for i in 0..5 {
            let x = -2.0 + i as f32 * 1.0;
            world.push_body(polygon_body(
                Vec2::new(x, 4.0 + i as f32 * 0.2),
                Vec2::ZERO,
                6,
                0.4,
                1.0,
            ));
        }

        world.push_field(Box::new(crate::field::Gravity::new(Vec2::new(0.0, -9.8))));

        for _ in 0..240 {
            world.step(1.0 / 60.0);
        }

        // Every dynamic body's lowest point should be ≥ floor_top minus a small slop tolerance
        // (one tick of gravity-driven residual penetration is expected in a non-iterative
        // solver).
        for (idx, body) in world.bodies.iter().enumerate().skip(1) {
            let lowest = body.position.y - 0.4;
            assert!(
                lowest >= floor_top - 0.15,
                "body {idx} tunneled: lowest={lowest}, floor_top={floor_top}"
            );
            assert!(body.position.y.is_finite(), "body {idx} NaN position");
        }
    }

    #[test]
    fn head_on_collision_separates_spheres() {
        let mut world = World::new(EuclideanR2);
        register_default_narrowphase(&mut world.narrowphase);

        world.push_body(sphere_body(
            Vec2::new(-1.0, 0.0),
            Vec2::new(2.0, 0.0),
            0.5,
            1.0,
        ));
        world.push_body(sphere_body(
            Vec2::new(1.0, 0.0),
            Vec2::new(-2.0, 0.0),
            0.5,
            1.0,
        ));

        // Step a few times so they meet.
        for _ in 0..30 {
            world.step(1.0 / 60.0);
        }

        // Velocities should have their sign reversed in the x direction for an elastic-ish
        // bounce.
        assert!(
            world.bodies[0].velocity.x < 0.0,
            "body 0 should bounce back"
        );
        assert!(
            world.bodies[1].velocity.x > 0.0,
            "body 1 should bounce back"
        );
    }

    #[test]
    fn box_stack_settles_to_rest() {
        // Stack three unit boxes vertically on a static floor and run long enough for them to
        // come to rest. Without persistent manifolds + iterative PGS this stack jitters
        // indefinitely: single-contact-per-pair single-pass resolution can't satisfy the bottom
        // box's two simultaneous constraints (floor below, box above) in one impulse
        // application. With manifolds + PGS, both constraints are visible to the solver each
        // iteration and the stack settles to rest.
        //
        // Capped at 3 because tall polygon stacks need SAT manifold clipping
        // (Sutherland-Hodgman edge-to-face) to produce two corner contacts per pair per frame,
        // which is what gives a stack torque resistance against tipping. Today's narrowphase
        // returns one contact per pair per frame; persistent manifolds accumulate corner
        // contacts over frames as the stack settles, but for fast-loading tall stacks the top
        // boxes can drift off before manifolds populate. SAT clipping (Sutherland-Hodgman
        // edge-to-face) is on the follow-up list.
        let mut world = World::new(EuclideanR2);
        register_default_narrowphase(&mut world.narrowphase);

        // Floor at y = 0, top surface at y = 0.5.
        let floor_top = 0.5;
        world.push_body(static_wall(Vec2::new(0.0, 0.0), Vec2::new(8.0, 0.5)));

        const N: usize = 3;
        const HALF: f32 = 0.5;
        for i in 0..N {
            let y = floor_top + HALF + i as f32 * (2.0 * HALF + 0.05);
            let mut body = aa_box(Vec2::new(0.0, y), Vec2::splat(HALF), 1.0);
            // Stacking demos run with zero restitution. Default 0.2 would micro-bounce every
            // contact and prevent the stack from ever settling.
            body.restitution = 0.0;
            world.push_body(body);
        }
        // Tall stacks need more iterations than the default to converge — impulses propagate
        // one box per iteration through the chain of contacts.
        world.pgs_iters = 16;

        world.push_field(Box::new(crate::field::Gravity::new(Vec2::new(0.0, -9.8))));

        // Run for 5 seconds at 60 Hz, plenty of time to settle.
        for _ in 0..300 {
            world.step(1.0 / 60.0);
        }

        // After settling: every box is at rest (low linear and angular velocity), and the stack
        // hasn't fallen over (x-positions close to original).
        for (idx, body) in world.bodies.iter().enumerate().skip(1) {
            assert!(
                body.position.is_finite() && body.velocity.is_finite(),
                "body {idx} not finite"
            );
            assert!(
                body.velocity.length() < 0.5,
                "body {idx} still moving: |v| = {}",
                body.velocity.length()
            );
            assert!(
                body.angular_velocity.0.abs() < 1.0,
                "body {idx} still spinning: ω = {}",
                body.angular_velocity.0
            );
            assert!(
                body.position.x.abs() < 1.0,
                "body {idx} drifted off stack: x = {}",
                body.position.x
            );
            // No body should be below the floor surface (small slop tolerance for residual
            // penetration the Baumgarte bias hasn't fully corrected).
            assert!(
                body.position.y - HALF >= floor_top - 0.1,
                "body {idx} sank into floor: y_bottom = {}",
                body.position.y - HALF
            );
        }
    }
}
