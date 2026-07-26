//! `impl PhysicsSpace for EuclideanR2`, 2D Euclidean rigid-body physics.
//!
//! Angular velocity is a scalar ([`Bivector2`]); inertia is the scalar moment.

use glam::Vec2;

use loam_math::{Bivector, Bivector2, EuclideanR2, Iso2};

use crate::body::RigidBody;
use crate::collider::{Collider, ColliderKind};
use crate::integrator::PhysicsSpace;
use crate::narrowphase::Narrowphase;
use crate::response::Contact;

/// Scalar bivector component of `u ∧ v` in R²: `u.x*v.y - u.y*v.x`.
fn cross2d(u: Vec2, v: Vec2) -> f32 {
    u.x * v.y - u.y * v.x
}

/// Inverse moment of inertia; 0 for static or zero-inertia bodies (infinite
/// inertia).
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

    fn wedge(&self, a: Vec2, b: Vec2) -> Bivector2 {
        Bivector2(cross2d(a, b))
    }

    fn velocity_at_point(&self, body: &RigidBody<EuclideanR2>, p: Vec2) -> Vec2 {
        // v(r) = v_lin + ω × r; in 2D ω is scalar and ω × r = (-ω·r.y, ω·r.x).
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
        // τ = r × (j·dir); impulse subtracts from A, adds to B.
        let torque_a = -cross2d(ra, lin);
        let torque_b = cross2d(rb, lin);
        a.angular_velocity = Bivector2(a.angular_velocity.0 + torque_a * inv_i_a);
        b.angular_velocity = Bivector2(b.angular_velocity.0 + torque_b * inv_i_b);
    }
}

/// Moment of inertia for a solid disk: `I = ½·m·r²`.
pub fn disk_inertia(mass: f32, radius: f32) -> f32 {
    0.5 * mass * radius * radius
}

/// Build a dynamic circular body in R².
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

/// Register the 2D Euclidean narrowphase functions (reversed pairs handled by
/// the dispatch table's auto-flip).
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

    use loam_math::Space;
    let d = space.distance(a.position, b.position);
    let combined = ra + rb;
    if d >= combined {
        return None;
    }
    let log = space.log(a.position, b.position);
    let len = log.length();
    let normal = if len > 1e-8 { log / len } else { Vec2::Y };

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
// Polygon-polygon via SAT (Separating Axis Theorem). Axis of minimum overlap
// gives the contact normal and penetration depth. Local vertices must be CCW;
// outward edge normals are `(edge.y, -edge.x) / |edge|`.
// ---------------------------------------------------------------------------

use loam_math::Rotor;

fn world_vertices(local: &[Vec2], pos: Vec2, rot: loam_math::Rotor2) -> Vec<Vec2> {
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

/// Smallest-overlap axis over `sides`'s edge normals; `None` if any axis
/// separates the two polygons.
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

    let mut best = best_axis_from(&va, &vb)?;
    if let Some((n, o)) = best_axis_from(&vb, &va) {
        if o < best.1 {
            best = (n, o);
        }
    } else {
        return None;
    }

    let (mut normal, penetration) = best;

    // `Contact` convention: normal points from A's center toward B's center.
    let ab = b.position - a.position;
    if normal.dot(ab) < 0.0 {
        normal = -normal;
    }

    // Contact-point heuristic: deepest vertex of each polygon along the normal,
    // then whichever lies inside the other (the penetrating vertex in a
    // vertex-face contact). Edge-edge or grazing falls back to the midpoint.
    // Imperfect; a full Sutherland-Hodgman manifold would replace it.
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

/// True if `p` lies inside the convex polygon given by CCW vertices `poly`.
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
        // Center inside the polygon: push out toward the nearest edge along
        // (center - closest), flipped so the normal stays A->B.
        let dir = (center - closest).try_normalize().unwrap_or(Vec2::Y);
        return Some(Contact {
            normal: -dir,
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

/// CCW vertices of a regular n-gon with circumradius `r`; first vertex on +X.
pub fn regular_polygon_vertices(n: u32, r: f32) -> Vec<Vec2> {
    use std::f32::consts::TAU;
    (0..n)
        .map(|k| {
            let theta = TAU * (k as f32) / (n as f32);
            Vec2::new(theta.cos(), theta.sin()) * r
        })
        .collect()
}

/// Centroidal moment of inertia of a solid regular n-gon:
/// `I = (m·r²/6)·(1 + 2·cos²(π/n))`. Limits to the disk `m·r²/2` as n->∞.
pub fn regular_polygon_inertia(mass: f32, n: u32, r: f32) -> f32 {
    use std::f32::consts::PI;
    let c = (PI / n as f32).cos();
    (mass * r * r / 6.0) * (1.0 + 2.0 * c * c)
}

/// Build a dynamic regular n-gon body in R².
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

/// CCW corners of an axis-aligned rectangle centered at origin.
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
    // Centroidal inertia m·(w² + h²)/12 with w,h = 2·half -> m·(hx² + hy²)/3.
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

/// Build a static (infinite-mass) rectangular wall. `half_extents` is
/// (width/2, height/2).
pub fn static_wall(center: Vec2, half_extents: Vec2) -> RigidBody<EuclideanR2> {
    RigidBody::fixed(
        center,
        Collider::Polygon2D {
            vertices: rectangle_vertices(half_extents),
        },
        // Any finite value works; the solver gates angular response on
        // `inv_mass > 0`, so static walls never rotate.
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

        world.step(1.0 / 60.0);

        let body = &world.bodies[id];
        // v_y ≈ -9.8/60 ≈ -0.163 after one tick.
        assert!(body.velocity.y < -0.1 && body.velocity.y > -0.2);
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

        // Unit squares 1.5 apart along X -> x-extents overlap by 0.5.
        let a = aa_box(Vec2::ZERO, Vec2::ONE, 1.0);
        let b = aa_box(Vec2::new(1.5, 0.0), Vec2::ONE, 1.0);

        let c = np.test(&a, &b, &EuclideanR2).expect("should collide");
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

        // Squares (circumradius 1) 1.9 apart: unrotated x-extent ±1 overlaps.
        let a = polygon_body(Vec2::ZERO, Vec2::ZERO, 4, 1.0, 1.0);
        let b = polygon_body(Vec2::new(1.9, 0.0), Vec2::ZERO, 4, 1.0, 1.0);
        assert!(
            np.test(&a, &b, &EuclideanR2).is_some(),
            "unrotated squares at 1.9 should overlap"
        );

        // At 45° B's x-extent shrinks to ±√2/2 ≈ ±0.707, opening a gap.
        let mut b = polygon_body(Vec2::new(1.9, 0.0), Vec2::ZERO, 4, 1.0, 1.0);
        b.orientation = Iso2 {
            rotation: loam_math::Bivector2(std::f32::consts::FRAC_PI_4).exp(),
            translation: Vec2::ZERO,
        };
        assert!(np.test(&a, &b, &EuclideanR2).is_none());
    }

    #[test]
    fn sphere_polygon_edge_contact() {
        let mut np = Narrowphase::<EuclideanR2>::new();
        register_default_narrowphase(&mut np);

        // Square right edge at x=1; sphere r=0.5 at x=1.3 -> penetration 0.2.
        let square = polygon_body(Vec2::ZERO, Vec2::ZERO, 4, 1.0, 1.0);
        let sphere = sphere_body(Vec2::new(1.3, 0.0), Vec2::ZERO, 0.5, 1.0);

        let c = np
            .test(&sphere, &square, &EuclideanR2)
            .expect("should collide");
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
        // Registered (Sphere, Polygon2D); (Polygon2D, Sphere) must flip and
        // negate the normal via the dispatch table.
        let mut np = Narrowphase::<EuclideanR2>::new();
        register_default_narrowphase(&mut np);

        let square = polygon_body(Vec2::ZERO, Vec2::ZERO, 4, 1.0, 1.0);
        let sphere = sphere_body(Vec2::new(1.3, 0.0), Vec2::ZERO, 0.5, 1.0);

        let c = np
            .test(&square, &sphere, &EuclideanR2)
            .expect("should collide");
        // Normal polygon->sphere (A->B) now points +X.
        assert!(c.normal.dot(Vec2::X) > 0.99, "normal: {:?}", c.normal);
    }

    #[test]
    fn off_center_impact_produces_angular_velocity() {
        // A square hit off-center (top-right) must acquire clockwise (negative)
        // spin; without contact torque it would only translate.
        let mut world = World::new(EuclideanR2);
        register_default_narrowphase(&mut world.narrowphase);

        let square_id = world.push_body(aa_box(Vec2::ZERO, Vec2::ONE, 1.0));
        let _sphere_id = world.push_body(sphere_body(
            Vec2::new(0.6, 3.0),
            Vec2::new(0.0, -10.0),
            0.3,
            1.0,
        ));

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
        // Head-on sphere-sphere collision must stay purely linear; angular
        // response must not leak into axis-aligned contacts.
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
        // Polygons dropped on a static floor should rest above its surface.
        let mut world = World::new(EuclideanR2);
        register_default_narrowphase(&mut world.narrowphase);

        // Floor top surface at y = 0.5.
        let floor_top = 0.5;
        world.push_body(static_wall(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.5)));

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

        // Lowest point ≥ floor_top minus slop (one tick of residual gravity
        // penetration is expected).
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

        for _ in 0..30 {
            world.step(1.0 / 60.0);
        }

        // x-velocities reverse sign on an elastic-ish bounce.
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
        // Persistent manifolds + iterative PGS let the bottom box satisfy both
        // its constraints (floor below, box above); without them the stack
        // jitters forever. Capped at N=3: one contact per pair per frame can't
        // resist tipping in a tall stack, and manifolds populate too slowly for
        // fast-loading ones. SAT clipping (Sutherland-Hodgman) is the fix.
        let mut world = World::new(EuclideanR2);
        register_default_narrowphase(&mut world.narrowphase);

        // Floor top surface at y = 0.5.
        let floor_top = 0.5;
        world.push_body(static_wall(Vec2::new(0.0, 0.0), Vec2::new(8.0, 0.5)));

        const N: usize = 3;
        const HALF: f32 = 0.5;
        for i in 0..N {
            let y = floor_top + HALF + i as f32 * (2.0 * HALF + 0.05);
            let mut body = aa_box(Vec2::new(0.0, y), Vec2::splat(HALF), 1.0);
            // Zero restitution; the default 0.2 would micro-bounce forever.
            body.restitution = 0.0;
            world.push_body(body);
        }
        // Impulses propagate one box per iteration through the contact chain.
        world.pgs_iters = 16;

        world.push_field(Box::new(crate::field::Gravity::new(Vec2::new(0.0, -9.8))));

        for _ in 0..300 {
            world.step(1.0 / 60.0);
        }

        // Every box at rest and the stack upright.
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
            // Slop tolerates residual penetration the Baumgarte bias leaves.
            assert!(
                body.position.y - HALF >= floor_top - 0.1,
                "body {idx} sank into floor: y_bottom = {}",
                body.position.y - HALF
            );
        }
    }
}
