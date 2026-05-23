//! Bivectors and rotors, the N-dim rotation primitive.
//!
//! A **bivector** represents an oriented plane of rotation plus a magnitude. Its exponential is
//! a **rotor**, which acts on vectors via the sandwich product and composes via Clifford
//! multiplication.
//!
//! This module exposes two traits ([`Bivector`], [`Rotor`]) so physics and other consumers can
//! be written generically over dimension. Concrete implementations ship per-N: [`Bivector2`] /
//! [`Rotor2`], [`Bivector3`] / [`Rotor3`], [`Bivector4`] / [`Rotor4`].
//!
//! Per-N hand-rolled impls (rather than `generic_const_exprs`) because the Clifford
//! multiplication tables are dimension-specific anyway and the closed-form exp/log derivations differ
//! qualitatively (3D is always single-plane; 4D needs invariant decomposition).
//!
//! ## Convention
//!
//! `R = exp(B/2)`, so a bivector of magnitude θ in plane `e_i ∧ e_j` produces a rotor that
//! rotates by θ in that plane, from `e_i` toward `e_j`. Applying a rotor to a vector uses the
//! sandwich product `v' = R̃ · v · R`, implemented here in closed form per dimension.

use std::ops::{Add, Mul};

use glam::{Vec2, Vec3, Vec4};

/// A bivector in some geometric algebra G(N, 0). Represents an oriented plane of rotation ×
/// magnitude; exponentiates to a rotor.
pub trait Bivector: Copy + Add<Output = Self> + Mul<f32, Output = Self> {
    type Rotor: Rotor<Bivector = Self>;

    /// Zero bivector. `zero().exp()` is the identity rotor.
    fn zero() -> Self;

    /// Exponential map: `exp(B) = sum_k B^k / k!`, closed-form per dimension. Rotates by the
    /// bivector's magnitude in its plane.
    fn exp(self) -> Self::Rotor;
}

/// A rotor in some geometric algebra G(N, 0). Elements of Spin(N), unit-norm by construction.
pub trait Rotor: Copy + Mul<Output = Self> {
    type Bivector: Bivector<Rotor = Self>;
    type Vector: Copy;

    /// Identity rotor. Applying to any vector returns it unchanged.
    fn identity() -> Self;

    /// Reverse / conjugate rotor. `R · R.inverse() == identity()` within floating-point error.
    fn inverse(self) -> Self;

    /// Rotate a vector via the sandwich product.
    fn apply(&self, v: Self::Vector) -> Self::Vector;

    /// Logarithm: the bivector whose `exp` is this rotor. Inverse of [`Bivector::exp`] modulo
    /// branch selection (we pick the principal branch, angle in `[−π, π]`).
    fn log(self) -> Self::Bivector;
}

// ---------------------------------------------------------------------------
// 2D: Bivector2 is a single scalar (the e1∧e2 coefficient); Rotor2 is a
// unit complex number (cos(θ/2), sin(θ/2)).
// ---------------------------------------------------------------------------

/// 2D bivector: a scalar coefficient on the single basis plane `e1∧e2`. Represents a rotation
/// angle in radians (from `x` toward `y`).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Bivector2(pub f32);

impl Add for Bivector2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Mul<f32> for Bivector2 {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self(self.0 * k)
    }
}

impl Bivector for Bivector2 {
    type Rotor = Rotor2;

    fn zero() -> Self {
        Self(0.0)
    }

    fn exp(self) -> Rotor2 {
        let half = self.0 * 0.5;
        Rotor2 {
            a: half.cos(),
            b: half.sin(),
        }
    }
}

/// 2D rotor: a unit complex number `a + b·e1e2` with `a² + b² = 1`. `a = cos(θ/2)`,
/// `b = sin(θ/2)` for a rotation of θ.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rotor2 {
    pub a: f32,
    pub b: f32,
}

impl Rotor2 {
    pub const IDENTITY: Self = Self { a: 1.0, b: 0.0 };
}

impl Default for Rotor2 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mul for Rotor2 {
    type Output = Self;
    /// Complex multiplication, equivalent to Clifford product in 2D.
    fn mul(self, rhs: Self) -> Self {
        Self {
            a: self.a * rhs.a - self.b * rhs.b,
            b: self.a * rhs.b + self.b * rhs.a,
        }
    }
}

impl Rotor for Rotor2 {
    type Bivector = Bivector2;
    type Vector = Vec2;

    fn identity() -> Self {
        Self::IDENTITY
    }

    fn inverse(self) -> Self {
        // Reverse = conjugate for unit rotors.
        Self {
            a: self.a,
            b: -self.b,
        }
    }

    fn apply(&self, v: Vec2) -> Vec2 {
        // Sandwich collapses to the standard 2D rotation matrix where cos(θ) = a² − b² and
        // sin(θ) = 2ab.
        let c = self.a * self.a - self.b * self.b;
        let s = 2.0 * self.a * self.b;
        Vec2::new(c * v.x - s * v.y, s * v.x + c * v.y)
    }

    fn log(self) -> Bivector2 {
        // Full angle θ = 2·atan2(b, a); principal branch by construction of atan2.
        Bivector2(2.0 * self.b.atan2(self.a))
    }
}

// ---------------------------------------------------------------------------
// 3D: Bivector3 has three basis planes (e1∧e2, e2∧e3, e3∧e1). Rotor3 is
// the even subalgebra element `s + xy·e12 + yz·e23 + zx·e31`, isomorphic
// to a quaternion but with a sign convention tied to the rotor sandwich
// `v' = R̃·v·R` rather than the quaternion `q·v·q*`.
// ---------------------------------------------------------------------------

/// 3D bivector. The three components are the coefficients on the basis planes `e1∧e2`, `e2∧e3`,
/// `e3∧e1`. Magnitude encodes rotation angle.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Bivector3 {
    /// Coefficient on `e1∧e2`, rotation in the xy plane.
    pub xy: f32,
    /// Coefficient on `e2∧e3`, rotation in the yz plane.
    pub yz: f32,
    /// Coefficient on `e3∧e1`, rotation in the zx plane.
    pub zx: f32,
}

impl Bivector3 {
    pub const ZERO: Self = Self {
        xy: 0.0,
        yz: 0.0,
        zx: 0.0,
    };

    pub fn new(xy: f32, yz: f32, zx: f32) -> Self {
        Self { xy, yz, zx }
    }

    /// Magnitude of the bivector, the rotation angle when used as a rotor generator.
    pub fn magnitude(self) -> f32 {
        (self.xy * self.xy + self.yz * self.yz + self.zx * self.zx).sqrt()
    }
}

impl Add for Bivector3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            xy: self.xy + rhs.xy,
            yz: self.yz + rhs.yz,
            zx: self.zx + rhs.zx,
        }
    }
}

impl Mul<f32> for Bivector3 {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self {
            xy: self.xy * k,
            yz: self.yz * k,
            zx: self.zx * k,
        }
    }
}

impl Bivector for Bivector3 {
    type Rotor = Rotor3;

    fn zero() -> Self {
        Self::ZERO
    }

    /// Exponential map. For a bivector `B` of magnitude `θ`, `exp(B/2) = cos(θ/2) + sin(θ/2)·B̂`.
    /// In 3D every bivector is simple (single plane), so the closed form is direct, no
    /// decomposition needed (unlike 4D).
    fn exp(self) -> Rotor3 {
        let mag_sq = self.xy * self.xy + self.yz * self.yz + self.zx * self.zx;
        // 1e-16 is below the squared-angle below which f32 sin/cos
        // round-trip is dominated by floating-point noise; the linear
        // Taylor approximation below is exact to within f32 precision.
        if mag_sq < 1e-16 {
            // Small-angle: Taylor expand `sin(θ/2)/θ ≈ 1/2`.
            return Rotor3 {
                s: 1.0,
                xy: self.xy * 0.5,
                yz: self.yz * 0.5,
                zx: self.zx * 0.5,
            };
        }
        let mag = mag_sq.sqrt();
        let half = mag * 0.5;
        let c = half.cos();
        let k = half.sin() / mag;
        Rotor3 {
            s: c,
            xy: self.xy * k,
            yz: self.yz * k,
            zx: self.zx * k,
        }
    }
}

/// 3D rotor: scalar part plus a bivector part, unit-norm by construction.
/// `s² + xy² + yz² + zx² = 1` for any rotor produced by `Bivector3::exp` or by composing rotors.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rotor3 {
    pub s: f32,
    pub xy: f32,
    pub yz: f32,
    pub zx: f32,
}

impl Rotor3 {
    pub const IDENTITY: Self = Self {
        s: 1.0,
        xy: 0.0,
        yz: 0.0,
        zx: 0.0,
    };
}

impl Default for Rotor3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mul for Rotor3 {
    type Output = Self;
    /// Geometric product of two rotors. Derived from the Clifford multiplication table for
    /// basis elements of G(3,0).
    fn mul(self, rhs: Self) -> Self {
        let (s1, a1, b1, c1) = (self.s, self.xy, self.yz, self.zx);
        let (s2, a2, b2, c2) = (rhs.s, rhs.xy, rhs.yz, rhs.zx);
        Self {
            s: s1 * s2 - a1 * a2 - b1 * b2 - c1 * c2,
            xy: s1 * a2 + s2 * a1 - b1 * c2 + c1 * b2,
            yz: s1 * b2 + s2 * b1 + a1 * c2 - c1 * a2,
            zx: s1 * c2 + s2 * c1 - a1 * b2 + b1 * a2,
        }
    }
}

impl Rotor for Rotor3 {
    type Bivector = Bivector3;
    type Vector = Vec3;

    fn identity() -> Self {
        Self::IDENTITY
    }

    fn inverse(self) -> Self {
        // Reverse: flip sign of grade-2 part. For unit rotors this is the geometric inverse.
        Self {
            s: self.s,
            xy: -self.xy,
            yz: -self.yz,
            zx: -self.zx,
        }
    }

    /// Apply the rotor to a vector via the sandwich `R̃ · v · R`. Computed in two stages: first
    /// `R̃ · v` (produces vector + trivector), then multiply by `R` (the trivector part cancels
    /// for unit rotors).
    fn apply(&self, v: Vec3) -> Vec3 {
        let (s, a, b, c) = (self.s, self.xy, self.yz, self.zx);
        let (vx, vy, vz) = (v.x, v.y, v.z);

        // R̃·v: vector part (p1, p2, p3) and trivector-coefficient pt.
        let p1 = s * vx - a * vy + c * vz;
        let p2 = s * vy + a * vx - b * vz;
        let p3 = s * vz + b * vy - c * vx;
        let pt = -(a * vz + b * vx + c * vy);

        // (P + pt·I) · R: vector part (trivector cancels for unit R).
        Vec3::new(
            p1 * s - p2 * a + p3 * c - pt * b,
            p2 * s + p1 * a - p3 * b - pt * c,
            p3 * s + p2 * b - p1 * c - pt * a,
        )
    }

    fn log(self) -> Bivector3 {
        // Inverse of `Bivector3::exp`. The bivector part has magnitude sin(θ/2); the scalar is
        // cos(θ/2). θ = 2·atan2(sin, cos).
        let mag_sq = self.xy * self.xy + self.yz * self.yz + self.zx * self.zx;
        if mag_sq < 1e-16 {
            // Near-identity: use linear term (log ≈ 2·bivector_part).
            return Bivector3 {
                xy: self.xy * 2.0,
                yz: self.yz * 2.0,
                zx: self.zx * 2.0,
            };
        }
        let mag = mag_sq.sqrt();
        let theta = 2.0 * mag.atan2(self.s);
        let k = theta / mag;
        Bivector3 {
            xy: self.xy * k,
            yz: self.yz * k,
            zx: self.zx * k,
        }
    }
}

// ---------------------------------------------------------------------------
// 4D: Bivector4 has six basis planes (xy, xz, xw, yz, yw, zw). Rotor4 is
// the even-graded element of G(4,0): one scalar + six bivectors + one
// pseudoscalar, eight components.
//
// The interesting math vs 3D: in 4D a generic bivector is **not simple**;
// it cannot be written as a single wedge `a ∧ b`. Instead it splits
// uniquely into two orthogonal simple bivectors B = B_a + B_b that live
// in complementary 2-planes. Each part exponentiates via the 3D-style
// closed form; the overall rotor is their product. When `B ∧ B = 0` the
// bivector is simple and the decomposition is trivial.
// ---------------------------------------------------------------------------

/// 4D bivector with six components, one per basis plane `e_i ∧ e_j` for `i < j`. Magnitude
/// encodes rotation angle(s).
///
/// Unlike [`Bivector3`], a 4D bivector can describe a **double rotation** (two independent
/// rotation planes at once). The exponential handles this via the invariant decomposition.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Bivector4 {
    pub xy: f32,
    pub xz: f32,
    pub xw: f32,
    pub yz: f32,
    pub yw: f32,
    pub zw: f32,
}

impl Bivector4 {
    pub const ZERO: Self = Self {
        xy: 0.0,
        xz: 0.0,
        xw: 0.0,
        yz: 0.0,
        yw: 0.0,
        zw: 0.0,
    };

    pub fn new(xy: f32, xz: f32, xw: f32, yz: f32, yw: f32, zw: f32) -> Self {
        Self {
            xy,
            xz,
            xw,
            yz,
            yw,
            zw,
        }
    }

    /// Unit basis bivector at index `i` in `Plane4`'s ordering:
    /// `0=xy, 1=xz, 2=xw, 3=yz, 4=yw, 5=zw`. Panics if `i >= 6`. Use
    /// [`Plane4::unit_bivector`] for the type-safe form.
    pub fn basis(i: usize) -> Self {
        let mut c = [0.0_f32; 6];
        c[i] = 1.0;
        Self::new(c[0], c[1], c[2], c[3], c[4], c[5])
    }

    /// `|B|² = Σ α²_ij` over all six components. Equal to the sum of squared eigenvalue
    /// magnitudes: `θ₁² + θ₂²`.
    pub fn magnitude_squared(self) -> f32 {
        self.xy * self.xy
            + self.xz * self.xz
            + self.xw * self.xw
            + self.yz * self.yz
            + self.yw * self.yw
            + self.zw * self.zw
    }

    pub fn magnitude(self) -> f32 {
        self.magnitude_squared().sqrt()
    }

    /// Read the coefficient of a basis plane. The 6 basis bivectors form an orthonormal basis
    /// of the bivector space; this returns the bivector's component along the given basis direction.
    pub fn component(self, plane: Plane4) -> f32 {
        match plane {
            Plane4::Xy => self.xy,
            Plane4::Xz => self.xz,
            Plane4::Xw => self.xw,
            Plane4::Yz => self.yz,
            Plane4::Yw => self.yw,
            Plane4::Zw => self.zw,
        }
    }

    /// Set the coefficient of a basis plane in place. Mirror of [`Self::component`].
    pub fn set_component(&mut self, plane: Plane4, value: f32) {
        match plane {
            Plane4::Xy => self.xy = value,
            Plane4::Xz => self.xz = value,
            Plane4::Xw => self.xw = value,
            Plane4::Yz => self.yz = value,
            Plane4::Yw => self.yw = value,
            Plane4::Zw => self.zw = value,
        }
    }

    /// Inner product treating both bivectors as 6-component vectors over the orthonormal basis
    /// `{xy, xz, xw, yz, yw, zw}`. Equivalent to `-(A * B).scalar_part()` for the Clifford
    /// product when `A * B` is computed in G(4, 0), since the basis bivectors satisfy
    /// `e_ij · e_ij = -1`; the convention used here is the *Euclidean* inner product on the
    /// coefficient vector, which is positive definite. Used to project one bivector onto
    /// another's direction (e.g., the angle a rotor has accumulated along a fixed bivector axis).
    pub fn dot(self, other: Self) -> f32 {
        self.xy * other.xy
            + self.xz * other.xz
            + self.xw * other.xw
            + self.yz * other.yz
            + self.yw * other.yw
            + self.zw * other.zw
    }

    /// Coefficient of the pseudoscalar `I = e1∧e2∧e3∧e4` in the wedge product `B ∧ B`. Computed
    /// from the three complementary-plane pairings: `xy·zw − xz·yw + xw·yz`, times 2. For a
    /// simple bivector this is zero; a nonzero value means `B` is compound and the exponential
    /// requires the invariant decomposition.
    pub fn wedge_self_coeff(self) -> f32 {
        2.0 * (self.xy * self.zw - self.xz * self.yw + self.xw * self.yz)
    }

    /// Wedge product of two 4-vectors: `u ∧ v` as a bivector. Each basis plane coefficient is
    /// the 2×2 determinant of the two components it projects onto, e.g. `xy = u.x·v.y − u.y·v.x`.
    /// Used by physics to build the torque bivector `r ∧ f`.
    pub fn wedge(u: Vec4, v: Vec4) -> Self {
        Self {
            xy: u.x * v.y - u.y * v.x,
            xz: u.x * v.z - u.z * v.x,
            xw: u.x * v.w - u.w * v.x,
            yz: u.y * v.z - u.z * v.y,
            yw: u.y * v.w - u.w * v.y,
            zw: u.z * v.w - u.w * v.z,
        }
    }

    /// Clifford left-contraction `B ⌋ v`, the grade-1 part of the geometric product `B · v`,
    /// with the standard mathematical sign convention. For `B = e_xy` and `v = e_x` this returns
    /// `−e_y` (because `e_xy · e_x = e_x e_y e_x = −e_x e_x e_y = −e_y`).
    ///
    /// **Note for physics callers**: rigid-body dynamics wants `ω × r` with the *opposite* sign,
    /// `e_xy` "applied to" `e_x` should give `+e_y`, since rotating in the +xy plane sends the
    /// +x axis toward +y. Use `rye_physics::euclidean_r4::omega_cross_r` (or just negate the
    /// result of this function) when you want the physics convention. Keeping the math primitive
    /// Clifford-pure means future generic-`N` callers and the inevitable `Bivector5`/`Bivector6`
    /// get consistent semantics across dimensions, instead of a surprise sign flip at one
    /// specific dimension.
    pub fn contract_vec(self, v: Vec4) -> Vec4 {
        Vec4::new(
            self.xy * v.y + self.xz * v.z + self.xw * v.w,
            -self.xy * v.x + self.yz * v.z + self.yw * v.w,
            -self.xz * v.x - self.yz * v.y + self.zw * v.w,
            -self.xw * v.x - self.yw * v.y - self.zw * v.z,
        )
    }

    /// Hodge dual `B* = B · I`. Swaps each plane with its orthogonal
    /// complement (with signs from reordering basis vectors): `xy <-> −zw`,
    /// `xz <-> +yw`, `xw <-> −yz` (and the reverse swaps for the other
    /// three). Used inside the invariant decomposition.
    pub fn dual(self) -> Self {
        Self {
            xy: -self.zw,
            xz: self.yw,
            xw: -self.yz,
            yz: -self.xw,
            yw: self.xz,
            zw: -self.xy,
        }
    }
}

/// One of the six elementary 4D rotation planes; a basis bivector of [`Bivector4`]. Indices
/// and label match `Bivector4`'s field order: `0=xy, 1=xz, 2=xw, 3=yz, 4=yw, 5=zw`.
///
/// The three w-involving planes (`xw`, `yw`, `zw`) pull visible axes into the hidden 4th
/// dimension; the three pure-3D planes (`xy`, `xz`, `yz`) act as ordinary 3D rotations on the
/// 3D cross-section. Sum-of-bivectors composition is **commutative** (vector-space addition),
/// so building `omega = sum_of_planes · rate` is order-independent and the resulting rotor
/// `omega.exp()` depends only on the *set* of active planes, not their order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Plane4 {
    Xy = 0,
    Xz = 1,
    Xw = 2,
    Yz = 3,
    Yw = 4,
    Zw = 5,
}

impl Plane4 {
    /// All six planes in the canonical `Bivector4` field order.
    pub const ALL: [Self; 6] = [Self::Xy, Self::Xz, Self::Xw, Self::Yz, Self::Yw, Self::Zw];

    /// Lowercase two-letter label, e.g. `"xy"`. Stable identifier suitable for UI labels, debug
    /// output, and serialization keys.
    pub fn label(self) -> &'static str {
        match self {
            Self::Xy => "xy",
            Self::Xz => "xz",
            Self::Xw => "xw",
            Self::Yz => "yz",
            Self::Yw => "yw",
            Self::Zw => "zw",
        }
    }

    /// Unit-magnitude basis bivector for this plane.
    pub fn unit_bivector(self) -> Bivector4 {
        Bivector4::basis(self as usize)
    }
}

impl Add for Bivector4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            xy: self.xy + rhs.xy,
            xz: self.xz + rhs.xz,
            xw: self.xw + rhs.xw,
            yz: self.yz + rhs.yz,
            yw: self.yw + rhs.yw,
            zw: self.zw + rhs.zw,
        }
    }
}

impl Mul<f32> for Bivector4 {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self {
            xy: self.xy * k,
            xz: self.xz * k,
            xw: self.xw * k,
            yz: self.yz * k,
            yw: self.yw * k,
            zw: self.zw * k,
        }
    }
}

impl Bivector for Bivector4 {
    type Rotor = Rotor4;

    fn zero() -> Self {
        Self::ZERO
    }

    /// Exponential map via the invariant decomposition.
    ///
    /// The angles `θ₁ ≥ |θ₂| ≥ 0` of the two orthogonal simple parts
    /// `B_a + B_b` solve
    ///
    /// ```text
    ///   θ₁² + θ₂² = |B|²
    ///   2·θ₁·θ₂   = wedge_self_coeff(B)
    /// ```
    ///
    /// Three cases, handled separately to keep f32 stable:
    ///
    /// 1. `|B|² ≈ 0` -> identity rotor.
    /// 2. `δ ≈ 0` (simple bivector, single rotation plane) -> 3D-style
    ///    closed form `cos(|B|/2) + sin(|B|/2)/|B| · B`.
    /// 3. `disc ≈ 0` with `δ ≠ 0` (isoclinic: equal angles in two
    ///    orthogonal planes) -> closed form using `sinc(θ)` so the general-case
    ///    `1/(θ₁²−θ₂²)` singularity never appears.
    /// 4. General case (`θ₁ ≠ θ₂`, both nonzero) -> expansion of
    ///    `exp(B_a/2)·exp(B_b/2)` in the basis `{1, B, B*, I}`.
    fn exp(self) -> Rotor4 {
        let s = self.magnitude_squared();
        if s < 1e-16 {
            return Rotor4::IDENTITY;
        }
        let delta = self.wedge_self_coeff();

        // Simple case: δ = 0 means `B ∧ B = 0`, i.e. the bivector lies
        // in a single plane. 3D-style rotor.
        if delta.abs() < 1e-6 * s.max(1.0) {
            let mag = s.sqrt();
            let half = mag * 0.5;
            let c = half.cos();
            let k = half.sin() / mag;
            return Rotor4 {
                s: c,
                xy: self.xy * k,
                xz: self.xz * k,
                xw: self.xw * k,
                yz: self.yz * k,
                yw: self.yw * k,
                zw: self.zw * k,
                xyzw: 0.0,
            };
        }

        let disc_sq = (s * s - delta * delta).max(0.0);
        let disc = disc_sq.sqrt();

        // Isoclinic case: t₁ ≈ t₂. Both planes rotate by the same
        // angle; direct closed form avoids 0/0 in the general path.
        // At `t₁ = t₂ = θ`, `B_a·B_b = θ²·I` and
        // `exp(B/2) = cos²(θ/2) + (sin(θ)/(2θ))·B + sin²(θ/2)·sign(δ)·I`.
        if disc < 1e-6 * s.max(1.0) {
            let theta_sq = s * 0.5;
            let theta = theta_sq.sqrt();
            let half = theta * 0.5;
            let ch = half.cos();
            let sh = half.sin();
            let sign_i = delta.signum();
            // b_coef = sin(θ)/(2·θ). Use the trig identity
            // sin(θ) = 2·sin(θ/2)·cos(θ/2) and divide by 2·θ.
            // 1e-8 is the angle below which `sin(θ)/(2θ)` rounds away
            // its leading term in f32; substitute the limit `1/2`.
            let b_coef = if theta > 1e-8 {
                (theta.sin()) / (2.0 * theta)
            } else {
                0.5
            };
            return Rotor4 {
                s: ch * ch,
                xy: self.xy * b_coef,
                xz: self.xz * b_coef,
                xw: self.xw * b_coef,
                yz: self.yz * b_coef,
                yw: self.yw * b_coef,
                zw: self.zw * b_coef,
                xyzw: sh * sh * sign_i,
            };
        }

        // General compound case: both θ₁, θ₂ nonzero and distinct.
        // Take positive root for θ₁, signed root for θ₂ matching δ.
        let t1 = ((s + disc) * 0.5).max(0.0).sqrt();
        let t2 = ((s - disc) * 0.5).max(0.0).sqrt() * delta.signum();

        let half1 = t1 * 0.5;
        let half2 = t2 * 0.5;
        let c1 = half1.cos();
        let s1 = half1.sin();
        let c2 = half2.cos();
        let s2 = half2.sin();

        // `s1/t1` is `sinc(t1/2)/2`; safe to divide since t1 > 0 here.
        // `s2/t2` likewise, t2 may be negative but is nonzero.
        let s1_t1 = s1 / t1;
        let s2_t2 = s2 / t2;

        // Expanding `exp(B_a/2)·exp(B_b/2)` with
        //   B_a = ((s + disc)·B + δ·B*) / (2·disc)
        //   B_b = ((disc − s)·B − δ·B*) / (2·disc)
        // yields:
        //   b_coef     = [s1·c2·(s+disc)/t1 − c1·s2·(s−disc)/t2] / (2·disc)
        //   bstar_coef = δ·[s1·c2/t1 − c1·s2/t2] / (2·disc)
        //   pseudo     = s1·s2         (sign carried in t2/s2)
        let b_coef = (s1_t1 * c2 * (s + disc) - c1 * s2_t2 * (s - disc)) / (2.0 * disc);
        let bstar_coef = delta * (s1_t1 * c2 - c1 * s2_t2) / (2.0 * disc);

        let dual = self.dual();
        Rotor4 {
            s: c1 * c2,
            xy: self.xy * b_coef + dual.xy * bstar_coef,
            xz: self.xz * b_coef + dual.xz * bstar_coef,
            xw: self.xw * b_coef + dual.xw * bstar_coef,
            yz: self.yz * b_coef + dual.yz * bstar_coef,
            yw: self.yw * b_coef + dual.yw * bstar_coef,
            zw: self.zw * b_coef + dual.zw * bstar_coef,
            xyzw: s1 * s2,
        }
    }
}

/// 4D rotor: even-graded element of G(4,0). Eight components: scalar, six bivector coefficients,
/// and a pseudoscalar. Unit-norm by construction when produced by [`Bivector4::exp`] or by
/// composing rotors.
#[derive(Copy, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Rotor4 {
    pub s: f32,
    pub xy: f32,
    pub xz: f32,
    pub xw: f32,
    pub yz: f32,
    pub yw: f32,
    pub zw: f32,
    /// Pseudoscalar coefficient on `I = e1∧e2∧e3∧e4`.
    pub xyzw: f32,
}

impl Rotor4 {
    pub const IDENTITY: Self = Self {
        s: 1.0,
        xy: 0.0,
        xz: 0.0,
        xw: 0.0,
        yz: 0.0,
        yw: 0.0,
        zw: 0.0,
        xyzw: 0.0,
    };

    /// Identity rotor packed into the `[s, xy, xz, xw, yz, yw, zw, xyzw]` slot order used by
    /// `From<Rotor4> for [f32; 8]`. Convenience for initializing GPU uniform-buffer rotor
    /// fields without having to spell out the eight-tuple at every call site.
    pub const IDENTITY_SLOT: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    /// Squared norm under the reverse: `<R̃·R>_0`. For a proper rotor this is `1`. Used
    /// internally for renormalization after long chains of compositions.
    pub fn norm_squared(self) -> f32 {
        self.s * self.s
            + self.xy * self.xy
            + self.xz * self.xz
            + self.xw * self.xw
            + self.yz * self.yz
            + self.yw * self.yw
            + self.zw * self.zw
            + self.xyzw * self.xyzw
    }

    /// Renormalize onto the unit manifold of Spin(4). Apply after long integrator runs to
    /// counter f32 drift.
    pub fn normalize(self) -> Self {
        let n = self.norm_squared().sqrt();
        if n > 0.0 {
            let k = 1.0 / n;
            Self {
                s: self.s * k,
                xy: self.xy * k,
                xz: self.xz * k,
                xw: self.xw * k,
                yz: self.yz * k,
                yw: self.yw * k,
                zw: self.zw * k,
                xyzw: self.xyzw * k,
            }
        } else {
            Self::IDENTITY
        }
    }
}

impl Default for Rotor4 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Pack into the `[s, xy, xz, xw, yz, yw, zw, xyzw]` slot order used by GPU uniform buffers
/// (e.g. `rye-render`'s `BodyUniform`). Centralizing this packing convention here means
/// downstream crates and examples never spell out the field order by hand.
impl From<Rotor4> for [f32; 8] {
    fn from(r: Rotor4) -> Self {
        [r.s, r.xy, r.xz, r.xw, r.yz, r.yw, r.zw, r.xyzw]
    }
}

impl Rotor4 {
    /// Materialize this rotor as a 4×4 rotation matrix, columns being the
    /// rotor applied to the canonical basis vectors e₁, e₂, e₃, e₄. Output
    /// layout is `[col0, col1, col2, col3]` (column-major), which matches
    /// both glam's `Mat4` storage and WGSL's `mat4x4<f32>` upload convention
    ///; `bytemuck::cast` directly into a GPU uniform slot without
    /// transposition.
    ///
    /// Cost: four [`Rotor::apply`] calls (~120 flops total). Negligible
    /// relative to a single per-vertex shader-side rotor sandwich, and
    /// keeps the WGSL side a plain `mat4x4<f32> * vec4<f32>` multiply
    /// instead of porting the full Clifford reduction. Use when the host
    /// computes the rotor on CPU (deterministic sim, rotor integration)
    /// and the shader needs to apply it per vertex.
    ///
    /// Textbook reference: any rotation in O(4) can be encoded as four
    /// images of an orthonormal basis; this is the standard "matrix from
    /// rotation operator" construction (e.g. Hestenes, *New Foundations
    /// for Classical Mechanics*, 2nd ed., §2.5).
    pub fn to_mat4(&self) -> [[f32; 4]; 4] {
        let c0 = <Self as Rotor>::apply(self, Vec4::new(1.0, 0.0, 0.0, 0.0));
        let c1 = <Self as Rotor>::apply(self, Vec4::new(0.0, 1.0, 0.0, 0.0));
        let c2 = <Self as Rotor>::apply(self, Vec4::new(0.0, 0.0, 1.0, 0.0));
        let c3 = <Self as Rotor>::apply(self, Vec4::new(0.0, 0.0, 0.0, 1.0));
        [c0.to_array(), c1.to_array(), c2.to_array(), c3.to_array()]
    }
}

impl Mul for Rotor4 {
    type Output = Self;
    /// Geometric product of two 4D rotors, expanded in the even-graded basis
    /// `{1, e12, e13, e14, e23, e24, e34, I}`. The product of each pair of basis elements is
    /// worked out by direct Clifford reduction, e.g. `e12·e13 = −e23` (swap `e2·e1 = −e1·e2`,
    /// then `e1·e1 = 1`), `e12·e34 = I`, `e12·I = −e34`, etc.
    fn mul(self, rhs: Self) -> Self {
        let (a0, a12, a13, a14, a23, a24, a34, a_i) = (
            self.s, self.xy, self.xz, self.xw, self.yz, self.yw, self.zw, self.xyzw,
        );
        let (b0, b12, b13, b14, b23, b24, b34, b_i) = (
            rhs.s, rhs.xy, rhs.xz, rhs.xw, rhs.yz, rhs.yw, rhs.zw, rhs.xyzw,
        );

        // Scalar part: comes from `1·1`, plus `e_ij·e_ij = −1`
        // (six such), plus `I·I = +1` in G(4,0).
        let s = a0 * b0 - a12 * b12 - a13 * b13 - a14 * b14 - a23 * b23 - a24 * b24 - a34 * b34
            + a_i * b_i;

        // Bivector-plane components. Each is `a0·b_ij + a_ij·b0` plus
        // the bivector-from-bivector contributions (two shared-index
        // products) plus the bivector-from-pseudoscalar pieces. The
        // signs come from the duality table:
        //   e12·I = −e34, e13·I = +e24, e14·I = −e23,
        //   e23·I = −e14, e24·I = +e13, e34·I = −e12.
        // (Left-multiplication by I has the same signs since I
        // commutes with every bivector in G(4,0).)
        // Bivector-bivector contributions to `e12`:
        //   e13·e23 = -e12, e23·e13 = +e12, e14·e24 = -e12, e24·e14 = +e12.
        // The previous version had the e14·e24 / e24·e14 pair's signs
        // swapped, which produced visible drift under sustained rotation
        // by non-simple bivectors mixing w-planes (issue #37).
        let xy = a0 * b12 + a12 * b0 - a13 * b23 + a23 * b13 - a14 * b24 + a24 * b14
            - a34 * b_i
            - a_i * b34;
        let xz = a0 * b13 + a13 * b0 + a12 * b23 - a14 * b34 - a23 * b12
            + a34 * b14
            + a24 * b_i
            + a_i * b24;
        let xw = a0 * b14 + a14 * b0 + a12 * b24 + a13 * b34
            - a24 * b12
            - a34 * b13
            - a23 * b_i
            - a_i * b23;
        let yz = a0 * b23 + a23 * b0 - a12 * b13 + a13 * b12 - a24 * b34 + a34 * b24
            - a14 * b_i
            - a_i * b14;
        let yw = a0 * b24 + a24 * b0 - a12 * b14 + a14 * b12 + a23 * b34 - a34 * b23
            + a13 * b_i
            + a_i * b13;
        let zw = a0 * b34 + a34 * b0 - a13 * b14 + a14 * b13 - a23 * b24 + a24 * b23
            - a12 * b_i
            - a_i * b12;

        // Pseudoscalar part: from wedges of disjoint-plane bivectors
        // (`e12·e34 = I`, `e13·e24 = −I`, `e14·e23 = I`, plus the
        // reverse-order copies) and from `a0·b_i + a_i·b0`.
        let xyzw = a0 * b_i + a_i * b0 + a12 * b34 + a34 * b12 - a13 * b24 - a24 * b13
            + a14 * b23
            + a23 * b14;

        Self {
            s,
            xy,
            xz,
            xw,
            yz,
            yw,
            zw,
            xyzw,
        }
    }
}

impl Rotor for Rotor4 {
    type Bivector = Bivector4;
    type Vector = Vec4;

    fn identity() -> Self {
        Self::IDENTITY
    }

    /// Reverse `R̃`: flip the sign of every grade whose `k(k−1)/2 mod 2 = 1`. For a 4D rotor
    /// that's grade 2 (bivectors) only; grade 0 (scalar) and grade 4 (pseudoscalar in G(4,0),
    /// since `4·3/2 = 6` is even) both keep their sign.
    fn inverse(self) -> Self {
        Self {
            s: self.s,
            xy: -self.xy,
            xz: -self.xz,
            xw: -self.xw,
            yz: -self.yz,
            yw: -self.yw,
            zw: -self.zw,
            xyzw: self.xyzw,
        }
    }

    /// Apply the rotor to a 4-vector via the sandwich `R̃ · v · R`.
    ///
    /// Two-stage multivector multiplication. Stage 1 is `R̃ · v` yielding an odd multivector
    /// (1-vector + 3-vector, eight coefficients). Stage 2 multiplies by `R` on the right and
    /// extracts the 1-vector part. Every basis-element product used below is derived by direct
    /// Clifford reduction in G(4,0):
    ///
    /// - `e_ab · e_c`: if `c ∈ {a, b}` gives `±e_k` (bivector-vector collapses to the other
    ///   index, with a sign from anticommuting through); else gives a trivector `e_{abc}` (sorted).
    /// - `I · e_k`: `−e_234, +e_134, −e_124, +e_123` for `k = 1..4`.
    /// - `e_ijk · e_lm` with `{l,m} ⊂ {i,j,k}`: collapses to `±e_{one remaining}` per the
    ///   standard reduction.
    /// - `e_ijk · I`: `−e_l` (with `l` the missing index), signed by the parity of the
    ///   `e_{ijk}·e_l -> I` permutation.
    fn apply(&self, v: Vec4) -> Vec4 {
        let (vx, vy, vz, vw) = (v.x, v.y, v.z, v.w);
        let rs = self.s;
        let rxy = self.xy;
        let rxz = self.xz;
        let rxw = self.xw;
        let ryz = self.yz;
        let ryw = self.yw;
        let rzw = self.zw;
        let r_i = self.xyzw;

        // Stage 1: R̃·v. R̃ has the bivector signs flipped relative to R (grade-2 reverses), but
        // scalar and pseudoscalar stay. Using R's components directly, with the sign flip folded
        // into the formulas:
        let p1 = rs * vx - rxy * vy - rxz * vz - rxw * vw;
        let p2 = rs * vy + rxy * vx - ryz * vz - ryw * vw;
        let p3 = rs * vz + rxz * vx + ryz * vy - rzw * vw;
        let p4 = rs * vw + rxw * vx + ryw * vy + rzw * vz;

        // 3-vector part of R̃·v in basis (e123, e124, e134, e234). Each comes from one of the
        // three bivector contributions whose planes span that trivector, plus a pseudoscalar term
        // from `I·e_k`.
        let t123 = -rxy * vz + rxz * vy - ryz * vx + r_i * vw;
        let t124 = -rxy * vw + rxw * vy - ryw * vx - r_i * vz;
        let t134 = -rxz * vw + rxw * vz - rzw * vx + r_i * vy;
        let t234 = -ryz * vw + ryw * vz - rzw * vy - r_i * vx;

        // Stage 2: (1-vec + 3-vec) · R, extract the 1-vec output.
        //
        // From the 1-vec part `(p) · R`:
        //   e_k coef: rs·p_k + (bivector/vector collapse terms)
        // From the 3-vec part `(t) · R`:
        //   bivector sharing both indices with the trivector collapses
        //   to the remaining e_k; pseudoscalar maps the trivector to
        //   its complementary vector.
        let q1 = rs * p1 - rxy * p2 - rxz * p3 - rxw * p4 - ryz * t123 - ryw * t124 - rzw * t134
            + r_i * t234;
        let q2 = rs * p2 + rxy * p1 - ryz * p3 - ryw * p4 + rxz * t123 + rxw * t124
            - rzw * t234
            - r_i * t134;
        let q3 = rs * p3 + rxz * p1 + ryz * p2 - rzw * p4 - rxy * t123
            + rxw * t134
            + ryw * t234
            + r_i * t124;
        let q4 = rs * p4 + rxw * p1 + ryw * p2 + rzw * p3
            - rxy * t124
            - rxz * t134
            - ryz * t234
            - r_i * t123;

        Vec4::new(q1, q2, q3, q4)
    }

    /// Logarithm: inverse of [`Bivector4::exp`]. Recovers the bivector whose half-exponential is
    /// this rotor. Uses the same invariant decomposition: scalar + pseudoscalar recover the two
    /// half-angles, then the bivector & dual parts give the two planes.
    fn log(self) -> Bivector4 {
        // `c = cos(θ₁/2)·cos(θ₂/2)`, `p = sin(θ₁/2)·sin(θ₂/2)`.
        // Product-to-sum: `c ± p = cos((θ₁ ∓ θ₂)/2)`.
        let c = self.s;
        let p = self.xyzw;
        let hs = (c - p).clamp(-1.0, 1.0).acos(); // (θ₁ + θ₂) / 2
        let hd = (c + p).clamp(-1.0, 1.0).acos(); // (θ₁ − θ₂) / 2
        let t1 = hs + hd;
        let t2 = hs - hd;
        let s_target = t1 * t1 + t2 * t2;
        let delta_target = 2.0 * t1 * t2;

        let br = Bivector4 {
            xy: self.xy,
            xz: self.xz,
            xw: self.xw,
            yz: self.yz,
            yw: self.yw,
            zw: self.zw,
        };
        let br_mag_sq = br.magnitude_squared();
        if br_mag_sq < 1e-16 {
            return Bivector4::ZERO;
        }

        // Simple (single-plane) case: `δ ≈ 0`, i.e. `t₂ ≈ 0`.
        if delta_target.abs() < 1e-6 * s_target.max(1.0) {
            // `R = cos(θ/2) + sin(θ/2)·B̂` => θ = 2·atan2(|B_r|, c).
            let theta = 2.0 * br_mag_sq.sqrt().atan2(c);
            let k = theta / br_mag_sq.sqrt();
            return br * k;
        }

        // Compound case. Reconstruct the exp-forward coefficients
        // `b_coef`, `bstar_coef` from the recovered angles, then
        // invert the 2×2 system tying bivector & dual to `B`, `B*`.
        //
        // For every complementary plane pair (xy-zw, xz-yw, xw-yz) the
        // relationship unifies to
        //   B = (b_coef · R − bstar_coef · R_dual) / (b_coef² − bstar_coef²)
        // regardless of the pair's dual-sign.
        let disc_target = (s_target * s_target - delta_target * delta_target)
            .max(0.0)
            .sqrt();

        // Isoclinic branch within `log`: t₁ ≈ t₂. bstar_coef -> 0, so
        // the inverse simplifies to `B = R / b_coef` and `b_coef` has
        // the isoclinic closed form `sin(θ)/(2·θ)`.
        if disc_target < 1e-6 * s_target.max(1.0) {
            let theta = (s_target * 0.5).sqrt();
            // 1e-8: same sin(θ)/(2θ) cutover as `Rotor4::exp`'s
            // isoclinic branch; below it the `0.5` limit is exact in f32.
            let b_coef = if theta > 1e-8 {
                theta.sin() / (2.0 * theta)
            } else {
                0.5
            };
            // 1e-12: the inverted coefficient would amplify f32 noise
            // beyond the correctness bound. Treat as zero rather than
            // divide.
            if b_coef.abs() < 1e-12 {
                return Bivector4::ZERO;
            }
            return br * (1.0 / b_coef);
        }

        let t1p = ((s_target + disc_target) * 0.5).max(0.0).sqrt();
        let t2p = ((s_target - disc_target) * 0.5).max(0.0).sqrt() * delta_target.signum();
        let half1 = t1p * 0.5;
        let half2 = t2p * 0.5;
        let c1 = half1.cos();
        let s1 = half1.sin();
        let c2 = half2.cos();
        let s2 = half2.sin();
        let s1_t1 = s1 / t1p;
        let s2_t2 = s2 / t2p;

        let b_coef = (s1_t1 * c2 * (s_target + disc_target)
            - c1 * s2_t2 * (s_target - disc_target))
            / (2.0 * disc_target);
        let bstar_coef = delta_target * (s1_t1 * c2 - c1 * s2_t2) / (2.0 * disc_target);

        let det = b_coef * b_coef - bstar_coef * bstar_coef;
        if det.abs() < 1e-12 {
            return Bivector4::ZERO;
        }
        let inv_a = b_coef / det;
        let inv_b = -bstar_coef / det;

        let br_dual = br.dual();
        Bivector4 {
            xy: inv_a * br.xy + inv_b * br_dual.xy,
            xz: inv_a * br.xz + inv_b * br_dual.xz,
            xw: inv_a * br.xw + inv_b * br_dual.xw,
            yz: inv_a * br.yz + inv_b * br_dual.yz,
            yw: inv_a * br.yw + inv_b * br_dual.yw,
            zw: inv_a * br.zw + inv_b * br_dual.zw,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI, TAU};

    fn assert_close(a: f32, b: f32) {
        assert!(
            (a - b).abs() <= 1e-5,
            "expected {a} close to {b} (diff {})",
            (a - b).abs()
        );
    }

    fn assert_vec2_close(a: Vec2, b: Vec2) {
        assert!(
            (a - b).length() <= 1e-5,
            "expected {a:?} close to {b:?} (diff {})",
            (a - b).length()
        );
    }

    #[test]
    fn zero_bivector_exp_is_identity() {
        let r = Bivector2::zero().exp();
        assert_eq!(r, Rotor2::identity());
    }

    #[test]
    fn quarter_turn_rotates_x_to_y() {
        let r = Bivector2(FRAC_PI_2).exp();
        assert_vec2_close(r.apply(Vec2::X), Vec2::Y);
        assert_vec2_close(r.apply(Vec2::Y), -Vec2::X);
    }

    #[test]
    fn half_turn_negates() {
        let r = Bivector2(PI).exp();
        assert_vec2_close(r.apply(Vec2::new(1.0, 2.0)), Vec2::new(-1.0, -2.0));
    }

    #[test]
    fn full_turn_is_identity_up_to_sign() {
        // exp(B·τ/2) is the negative identity rotor (−1 + 0·e12), still
        // acts as identity on vectors (sandwich squares the sign out).
        let r = Bivector2(TAU).exp();
        assert_vec2_close(r.apply(Vec2::X), Vec2::X);
    }

    #[test]
    fn composition_adds_angles() {
        let a = Bivector2(0.3).exp();
        let b = Bivector2(0.5).exp();
        let composed = a * b;
        let direct = Bivector2(0.8).exp();
        assert_close(composed.a, direct.a);
        assert_close(composed.b, direct.b);
    }

    #[test]
    fn inverse_cancels() {
        let r = Bivector2(1.234).exp();
        let id = r * r.inverse();
        assert_close(id.a, 1.0);
        assert_close(id.b, 0.0);
    }

    #[test]
    fn rotor4_to_mat4_agrees_with_apply() {
        // Build a non-trivial 4D rotation (xy + zw simple plane) and check
        // that the matrix columns equal the rotor's image of each basis vector,
        // AND that applying the matrix to a few test vectors matches applying
        // the rotor directly.
        let b = Bivector4 {
            xy: 0.7,
            xz: 0.0,
            xw: 0.0,
            yz: 0.0,
            yw: 0.0,
            zw: 0.3,
        };
        let r: Rotor4 = b.exp();
        let m = r.to_mat4();

        // Column 0 should be R(e1).
        let e1 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let img1 = <Rotor4 as Rotor>::apply(&r, e1);
        assert_close(m[0][0], img1.x);
        assert_close(m[0][1], img1.y);
        assert_close(m[0][2], img1.z);
        assert_close(m[0][3], img1.w);

        // Random-ish vector: matrix-vector product should equal R-apply.
        let v = Vec4::new(0.6, -0.4, 0.2, 0.9);
        // Column-major mat4: result = sum_i col_i * v[i].
        let row = |k: usize| m[0][k] * v.x + m[1][k] * v.y + m[2][k] * v.z + m[3][k] * v.w;
        let by_matrix = Vec4::new(row(0), row(1), row(2), row(3));
        let by_rotor = <Rotor4 as Rotor>::apply(&r, v);
        assert!(
            (by_matrix - by_rotor).length() < 1e-5,
            "matrix application {by_matrix:?} should match rotor application {by_rotor:?}"
        );
    }

    #[test]
    fn apply_preserves_length() {
        let r = Bivector2(0.7).exp();
        let v = Vec2::new(3.0, 4.0);
        assert_close(r.apply(v).length(), v.length());
    }

    #[test]
    fn log_is_inverse_of_exp() {
        for &theta in &[-1.0_f32, -0.1, 0.0, 0.1, 1.0, 3.0] {
            let back = Bivector2(theta).exp().log();
            assert_close(back.0, theta);
        }
    }

    #[test]
    fn scalar_mul_scales_angle() {
        let b = Bivector2(0.5) * 2.0;
        assert_close(b.0, 1.0);
    }

    #[test]
    fn bivector_add_sums_angles() {
        let b = Bivector2(0.3) + Bivector2(0.5);
        assert_close(b.0, 0.8);
    }

    // ---- 3D tests ----

    fn assert_vec3_close(a: Vec3, b: Vec3) {
        assert!((a - b).length() <= 1e-5, "expected {a:?} close to {b:?}");
    }

    #[test]
    fn bivector3_zero_exp_is_identity() {
        let r = Bivector3::ZERO.exp();
        assert_eq!(r, Rotor3::IDENTITY);
    }

    #[test]
    fn rotor3_identity_leaves_vector_unchanged() {
        let v = Vec3::new(1.2, -3.4, 0.5);
        assert_vec3_close(Rotor3::IDENTITY.apply(v), v);
    }

    #[test]
    fn bivector3_xy_quarter_turn_sends_x_to_y() {
        // Rotation in xy-plane by π/2: e1 -> e2.
        let r = Bivector3::new(FRAC_PI_2, 0.0, 0.0).exp();
        assert_vec3_close(r.apply(Vec3::X), Vec3::Y);
        assert_vec3_close(r.apply(Vec3::Y), -Vec3::X);
        // z-axis is unaffected by xy-plane rotation.
        assert_vec3_close(r.apply(Vec3::Z), Vec3::Z);
    }

    #[test]
    fn bivector3_yz_rotation() {
        // Rotation in yz-plane (e2 -> e3) by π/2.
        let r = Bivector3::new(0.0, FRAC_PI_2, 0.0).exp();
        assert_vec3_close(r.apply(Vec3::Y), Vec3::Z);
        assert_vec3_close(r.apply(Vec3::Z), -Vec3::Y);
        assert_vec3_close(r.apply(Vec3::X), Vec3::X);
    }

    #[test]
    fn bivector3_zx_rotation() {
        // Rotation in zx-plane (e3 -> e1) by π/2.
        let r = Bivector3::new(0.0, 0.0, FRAC_PI_2).exp();
        assert_vec3_close(r.apply(Vec3::Z), Vec3::X);
        assert_vec3_close(r.apply(Vec3::X), -Vec3::Z);
        assert_vec3_close(r.apply(Vec3::Y), Vec3::Y);
    }

    #[test]
    fn rotor3_full_turn_is_identity_on_vectors() {
        let r = Bivector3::new(TAU, 0.0, 0.0).exp();
        assert_vec3_close(r.apply(Vec3::new(1.0, 2.0, 3.0)), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn rotor3_inverse_cancels() {
        let r = Bivector3::new(0.7, -0.3, 0.5).exp();
        let id = r * r.inverse();
        assert_close(id.s, 1.0);
        assert_close(id.xy, 0.0);
        assert_close(id.yz, 0.0);
        assert_close(id.zx, 0.0);
    }

    #[test]
    fn rotor3_composition_matches_sequential_apply() {
        // With the `v' = R̃·v·R` sandwich, multiplication order equals application order:
        // `(ra · rb).apply(v) = rb.apply(ra.apply(v))`, i.e. `ra` is applied first, then `rb`.
        let ra = Bivector3::new(0.4, 0.0, 0.0).exp();
        let rb = Bivector3::new(0.0, 0.5, 0.0).exp();
        let composed = ra * rb;
        let v = Vec3::new(0.7, -0.3, 1.1);
        assert_vec3_close(composed.apply(v), rb.apply(ra.apply(v)));
    }

    #[test]
    fn rotor3_preserves_length() {
        let r = Bivector3::new(0.3, 0.4, 0.5).exp();
        for &v in &[Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(1.0, 2.0, -3.0)] {
            assert!(
                (r.apply(v).length() - v.length()).abs() < 1e-5,
                "length not preserved: input {v}, output length {}",
                r.apply(v).length()
            );
        }
    }

    #[test]
    fn rotor3_log_is_inverse_of_exp() {
        for bv in [
            Bivector3::new(0.0, 0.0, 0.0),
            Bivector3::new(0.1, 0.0, 0.0),
            Bivector3::new(0.0, 0.6, 0.0),
            Bivector3::new(-0.3, 0.4, 0.5),
            Bivector3::new(1.0, -0.5, 0.2),
        ] {
            let back = bv.exp().log();
            assert!(
                (back.xy - bv.xy).abs() < 1e-5
                    && (back.yz - bv.yz).abs() < 1e-5
                    && (back.zx - bv.zx).abs() < 1e-5,
                "log∘exp mismatch: {bv:?} -> {back:?}"
            );
        }
    }

    #[test]
    fn rotor3_matches_glam_quat_for_axis_rotation() {
        // Same rotation (axis + angle) computed two ways, Rye's rotor
        // and glam's Quat, must agree on the acted-upon vector. This
        // cross-checks the sign convention.
        use glam::Quat;

        let theta = 0.7;
        let v = Vec3::new(1.0, 2.0, 3.0).normalize();

        // xy-plane rotation <-> rotation about +z axis.
        let rotor = Bivector3::new(theta, 0.0, 0.0).exp();
        let quat = Quat::from_axis_angle(Vec3::Z, theta);
        assert_vec3_close(rotor.apply(v), quat * v);

        // yz-plane rotation <-> rotation about +x axis.
        let rotor = Bivector3::new(0.0, theta, 0.0).exp();
        let quat = Quat::from_axis_angle(Vec3::X, theta);
        assert_vec3_close(rotor.apply(v), quat * v);

        // zx-plane rotation <-> rotation about +y axis.
        let rotor = Bivector3::new(0.0, 0.0, theta).exp();
        let quat = Quat::from_axis_angle(Vec3::Y, theta);
        assert_vec3_close(rotor.apply(v), quat * v);
    }

    // ---- 4D tests ----

    fn assert_vec4_close(a: Vec4, b: Vec4) {
        assert!(
            (a - b).length() <= 1e-4,
            "expected {a:?} close to {b:?} (diff {})",
            (a - b).length()
        );
    }

    fn assert_vec4_close_tol(a: Vec4, b: Vec4, tol: f32) {
        assert!(
            (a - b).length() <= tol,
            "expected {a:?} close to {b:?} (diff {})",
            (a - b).length()
        );
    }

    #[test]
    fn bivector4_zero_exp_is_identity() {
        let r = Bivector4::ZERO.exp();
        assert_eq!(r, Rotor4::IDENTITY);
    }

    /// `component(plane)` reads the plane's coefficient; `set_component` writes it; round-trip
    /// is the identity.
    #[test]
    fn bivector4_component_round_trip() {
        let b = Bivector4::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!(b.component(Plane4::Xy), 1.0);
        assert_eq!(b.component(Plane4::Xz), 2.0);
        assert_eq!(b.component(Plane4::Xw), 3.0);
        assert_eq!(b.component(Plane4::Yz), 4.0);
        assert_eq!(b.component(Plane4::Yw), 5.0);
        assert_eq!(b.component(Plane4::Zw), 6.0);
        let mut c = Bivector4::ZERO;
        for plane in Plane4::ALL {
            c.set_component(plane, b.component(plane));
        }
        assert_eq!(c, b);
    }

    /// `dot` is the Euclidean inner product on the 6-coefficient vector. Sanity checks:
    /// orthogonal basis bivectors give 0; a basis bivector dotted with itself gives 1;
    /// bilinearity holds.
    #[test]
    fn bivector4_dot_inner_product_invariants() {
        let xy = Plane4::Xy.unit_bivector();
        let zw = Plane4::Zw.unit_bivector();
        let xw = Plane4::Xw.unit_bivector();
        assert_eq!(xy.dot(xy), 1.0);
        assert_eq!(xy.dot(zw), 0.0);
        assert_eq!(xy.dot(xw), 0.0);
        // Bilinearity: dot(a, b + c) = dot(a, b) + dot(a, c).
        let a = Bivector4::new(1.0, 0.5, -0.25, 0.75, -1.0, 2.0);
        let b = xy * 3.0;
        let c = xw * 4.0;
        assert!((a.dot(b + c) - (a.dot(b) + a.dot(c))).abs() < 1e-6);
    }

    #[test]
    fn rotor4_identity_leaves_vector_unchanged() {
        let v = Vec4::new(1.2, -3.4, 0.5, 0.7);
        assert_vec4_close(Rotor4::IDENTITY.apply(v), v);
    }

    /// Single-plane rotations in each of the six basis planes behave
    /// like 3D rotations: two coordinates swap under a quarter turn,
    /// the other two are fixed.
    #[test]
    fn bivector4_single_plane_rotations_are_planar() {
        // xy-plane (θ = π/2): x -> y, y -> −x, z / w fixed.
        let r = Bivector4::new(FRAC_PI_2, 0.0, 0.0, 0.0, 0.0, 0.0).exp();
        assert_vec4_close(r.apply(Vec4::X), Vec4::Y);
        assert_vec4_close(r.apply(Vec4::Y), -Vec4::X);
        assert_vec4_close(r.apply(Vec4::Z), Vec4::Z);
        assert_vec4_close(r.apply(Vec4::W), Vec4::W);

        // xz-plane.
        let r = Bivector4::new(0.0, FRAC_PI_2, 0.0, 0.0, 0.0, 0.0).exp();
        assert_vec4_close(r.apply(Vec4::X), Vec4::Z);
        assert_vec4_close(r.apply(Vec4::Z), -Vec4::X);
        assert_vec4_close(r.apply(Vec4::Y), Vec4::Y);
        assert_vec4_close(r.apply(Vec4::W), Vec4::W);

        // xw-plane.
        let r = Bivector4::new(0.0, 0.0, FRAC_PI_2, 0.0, 0.0, 0.0).exp();
        assert_vec4_close(r.apply(Vec4::X), Vec4::W);
        assert_vec4_close(r.apply(Vec4::W), -Vec4::X);

        // yz-plane.
        let r = Bivector4::new(0.0, 0.0, 0.0, FRAC_PI_2, 0.0, 0.0).exp();
        assert_vec4_close(r.apply(Vec4::Y), Vec4::Z);
        assert_vec4_close(r.apply(Vec4::Z), -Vec4::Y);

        // yw-plane.
        let r = Bivector4::new(0.0, 0.0, 0.0, 0.0, FRAC_PI_2, 0.0).exp();
        assert_vec4_close(r.apply(Vec4::Y), Vec4::W);
        assert_vec4_close(r.apply(Vec4::W), -Vec4::Y);

        // zw-plane.
        let r = Bivector4::new(0.0, 0.0, 0.0, 0.0, 0.0, FRAC_PI_2).exp();
        assert_vec4_close(r.apply(Vec4::Z), Vec4::W);
        assert_vec4_close(r.apply(Vec4::W), -Vec4::Z);
    }

    /// A simple bivector (one plane only) has `B ∧ B = 0`, this test locks the wedge-coefficient
    /// helper against that invariant.
    #[test]
    fn simple_bivector_has_zero_wedge_self() {
        for b in [
            Bivector4::new(0.7, 0.0, 0.0, 0.0, 0.0, 0.0),
            Bivector4::new(0.0, 0.5, 0.0, 0.0, 0.0, 0.0),
            Bivector4::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.3),
        ] {
            assert_close(b.wedge_self_coeff(), 0.0);
        }
    }

    /// A "double rotation" uses two complementary planes simultaneously.
    /// Rotating by π/2 in both xy and zw: x -> y, y -> −x, z -> w,
    /// w -> −z. The pseudoscalar component of the rotor is nonzero
    /// here, that's the fingerprint of the compound decomposition.
    #[test]
    fn bivector4_double_rotation_xy_plus_zw() {
        let theta = FRAC_PI_2;
        let r = Bivector4::new(theta, 0.0, 0.0, 0.0, 0.0, theta).exp();
        assert_vec4_close(r.apply(Vec4::X), Vec4::Y);
        assert_vec4_close(r.apply(Vec4::Y), -Vec4::X);
        assert_vec4_close(r.apply(Vec4::Z), Vec4::W);
        assert_vec4_close(r.apply(Vec4::W), -Vec4::Z);
        // Pseudoscalar should be sin(π/4)·sin(π/4) = 0.5.
        assert_close(r.xyzw, 0.5);
    }

    /// An isoclinic rotation (equal angles in two orthogonal planes)
    /// is the 4D oddity that can't happen in 3D. Verify vectors rotate
    /// consistently and length is preserved.
    #[test]
    fn bivector4_isoclinic_rotation_preserves_length() {
        let theta = 0.9;
        let r = Bivector4::new(theta, 0.0, 0.0, 0.0, 0.0, theta).exp();
        for v in [
            Vec4::X,
            Vec4::new(0.3, -0.4, 0.5, -0.6),
            Vec4::new(1.0, 2.0, 3.0, 4.0),
        ] {
            let rotated = r.apply(v);
            assert!(
                (rotated.length() - v.length()).abs() < 1e-4,
                "length drift: |v|={}, |Rv|={}",
                v.length(),
                rotated.length()
            );
        }
    }

    #[test]
    fn rotor4_full_turn_is_identity_on_vectors() {
        // A 2π rotation in one plane negates the rotor (scalar flips)
        // but leaves vectors fixed via the sandwich.
        let r = Bivector4::new(TAU, 0.0, 0.0, 0.0, 0.0, 0.0).exp();
        assert_vec4_close_tol(
            r.apply(Vec4::new(1.0, 2.0, 3.0, 4.0)),
            Vec4::new(1.0, 2.0, 3.0, 4.0),
            3e-4,
        );
    }

    #[test]
    fn rotor4_inverse_cancels() {
        let r = Bivector4::new(0.3, -0.2, 0.4, 0.1, -0.5, 0.25).exp();
        let id = r * r.inverse();
        assert_close(id.s, 1.0);
        assert_close(id.xy, 0.0);
        assert_close(id.xz, 0.0);
        assert_close(id.xw, 0.0);
        assert_close(id.yz, 0.0);
        assert_close(id.yw, 0.0);
        assert_close(id.zw, 0.0);
        assert_close(id.xyzw, 0.0);
    }

    #[test]
    fn rotor4_is_unit_norm() {
        for b in [
            Bivector4::new(0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
            Bivector4::new(0.5, 0.3, -0.4, 0.0, 0.0, 0.0),
            // Compound double-rotation.
            Bivector4::new(0.7, 0.0, 0.0, 0.0, 0.0, 0.5),
            // General bivector (both simple planes non-orthogonal in
            // the naive sense, requires the decomposition).
            Bivector4::new(0.3, -0.2, 0.4, 0.1, -0.5, 0.25),
        ] {
            let r = b.exp();
            let n = r.norm_squared();
            assert!(
                (n - 1.0).abs() < 1e-4,
                "rotor not unit-norm: |R|² = {n} for B = {b:?}"
            );
        }
    }

    #[test]
    fn rotor4_preserves_length() {
        let r = Bivector4::new(0.3, -0.2, 0.4, 0.1, -0.5, 0.25).exp();
        for v in [
            Vec4::X,
            Vec4::Y,
            Vec4::Z,
            Vec4::W,
            Vec4::new(1.0, 2.0, 3.0, 4.0),
            Vec4::new(-0.5, 0.5, -0.5, 0.5),
        ] {
            let rotated = r.apply(v);
            assert!(
                (rotated.length() - v.length()).abs() < 1e-3,
                "length drift: {v:?} -> {rotated:?}"
            );
        }
    }

    #[test]
    fn rotor4_composition_matches_sequential_apply() {
        // With the `v' = R̃·v·R` sandwich, `(ra·rb).apply(v)` equals `rb.apply(ra.apply(v))`,
        // ra applied first.
        let ra = Bivector4::new(0.4, 0.0, 0.0, 0.0, 0.0, 0.0).exp();
        let rb = Bivector4::new(0.0, 0.0, 0.0, 0.5, 0.0, 0.0).exp();
        let composed = ra * rb;
        let v = Vec4::new(0.7, -0.3, 1.1, 0.2);
        assert_vec4_close(composed.apply(v), rb.apply(ra.apply(v)));
    }

    #[test]
    fn rotor4_log_is_inverse_of_exp_simple() {
        for b in [
            Bivector4::ZERO,
            Bivector4::new(0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
            Bivector4::new(0.0, 0.6, 0.0, 0.0, 0.0, 0.0),
            Bivector4::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.2),
            Bivector4::new(0.4, 0.3, 0.0, 0.0, 0.0, 0.0),
        ] {
            let back = b.exp().log();
            let diff = (back + b * (-1.0)).magnitude();
            assert!(diff < 1e-4, "log∘exp mismatch: {b:?} -> {back:?}");
        }
    }

    /// `log ∘ exp` for a compound bivector recovers it up to the branch ambiguity inherent in
    /// the invariant decomposition. Verified here by applying both bivectors and checking the
    /// resulting rotations match on a set of probe vectors.
    #[test]
    fn rotor4_log_is_inverse_of_exp_compound() {
        let bv = Bivector4::new(0.5, 0.0, 0.0, 0.0, 0.0, 0.3);
        let back = bv.exp().log();
        let rotor_a = bv.exp();
        let rotor_b = back.exp();
        for v in [
            Vec4::X,
            Vec4::Y,
            Vec4::Z,
            Vec4::W,
            Vec4::new(1.0, -0.5, 0.3, 0.7),
        ] {
            assert_vec4_close_tol(rotor_a.apply(v), rotor_b.apply(v), 1e-3);
        }
    }

    /// Cross-check cardinal-plane rotations against `glam::Mat4` rotations in 3D subspaces. An
    /// xy-plane rotation in 4D restricted to the xy-subspace should agree with
    /// `Mat4::from_rotation_z`.
    #[test]
    fn rotor4_xy_matches_mat4_rotation_z_on_xy_subspace() {
        use glam::Mat4;
        let theta = 0.7;
        let rotor = Bivector4::new(theta, 0.0, 0.0, 0.0, 0.0, 0.0).exp();
        let mat = Mat4::from_rotation_z(theta);

        // Vectors lying in the xy plane (z = w = 0).
        for v in [
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.6, -0.8, 0.0, 0.0),
        ] {
            let via_rotor = rotor.apply(v);
            let via_mat = mat * v;
            assert_vec4_close_tol(via_rotor, via_mat, 1e-4);
        }
    }

    #[test]
    fn rotor4_normalize_produces_unit() {
        let r = Bivector4::new(0.3, 0.1, -0.2, 0.4, 0.1, 0.0).exp();
        // Perturb off the unit manifold.
        let perturbed = Rotor4 {
            s: r.s * 1.01,
            xy: r.xy * 1.01,
            xz: r.xz * 1.01,
            xw: r.xw * 1.01,
            yz: r.yz * 1.01,
            yw: r.yw * 1.01,
            zw: r.zw * 1.01,
            xyzw: r.xyzw * 1.01,
        };
        let back = perturbed.normalize();
        assert_close(back.norm_squared(), 1.0);
    }

    /// Small-angle path accuracy: a bivector with magnitude near f32 precision must produce a
    /// rotor that rotates vectors by ≈ that magnitude rather than collapsing to identity.
    #[test]
    fn rotor4_small_angle_path() {
        let eps = 1e-3_f32;
        let r = Bivector4::new(eps, 0.0, 0.0, 0.0, 0.0, 0.0).exp();
        let rotated = r.apply(Vec4::X);
        let expected = Vec4::new(eps.cos(), eps.sin(), 0.0, 0.0);
        assert_vec4_close_tol(rotated, expected, 1e-5);
    }

    /// Composition `(R_xy * R_xw).apply(v)` must equal `R_xw.apply(R_xy.apply(v))`. The earlier
    /// `rotor4_composition_matches_sequential_apply` test only covers two pure-3D planes (xy +
    /// yz); a sign error specific to the geometric product's w-mixing terms would survive that
    /// test and only show up here.
    #[test]
    fn rotor4_composition_xy_then_xw_matches_sequential_apply() {
        let r_xy = Bivector4::new(0.4, 0.0, 0.0, 0.0, 0.0, 0.0).exp();
        let r_xw = Bivector4::new(0.0, 0.0, 0.5, 0.0, 0.0, 0.0).exp();
        let composed = r_xy * r_xw;
        for v in [
            Vec4::X,
            Vec4::Y,
            Vec4::Z,
            Vec4::W,
            Vec4::new(0.7, -0.3, 1.1, 0.2),
        ] {
            assert_vec4_close_tol(composed.apply(v), r_xw.apply(r_xy.apply(v)), 1e-4);
        }
    }

    /// The polytope_playground demo's update loop:
    ///   per frame at 60 Hz: orientation = (omega * dt).exp() * orientation
    /// Over N steps with constant `omega`, the result must equal the closed-form
    /// `(omega * N * dt).exp()` within float tolerance. Catches both additive
    /// (sum-of-plane-bivectors) and multiplicative (rotor compose) drift specific to compound
    /// bivectors that include w-mixing planes.
    ///
    /// `omega = e_xy + e_xz + e_xw + e_yz` is the exact bivector the
    /// failing-screenshot polytope_playground run had active. Regression gate
    /// for the issue #37 fix in `Rotor4::mul` (the e14·e24 / e24·e14 pair's
    /// signs in the e12 output were swapped). Pre-fix drift was
    /// ~1.3%; post-fix it sits at f32 noise (~5e-7).
    #[test]
    fn rotor4_compound_xy_xz_xw_yz_integrated_matches_closed_form() {
        let omega = Bivector4::new(1.0, 1.0, 1.0, 1.0, 0.0, 0.0);
        let dt = 1.0 / 60.0;
        let n_steps = 60_u32; // 1 second of integration
        let total_angle = dt * n_steps as f32; // 1.0
        let delta = (omega * dt).exp();

        let mut integrated = Rotor4::IDENTITY;
        for _ in 0..n_steps {
            integrated = delta * integrated;
        }
        let closed_form = (omega * total_angle).exp();

        // Compare on a probe vector set: rotors that map the same
        // way on a basis are equal up to global sign (and the
        // sandwich kills the global sign anyway).
        for v in [
            Vec4::X,
            Vec4::Y,
            Vec4::Z,
            Vec4::W,
            Vec4::new(1.0, 2.0, 3.0, 4.0),
            Vec4::new(-0.5, 0.5, -0.5, 0.5),
        ] {
            let via_integrated = integrated.apply(v);
            let via_closed = closed_form.apply(v);
            let diff = (via_integrated - via_closed).length();
            // 1e-5 gives ~20x headroom over the post-fix f32 noise
            // floor (observed: ~5e-7). Tight enough to catch a
            // regression of the e12-component sign error; loose
            // enough to survive cross-machine f32 variation.
            assert!(
                diff < 1e-5,
                "compound xy+xz+xw+yz integration drift: v={v:?} \
                 integrated={via_integrated:?} closed={via_closed:?} \
                 diff={diff}",
            );
        }
    }

    /// 60 Hz × 15 s = 900 rotor multiplications, mirroring the pentatope_smoke screenshot's
    /// `t = 14.97 s` state. The rotor's norm-squared must stay near 1; visible drift here would
    /// mean vertices grow / shrink under sustained spin (suspect for the "polytopes appear to
    /// grow" symptom).
    #[test]
    fn rotor4_compound_integration_preserves_unit_norm_over_900_steps() {
        let omega = Bivector4::new(1.0, 1.0, 1.0, 1.0, 0.0, 0.0);
        let dt = 1.0 / 60.0;
        let delta = (omega * dt).exp();
        let mut r = Rotor4::IDENTITY;
        for _ in 0..900 {
            r = delta * r;
        }
        let n2 = r.norm_squared();
        // Post-fix observed: |R|² ≈ 1.0 + 2.3e-6 after 900 raw
        // multiplications (no normalize). Pre-fix the bug pushed it
        // to 25.4. 1e-5 is a comfortable post-fix gate.
        assert!(
            (n2 - 1.0).abs() < 1e-5,
            "rotor norm drifted after 900 compositions: |R|² = {n2}",
        );
    }

    /// Vertex-level invariant for the polytope-playground "are the shapes growing?" question: a
    /// unit-radius polytope vertex started at `(1, 0, 0, 0)` must stay at radius 1 after
    /// sustained integration. Length drift here would expand the rendered cross-section even for
    /// pure 3D rotations. Tests **raw** composition without intermediate `.normalize()` to catch
    /// algebraic drift in the geometric product itself.
    #[test]
    fn polytope_vertex_stays_on_unit_hypersphere_over_900_steps() {
        let omega = Bivector4::new(1.0, 1.0, 1.0, 1.0, 0.0, 0.0);
        let dt = 1.0 / 60.0;
        let delta = (omega * dt).exp();
        let mut r = Rotor4::IDENTITY;
        for _ in 0..900 {
            r = delta * r;
        }
        for v0 in [
            Vec4::X,
            Vec4::Y,
            Vec4::Z,
            Vec4::W,
            Vec4::new(0.5, 0.5, 0.5, 0.5), // tesseract vertex (radius 1)
        ] {
            let v_rotated = r.apply(v0);
            let l0 = v0.length();
            let l_rot = v_rotated.length();
            // Pre-fix: vertex grew to ~16.4. Post-fix: drift sits at ~2e-6 from f32
            // accumulation over 900 raw composes.
            assert!(
                (l_rot - l0).abs() < 1e-5,
                "vertex length drift over 900 steps: {v0:?} (|v|={l0}) -> {v_rotated:?} (|Rv|={l_rot})",
            );
        }
    }

    /// Same 900-step integration but **with `.normalize()` after each
    /// composition**, mirroring the actual integrator path in
    /// `rye_physics::euclidean_r4::integrate_orientation`. If this fails, the
    /// polytope-playground "shapes growing" symptom is the visible bug; if
    /// this passes,
    /// the integrator's existing normalize handles the algebraic drift the previous test catches
    /// and the visual issue is somewhere else (cross-section shape change, dispatcher inverse
    /// rotor, etc.).
    #[test]
    fn polytope_vertex_stays_on_unit_hypersphere_with_normalize_path() {
        let omega = Bivector4::new(1.0, 1.0, 1.0, 1.0, 0.0, 0.0);
        let dt = 1.0 / 60.0;
        let delta = (omega * dt).exp();
        let mut r = Rotor4::IDENTITY;
        for _ in 0..900 {
            r = (delta * r).normalize();
        }
        for v0 in [
            Vec4::X,
            Vec4::Y,
            Vec4::Z,
            Vec4::W,
            Vec4::new(0.5, 0.5, 0.5, 0.5),
        ] {
            let v_rotated = r.apply(v0);
            let l0 = v0.length();
            let l_rot = v_rotated.length();
            // Pre-fix (with normalize): vertex shrunk to ~0.65 because normalization to a unit
            // 8-sphere is the wrong manifold when the rotor itself was wrong. Post-fix the drift
            // is essentially zero (the normalize re-projects onto the correct manifold each step).
            assert!(
                (l_rot - l0).abs() < 1e-5,
                "normalized-path vertex length drift over 900 steps: \
                 {v0:?} (|v|={l0}) -> {v_rotated:?} (|Rv|={l_rot})",
            );
        }
    }

    /// `Bivector4::contract_vec` is the **Clifford left-contraction** `B ⌋ v` (grade-1 part of
    /// `B · v`), not the physics-side `ω × r`. This test pins down the sign convention so the
    /// future `Bivector5` / `Bivector6` impls inherit consistent semantics, and so the deviation
    /// a physics caller needs (a negation) stays explicit at the call site rather than baked into
    /// the bivector type itself.
    #[test]
    fn bivector4_contract_vec_is_clifford_left_contraction() {
        // e_xy ⌋ e_x = -e_y (because e_xy · e_x = e_x e_y e_x =
        // -e_x e_x e_y = -e_y).
        let b = Bivector4::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_vec4_close_tol(b.contract_vec(Vec4::X), -Vec4::Y, 1e-6);
        // e_xy ⌋ e_y = +e_x (e_x e_y e_y = e_x).
        assert_vec4_close_tol(b.contract_vec(Vec4::Y), Vec4::X, 1e-6);
        // e_zw ⌋ e_z = -e_w; e_zw ⌋ e_w = +e_z.
        let b = Bivector4::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        assert_vec4_close_tol(b.contract_vec(Vec4::Z), -Vec4::W, 1e-6);
        assert_vec4_close_tol(b.contract_vec(Vec4::W), Vec4::Z, 1e-6);
        // Antisymmetry: B ⌋ v vanishes when v lies outside B's plane.
        let b = Bivector4::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0); // e_xy
        assert_vec4_close_tol(b.contract_vec(Vec4::Z), Vec4::ZERO, 1e-6);
        assert_vec4_close_tol(b.contract_vec(Vec4::W), Vec4::ZERO, 1e-6);
    }

    /// `Plane4::ALL[i].unit_bivector()` must equal `Bivector4::basis(i)`, and both must agree
    /// with `Bivector4`'s field ordering; index `i` sets exactly the i-th field to 1.
    #[test]
    fn plane4_unit_bivector_matches_bivector4_field_order() {
        // Order: 0=xy, 1=xz, 2=xw, 3=yz, 4=yw, 5=zw.
        let expected = [
            Bivector4::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Bivector4::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            Bivector4::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            Bivector4::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            Bivector4::new(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            Bivector4::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ];
        for (i, plane) in Plane4::ALL.iter().enumerate() {
            assert_eq!(
                plane.unit_bivector(),
                expected[i],
                "Plane4::{plane:?} did not produce the expected basis bivector"
            );
            assert_eq!(
                Bivector4::basis(i),
                expected[i],
                "Bivector4::basis({i}) drifted from Plane4 ordering"
            );
        }
    }

    #[test]
    fn plane4_label_matches_field_name() {
        assert_eq!(Plane4::Xy.label(), "xy");
        assert_eq!(Plane4::Xz.label(), "xz");
        assert_eq!(Plane4::Xw.label(), "xw");
        assert_eq!(Plane4::Yz.label(), "yz");
        assert_eq!(Plane4::Yw.label(), "yw");
        assert_eq!(Plane4::Zw.label(), "zw");
    }

    /// `From<Rotor4> for [f32; 8]` packs in the order `[s, xy, xz, xw, yz, yw, zw, xyzw]`. This
    /// pinned ordering is what GPU uniform buffers (e.g. `BodyUniform.rotor`) consume; a future
    /// refactor that swaps fields must also update this conversion in lockstep, and this test
    /// catches the mismatch.
    #[test]
    fn rotor4_to_slot_packs_in_canonical_order() {
        let r = Rotor4 {
            s: 1.0,
            xy: 2.0,
            xz: 3.0,
            xw: 4.0,
            yz: 5.0,
            yw: 6.0,
            zw: 7.0,
            xyzw: 8.0,
        };
        let slot: [f32; 8] = r.into();
        assert_eq!(slot, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn rotor4_identity_slot_matches_into_identity() {
        let from_into: [f32; 8] = Rotor4::IDENTITY.into();
        assert_eq!(Rotor4::IDENTITY_SLOT, from_into);
    }
}
