//! Bivectors and rotors, the N-dim rotation primitive.
//!
//! A bivector is an oriented plane of rotation times a magnitude; its
//! exponential is a rotor, which acts via the sandwich product and composes
//! via Clifford multiplication.
//!
//! Two traits ([`Bivector`], [`Rotor`]) let consumers be generic over
//! dimension. Concrete impls ship per-N: [`Bivector2`] / [`Rotor2`],
//! [`Bivector3`] / [`Rotor3`], [`Bivector4`] / [`Rotor4`].
//!
//! Per-N hand-rolled (not `generic_const_exprs`): the Clifford tables are
//! dimension-specific and the exp/log derivations differ qualitatively (3D
//! is always single-plane; 4D needs invariant decomposition).
//!
//! ## Convention
//!
//! `R = exp(B/2)`, so a bivector of magnitude θ in plane `e_i ∧ e_j` rotates
//! by θ from `e_i` toward `e_j`. Rotor application is the sandwich
//! `v' = R̃ · v · R`.

use std::ops::{Add, Mul};

use glam::{Vec2, Vec3, Vec4};

/// A bivector in G(N, 0): an oriented plane of rotation times magnitude;
/// exponentiates to a rotor.
pub trait Bivector: Copy + Add<Output = Self> + Mul<f32, Output = Self> {
    type Rotor: Rotor<Bivector = Self>;

    /// Zero bivector; `zero().exp()` is the identity rotor.
    fn zero() -> Self;

    /// Exponential map, closed-form per dimension.
    fn exp(self) -> Self::Rotor;
}

/// A rotor in G(N, 0): an element of Spin(N), unit-norm by construction.
pub trait Rotor: Copy + Mul<Output = Self> {
    type Bivector: Bivector<Rotor = Self>;
    type Vector: Copy;

    /// Identity rotor.
    fn identity() -> Self;

    /// Reverse / conjugate rotor; the inverse for unit rotors.
    fn inverse(self) -> Self;

    /// Rotate a vector via the sandwich product.
    fn apply(&self, v: Self::Vector) -> Self::Vector;

    /// Inverse of [`Bivector::exp`]; principal branch, angle in `[−π, π]`.
    fn log(self) -> Self::Bivector;
}

// 2D: Bivector2 is the e1∧e2 coefficient; Rotor2 is a unit complex number.

/// 2D bivector: scalar coefficient on `e1∧e2`, a rotation angle in radians
/// from `x` toward `y`.
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

/// 2D rotor: unit complex number `a + b·e1e2`, `a = cos(θ/2)`,
/// `b = sin(θ/2)`.
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
        Self {
            a: self.a,
            b: -self.b,
        }
    }

    fn apply(&self, v: Vec2) -> Vec2 {
        // cos(θ) = a² − b², sin(θ) = 2ab.
        let c = self.a * self.a - self.b * self.b;
        let s = 2.0 * self.a * self.b;
        Vec2::new(c * v.x - s * v.y, s * v.x + c * v.y)
    }

    fn log(self) -> Bivector2 {
        Bivector2(2.0 * self.b.atan2(self.a))
    }
}

// 3D: Rotor3 is the even subalgebra element `s + xy·e12 + yz·e23 + zx·e31`,
// isomorphic to a quaternion but with the sign convention of the sandwich
// `v' = R̃·v·R` rather than the quaternion `q·v·q*`.

/// 3D bivector with coefficients on `e1∧e2`, `e2∧e3`, `e3∧e1`. Magnitude
/// encodes rotation angle.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Bivector3 {
    /// Coefficient on `e1∧e2` (xy plane).
    pub xy: f32,
    /// Coefficient on `e2∧e3` (yz plane).
    pub yz: f32,
    /// Coefficient on `e3∧e1` (zx plane).
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

    /// Magnitude; the rotation angle when used as a rotor generator.
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

    /// Exponential map. Every 3D bivector is simple (single plane), so
    /// `exp(B/2) = cos(θ/2) + sin(θ/2)·B̂` directly, no decomposition.
    fn exp(self) -> Rotor3 {
        let mag_sq = self.xy * self.xy + self.yz * self.yz + self.zx * self.zx;
        // 1e-16: below this squared-angle the f32 sin/cos round-trip is
        // noise; the linear `sin(θ/2)/θ ≈ 1/2` limit is exact in f32.
        if mag_sq < 1e-16 {
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

/// 3D rotor: scalar plus bivector part, unit-norm by construction
/// (`s² + xy² + yz² + zx² = 1`).
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
    /// Geometric product of two rotors in G(3,0).
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
        // Reverse: flip grade-2 sign.
        Self {
            s: self.s,
            xy: -self.xy,
            yz: -self.yz,
            zx: -self.zx,
        }
    }

    /// Apply via the sandwich `R̃ · v · R`. Two stages: `R̃ · v`
    /// (vector + trivector), then `· R` (trivector cancels for unit R).
    fn apply(&self, v: Vec3) -> Vec3 {
        let (s, a, b, c) = (self.s, self.xy, self.yz, self.zx);
        let (vx, vy, vz) = (v.x, v.y, v.z);

        // R̃·v: vector part (p1, p2, p3) and trivector coefficient pt.
        let p1 = s * vx - a * vy + c * vz;
        let p2 = s * vy + a * vx - b * vz;
        let p3 = s * vz + b * vy - c * vx;
        let pt = -(a * vz + b * vx + c * vy);

        Vec3::new(
            p1 * s - p2 * a + p3 * c - pt * b,
            p2 * s + p1 * a - p3 * b - pt * c,
            p3 * s + p2 * b - p1 * c - pt * a,
        )
    }

    fn log(self) -> Bivector3 {
        // Inverse of exp: θ = 2·atan2(|bivector part|, scalar).
        let mag_sq = self.xy * self.xy + self.yz * self.yz + self.zx * self.zx;
        if mag_sq < 1e-16 {
            // Near-identity: log ≈ 2·bivector_part.
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

// 4D: Rotor4 is the even-graded element of G(4,0), eight components (scalar +
// six bivectors + pseudoscalar). Unlike 3D, a generic 4D bivector is not
// simple: it splits uniquely into two orthogonal simple parts B = B_a + B_b
// in complementary 2-planes, and the rotor is the product of their
// exponentials. `B ∧ B = 0` is the simple case.

/// 4D bivector, six components on the basis planes `e_i ∧ e_j` (`i < j`).
///
/// Unlike [`Bivector3`], a 4D bivector can describe a double rotation (two
/// independent planes); the exponential handles this via the invariant
/// decomposition.
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

    /// Unit basis bivector at index `i` (`0=xy, 1=xz, 2=xw, 3=yz, 4=yw,
    /// 5=zw`). Panics if `i >= 6`; see [`Plane4::unit_bivector`] for the
    /// type-safe form.
    pub fn basis(i: usize) -> Self {
        let mut c = [0.0_f32; 6];
        c[i] = 1.0;
        Self::new(c[0], c[1], c[2], c[3], c[4], c[5])
    }

    /// `|B|² = Σ α²_ij`, equal to `θ₁² + θ₂²` over the two simple parts.
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

    /// Read the coefficient of a basis plane.
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

    /// Write the coefficient of a basis plane; mirror of [`Self::component`].
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

    /// Euclidean inner product on the 6 coefficients (positive definite),
    /// not the Clifford scalar part (which carries the `e_ij·e_ij = -1`
    /// sign). Projects one bivector onto another's direction.
    pub fn dot(self, other: Self) -> f32 {
        self.xy * other.xy
            + self.xz * other.xz
            + self.xw * other.xw
            + self.yz * other.yz
            + self.yw * other.yw
            + self.zw * other.zw
    }

    /// Pseudoscalar coefficient of `B ∧ B`. Zero iff `B` is simple; nonzero
    /// means the exponential needs the invariant decomposition.
    pub fn wedge_self_coeff(self) -> f32 {
        2.0 * (self.xy * self.zw - self.xz * self.yw + self.xw * self.yz)
    }

    /// Wedge product `u ∧ v` as a bivector. Physics builds the torque
    /// bivector `r ∧ f` with this.
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

    /// Clifford left-contraction `B ⌋ v` (grade-1 part of `B · v`), standard
    /// math sign: `e_xy ⌋ e_x = −e_y`.
    ///
    /// Physics wants the opposite sign (`ω × r`); use
    /// `loam_physics::euclidean_r4::omega_cross_r` or negate. Kept
    /// Clifford-pure so future `Bivector5`/`Bivector6` stay consistent.
    pub fn contract_vec(self, v: Vec4) -> Vec4 {
        Vec4::new(
            self.xy * v.y + self.xz * v.z + self.xw * v.w,
            -self.xy * v.x + self.yz * v.z + self.yw * v.w,
            -self.xz * v.x - self.yz * v.y + self.zw * v.w,
            -self.xw * v.x - self.yw * v.y - self.zw * v.z,
        )
    }

    /// Hodge dual `B* = B · I`, swapping each plane with its orthogonal
    /// complement. Used inside the invariant decomposition.
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

/// One of the six elementary 4D rotation planes; a basis bivector of
/// [`Bivector4`]. Index and label match `Bivector4`'s field order
/// (`0=xy, 1=xz, 2=xw, 3=yz, 4=yw, 5=zw`).
///
/// The w-planes (`xw`, `yw`, `zw`) pull visible axes into the 4th dimension;
/// the pure-3D planes act on the 3D cross-section. Sum-of-bivectors
/// composition is commutative, so `omega.exp()` depends only on the set of
/// active planes, not their order.
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

    /// Lowercase two-letter label, e.g. `"xy"`. Stable serialization key.
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

    /// Exponential map via the invariant decomposition. The angles
    /// `θ₁ ≥ |θ₂| ≥ 0` of the two orthogonal simple parts solve
    ///
    /// ```text
    ///   θ₁² + θ₂² = |B|²
    ///   2·θ₁·θ₂   = wedge_self_coeff(B)
    /// ```
    ///
    /// Four f32-stable branches: zero, simple (`δ ≈ 0`), isoclinic
    /// (`disc ≈ 0`, equal angles; avoids the `1/(θ₁²−θ₂²)` singularity),
    /// and the general expansion of `exp(B_a/2)·exp(B_b/2)`.
    fn exp(self) -> Rotor4 {
        let s = self.magnitude_squared();
        if s < 1e-16 {
            return Rotor4::IDENTITY;
        }
        let delta = self.wedge_self_coeff();

        // Simple case (δ ≈ 0): single plane, 3D-style rotor.
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

        // Isoclinic case (t₁ ≈ t₂, equal angles): avoids 0/0 in the general
        // path. `exp(B/2) = cos²(θ/2) + (sin(θ)/(2θ))·B + sin²(θ/2)·sign(δ)·I`.
        if disc < 1e-6 * s.max(1.0) {
            let theta_sq = s * 0.5;
            let theta = theta_sq.sqrt();
            let half = theta * 0.5;
            let ch = half.cos();
            let sh = half.sin();
            let sign_i = delta.signum();
            // 1e-8: below this θ, sin(θ)/(2θ) loses its leading term in
            // f32; use the limit 1/2.
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

        // General compound case: θ₁, θ₂ nonzero and distinct. Positive root
        // for θ₁, δ-signed root for θ₂.
        let t1 = ((s + disc) * 0.5).max(0.0).sqrt();
        let t2 = ((s - disc) * 0.5).max(0.0).sqrt() * delta.signum();

        let half1 = t1 * 0.5;
        let half2 = t2 * 0.5;
        let c1 = half1.cos();
        let s1 = half1.sin();
        let c2 = half2.cos();
        let s2 = half2.sin();

        // t1 > 0 and t2 ≠ 0 here, so both divisions are safe.
        let s1_t1 = s1 / t1;
        let s2_t2 = s2 / t2;

        // Coefficients from expanding `exp(B_a/2)·exp(B_b/2)` in `{1,B,B*,I}`,
        // with B_a,b = ((s ± disc)·B ± δ·B*) / (2·disc).
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

/// 4D rotor: even-graded element of G(4,0), eight components (scalar, six
/// bivectors, pseudoscalar). Unit-norm by construction.
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

    /// Identity rotor in the `[s, xy, xz, xw, yz, yw, zw, xyzw]` slot order of
    /// `From<Rotor4> for [f32; 8]`, for GPU uniform initialization.
    pub const IDENTITY_SLOT: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    /// Squared norm `<R̃·R>_0`; `1` for a proper rotor.
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

    /// Renormalize onto the Spin(4) unit manifold; apply to counter f32
    /// drift after long integrator runs.
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

/// Pack into the `[s, xy, xz, xw, yz, yw, zw, xyzw]` slot order GPU uniform
/// buffers consume (e.g. `loam-render`'s `BodyUniform`).
impl From<Rotor4> for [f32; 8] {
    fn from(r: Rotor4) -> Self {
        [r.s, r.xy, r.xz, r.xw, r.yz, r.yw, r.zw, r.xyzw]
    }
}

impl Rotor4 {
    /// Rotor as a 4×4 rotation matrix, columns being the rotor applied to the
    /// basis vectors. Column-major `[col0..col3]`, matching glam's `Mat4` and
    /// WGSL's `mat4x4<f32>` so it casts into a GPU slot without transposition.
    ///
    /// Four [`Rotor::apply`] calls; lets the shader do a plain
    /// `mat4x4 * vec4` instead of the full Clifford reduction.
    ///
    /// Reference: the "matrix from rotation operator" construction
    /// (Hestenes, *New Foundations for Classical Mechanics*, 2nd ed., §2.5).
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
    /// Geometric product of two 4D rotors in the even-graded basis
    /// `{1, e12, e13, e14, e23, e24, e34, I}`, by direct Clifford reduction.
    fn mul(self, rhs: Self) -> Self {
        let (a0, a12, a13, a14, a23, a24, a34, a_i) = (
            self.s, self.xy, self.xz, self.xw, self.yz, self.yw, self.zw, self.xyzw,
        );
        let (b0, b12, b13, b14, b23, b24, b34, b_i) = (
            rhs.s, rhs.xy, rhs.xz, rhs.xw, rhs.yz, rhs.yw, rhs.zw, rhs.xyzw,
        );

        // Scalar: `1·1`, `e_ij·e_ij = −1` (six), `I·I = +1`.
        let s = a0 * b0 - a12 * b12 - a13 * b13 - a14 * b14 - a23 * b23 - a24 * b24 - a34 * b34
            + a_i * b_i;

        // Bivector components. Signs from the duality table
        // (e12·I = −e34, e13·I = +e24, e14·I = −e23, e23·I = −e14,
        // e24·I = +e13, e34·I = −e12) and the shared-index products
        // (for e12: e13·e23 = −e12, e23·e13 = +e12, e14·e24 = −e12,
        // e24·e14 = +e12; the last pair is sign-sensitive and drove
        // w-plane drift when swapped).
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

        // Pseudoscalar: disjoint-plane wedges (e12·e34 = I, e13·e24 = −I,
        // e14·e23 = I, plus reverses) and `a0·b_i + a_i·b0`.
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

    /// Reverse `R̃`: flips grades with `k(k−1)/2` odd, i.e. grade 2 only.
    /// Scalar and pseudoscalar keep their sign.
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

    /// Apply via the sandwich `R̃ · v · R`. Stage 1 is `R̃ · v` (1-vector +
    /// 3-vector); stage 2 multiplies by `R` and extracts the 1-vector part.
    /// All basis-element products are direct Clifford reductions in G(4,0).
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

        // Stage 1: R̃·v, with R̃'s grade-2 sign flip folded into the formulas.
        let p1 = rs * vx - rxy * vy - rxz * vz - rxw * vw;
        let p2 = rs * vy + rxy * vx - ryz * vz - ryw * vw;
        let p3 = rs * vz + rxz * vx + ryz * vy - rzw * vw;
        let p4 = rs * vw + rxw * vx + ryw * vy + rzw * vz;

        // 3-vector part of R̃·v in basis (e123, e124, e134, e234).
        let t123 = -rxy * vz + rxz * vy - ryz * vx + r_i * vw;
        let t124 = -rxy * vw + rxw * vy - ryw * vx - r_i * vz;
        let t134 = -rxz * vw + rxw * vz - rzw * vx + r_i * vy;
        let t234 = -ryz * vw + ryw * vz - rzw * vy - r_i * vx;

        // Stage 2: (1-vec + 3-vec) · R, extract the 1-vec output.
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

    /// Inverse of [`Bivector4::exp`]. Both maps are diagonal in the self-dual
    /// split of Λ²R⁴, which is what collapses the invariant decomposition to
    /// a single branch-free formula.
    ///
    /// Recovers the principal branch: `(θ₁ ± θ₂)/2 ∈ [0, π]`.
    fn log(self) -> Bivector4 {
        // Write R₂ for the bivector part and R₂* for its Hodge dual. The
        // combination R₂ ∓ R₂* carries the half-angle h± = (θ₁ ± θ₂)/2 alone:
        // over the same fixed plane pair U± (an eigenvector of the Hodge
        // dual, pinned by the rotor's invariant frame),
        //
        //     R₂ ∓ R₂* = sin(h±)·U±       B ∓ B* = 2·h±·U±
        //
        // so B = (h₊/sin h₊)·(R₂ − R₂*) + (h₋/sin h₋)·(R₂ + R₂*). The dual
        // pairs the components (xy,zw), (xz,yw), (xw,yz), so each eigenpart
        // has three independent coefficients, and the vector of those three
        // has length |sin h±|.
        //
        // The cosines come from the scalar and pseudoscalar: with
        // c = cos(θ₁/2)·cos(θ₂/2) and p = sin(θ₁/2)·sin(θ₂/2), product-to-sum
        // gives c ∓ p = cos(h±). Taking h± by atan2 against the sine holds
        // full relative precision where acos(c ∓ p) alone loses half the
        // mantissa, its argument sitting within an ulp of ±1 near identity
        // and near the isoclinic locus (Kahan 2006, "How Futile are Mindless
        // Assessments of Roundoff in Floating-Point Computation?", Mangled
        // Angles).
        let c = self.s;
        let p = self.xyzw;
        let sum_part = Vec3::new(self.xy + self.zw, self.xz - self.yw, self.xw + self.yz);
        let diff_part = Vec3::new(self.xy - self.zw, self.xz + self.yw, self.xw - self.yz);
        let sin_sum = sum_part.length();
        let sin_diff = diff_part.length();

        // h/sin(h) -> 1 as h -> 0, so the guard covers only the exact 0/0.
        // The
        // other zero of the sine, h = π, is the branch cut: the eigenpart's
        // plane is then unrecoverable from the rotor, and scaling the zero
        // vector by 1 drops that half-turn rather than returning infinities.
        let k_sum = if sin_sum > 0.0 {
            sin_sum.atan2(c - p) / sin_sum
        } else {
            1.0
        };
        let k_diff = if sin_diff > 0.0 {
            sin_diff.atan2(c + p) / sin_diff
        } else {
            1.0
        };

        Bivector4 {
            xy: k_sum * sum_part.x + k_diff * diff_part.x,
            xz: k_sum * sum_part.y + k_diff * diff_part.y,
            xw: k_sum * sum_part.z + k_diff * diff_part.z,
            yz: k_sum * sum_part.z - k_diff * diff_part.z,
            yw: k_diff * diff_part.y - k_sum * sum_part.y,
            zw: k_sum * sum_part.x - k_diff * diff_part.x,
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
        // exp gives the negative identity rotor; the sandwich squares the
        // sign out, so vectors are unchanged.
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
        // Under `v' = R̃·v·R`, `(ra·rb).apply(v) = rb.apply(ra.apply(v))`:
        // ra applies first.
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
        // Cross-check the sign convention against glam's Quat.
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

    /// `component` / `set_component` round-trip is the identity.
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

    /// `dot` invariants: orthogonality, self-dot 1, bilinearity.
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

    /// Single-plane rotations are planar: two coordinates swap under a
    /// quarter turn, the other two stay fixed.
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

    /// A simple bivector has `B ∧ B = 0`.
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

    /// Double rotation in xy and zw simultaneously; nonzero pseudoscalar is
    /// the fingerprint of the compound decomposition.
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

    /// Isoclinic rotation (equal angles in two orthogonal planes) preserves
    /// length.
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
            // General bivector, requires the decomposition.
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
        // Under the sandwich, `(ra·rb).apply(v) = rb.apply(ra.apply(v))`.
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

    /// `log ∘ exp` for a compound bivector, recovered up to the
    /// decomposition's branch ambiguity; checked via probe-vector rotations.
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

    /// Two orthogonal simple planes spanning R⁴, oriented so that
    /// `(p1·t₁ + p2·t₂).wedge_self_coeff() = 2·t₁·t₂`, which is the sign
    /// convention the invariant decomposition assigns to `(θ₁, θ₂)`. The
    /// tilted pair spreads each simple part over four components, so a
    /// swapped plane cannot hide in a single coefficient.
    ///
    /// Spreading over components is not the same as spreading over the
    /// self-dual split: both of those pairs still put one eigenpart on a
    /// single coordinate of that split, which is what
    /// [`nondegenerate_plane_pairs`] fixes.
    fn invariant_plane_pairs() -> [(Bivector4, Bivector4); 4] {
        let root_half = 0.5_f32.sqrt();
        let u1 = Vec4::new(root_half, root_half, 0.0, 0.0);
        let v1 = Vec4::new(0.0, 0.0, root_half, root_half);
        let u2 = Vec4::new(root_half, -root_half, 0.0, 0.0);
        let v2 = Vec4::new(0.0, 0.0, root_half, -root_half);
        let [oblique, orthogonal] = nondegenerate_plane_pairs();
        [
            (Bivector4::basis(0), Bivector4::basis(5)),
            (Bivector4::wedge(u1, v1), Bivector4::wedge(v2, u2)),
            oblique,
            orthogonal,
        ]
    }

    /// The complementary plane pair whose self-dual split, in the
    /// coordinates `Rotor4::log` reads it in, is `(sum, +diff)` for the
    /// first plane and `(sum, −diff)` for the second. Unit `sum` and `diff`
    /// make both planes simple and unit: `|p|² = (|sum|² + |diff|²)/2 = 1`
    /// and `wedge_self_coeff(p) = (|sum|² − |diff|²)/2 = 0`. The eigenparts
    /// of `p1·t₁ + p2·t₂` are then `(t₁+t₂)·sum` and `(t₁−t₂)·diff`, so its
    /// invariant angles are `t₁` and `t₂` and its `wedge_self_coeff` is
    /// `2·t₁·t₂`, the orientation the caller's convention requires.
    fn plane_pair_from_eigenparts(sum: Vec3, diff: Vec3) -> (Bivector4, Bivector4) {
        let plane = |d: Vec3| Bivector4 {
            xy: 0.5 * (sum.x + d.x),
            xz: 0.5 * (sum.y + d.y),
            xw: 0.5 * (sum.z + d.z),
            yz: 0.5 * (sum.z - d.z),
            yw: 0.5 * (d.y - sum.y),
            zw: 0.5 * (sum.x - d.x),
        };
        (plane(diff), plane(-diff))
    }

    /// Plane pairs that are non-degenerate in the self-dual 3-space: every
    /// coordinate of both the sum part and the difference part is nonzero,
    /// and the two directions are not parallel. `log` emits each output
    /// coefficient as one sum-part coordinate plus or minus one
    /// difference-part coordinate, so a fixture that zeroes either
    /// coordinate of a pair cannot see a sign error on that term.
    ///
    /// Both directions are unit. The first `diff` is oblique to `sum` and
    /// the second orthogonal to it, so the two pairs do not share one
    /// mutual angle between the eigenparts.
    fn nondegenerate_plane_pairs() -> [(Bivector4, Bivector4); 2] {
        let sum = Vec3::new(2.0, 3.0, 6.0) / 7.0;
        let oblique_diff = Vec3::new(9.0, 2.0, 6.0) / 11.0;
        let orthogonal_diff = Vec3::new(3.0, -6.0, 2.0) / 7.0;
        [
            plane_pair_from_eigenparts(sum, oblique_diff),
            plane_pair_from_eigenparts(sum, orthogonal_diff),
        ]
    }

    /// Angle pairs for the half-angle recovery. The small and the near-equal
    /// pairs put `cos((θ₁ ± θ₂)/2)` within an ulp of 1, which is where
    /// recovering that half-angle from its cosine alone loses half the
    /// mantissa; the last two are well-separated controls.
    const HALF_ANGLE_STRESS_PAIRS: [(f32, f32); 10] = [
        (1.0e-3, 1.0e-3),
        (1.0e-3, 5.0e-4),
        (1.0e-3, -1.0e-3),
        (1.0e-3, 0.0),
        (0.7, 0.7),
        (0.7, 0.6999),
        (1.0, 0.999),
        (2.0, 1.999),
        (1.2, 0.3),
        (0.5, -0.4),
    ];

    /// Both invariant angles survive `log`, not just the rotor's action on a
    /// probe vector. The reference rotor is the product of two simple
    /// rotors, so it is built entirely from `exp`'s well-conditioned simple
    /// branch and the residual here is `log`'s alone.
    #[test]
    fn rotor4_log_recovers_both_invariant_angles() {
        for (p1, p2) in invariant_plane_pairs() {
            for (t1, t2) in HALF_ANGLE_STRESS_PAIRS {
                let b = p1 * t1 + p2 * t2;
                let back = ((p1 * t1).exp() * (p2 * t2).exp()).log();
                let diff = (back + b * (-1.0)).magnitude();
                assert!(
                    diff <= 1e-6,
                    "log mismatch at ({t1}, {t2}): {b:?} -> {back:?}"
                );
            }
        }
    }

    /// `log ∘ exp` on the same stress table. The tolerance is looser than
    /// the test above because `exp`'s compound branch carries the
    /// discriminant `√(|B|⁴ − (B∧B)²)`, which cancels to noise as the two
    /// angles converge; that floor is `exp`'s, not the half-angle
    /// recovery's, and it scales with `|B|` rather than sitting at a fixed
    /// absolute size. The worst case in this table, `(2, 1.999)` on the
    /// eigenpart-orthogonal pair, measures `1.3e-5·|B|`.
    #[test]
    fn rotor4_log_of_exp_round_trips_at_small_and_near_equal_angles() {
        for (p1, p2) in invariant_plane_pairs() {
            for (t1, t2) in HALF_ANGLE_STRESS_PAIRS {
                let b = p1 * t1 + p2 * t2;
                let back = b.exp().log();
                let diff = (back + b * (-1.0)).magnitude();
                assert!(
                    diff <= 2e-5 * b.magnitude(),
                    "log∘exp mismatch at ({t1}, {t2}): {b:?} -> {back:?}"
                );
            }
        }
    }

    /// The fixtures really are invariant plane pairs: each plane simple and
    /// unit, the two orthogonal to each other, and the pair carrying the
    /// orientation the angle convention assumes. A fixture that failed this
    /// would make the round-trip tests above assert against a bivector whose
    /// invariant angles are not the `(t₁, t₂)` they were built from.
    #[test]
    fn invariant_plane_pairs_are_orthogonal_simple_unit_planes() {
        let (t1, t2) = (1.2_f32, -0.3_f32);
        for (p1, p2) in invariant_plane_pairs() {
            for p in [p1, p2] {
                assert_close(p.magnitude_squared(), 1.0);
                assert_close(p.wedge_self_coeff(), 0.0);
            }
            assert_close(p1.dot(p2), 0.0);
            assert_close((p1 * t1 + p2 * t2).wedge_self_coeff(), 2.0 * t1 * t2);
        }
    }

    /// What makes the last two fixtures discriminating: `log` writes each
    /// output coefficient as one sum-part coordinate plus or minus one
    /// difference-part coordinate, so a coordinate sitting at zero absorbs a
    /// sign error on the term it multiplies.
    #[test]
    fn nondegenerate_pairs_populate_every_self_dual_coordinate() {
        // The angle table scales these two directions by (t₁ + t₂) and
        // (t₁ − t₂), so a coordinate near zero here is near zero at every
        // angle pair at once. The bound sits below the smallest coordinate
        // the fixtures carry (2/11) and far above any f32 residue.
        const MIN_COORDINATE: f32 = 0.1;
        for (p1, _) in nondegenerate_plane_pairs() {
            let sum = Vec3::new(p1.xy + p1.zw, p1.xz - p1.yw, p1.xw + p1.yz);
            let diff = Vec3::new(p1.xy - p1.zw, p1.xz + p1.yw, p1.xw - p1.yz);
            for (s, d) in sum.to_array().into_iter().zip(diff.to_array()) {
                assert!(
                    s.abs() > MIN_COORDINATE && d.abs() > MIN_COORDINATE,
                    "eigenpart coordinate too small to pin a sign: {sum:?}, {diff:?}"
                );
            }
            assert!(
                sum.cross(diff).length() > MIN_COORDINATE,
                "eigenparts are parallel, so a swap between them stays hidden: \
                 {sum:?}, {diff:?}"
            );
        }
    }

    /// Branch cut: the isoclinic half-turn has zero bivector part, so its
    /// plane pair is unrecoverable. The recovered log must still be finite
    /// and must still exponentiate to that rotation (which is −1 on every
    /// vector, whatever plane pair it is read on).
    #[test]
    fn rotor4_log_stays_finite_at_the_isoclinic_branch_cut() {
        let rotor = Bivector4::new(PI, 0.0, 0.0, 0.0, 0.0, PI).exp();
        let back = rotor.log();
        assert!(
            back.magnitude_squared().is_finite(),
            "non-finite log at the branch cut: {back:?}"
        );
        for v in [Vec4::X, Vec4::W, Vec4::new(1.0, -0.5, 0.3, 0.7)] {
            assert_vec4_close_tol(back.exp().apply(v), v * -1.0, 1e-5);
        }
    }

    /// An xy-plane 4D rotation restricted to the xy-subspace agrees with
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

    /// Small-angle path: a near-precision magnitude must still rotate by that
    /// magnitude, not collapse to identity.
    #[test]
    fn rotor4_small_angle_path() {
        let eps = 1e-3_f32;
        let r = Bivector4::new(eps, 0.0, 0.0, 0.0, 0.0, 0.0).exp();
        let rotated = r.apply(Vec4::X);
        let expected = Vec4::new(eps.cos(), eps.sin(), 0.0, 0.0);
        assert_vec4_close_tol(rotated, expected, 1e-5);
    }

    /// Composition with a w-mixing plane (xy then xw). The pure-3D
    /// composition test would miss a sign error in the w-mixing terms.
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

    /// Integrating `orientation = (omega·dt).exp() · orientation` over N steps
    /// must equal the closed-form `(omega·N·dt).exp()`. Regression gate for
    /// the `Rotor4::mul` e14·e24 / e24·e14 sign fix in the e12 output: pre-fix
    /// drift was ~1.3%, post-fix it sits at f32 noise (~5e-7).
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

        // Probe vectors: the sandwich kills the rotors' global-sign ambiguity.
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
            // 1e-5: ~20x over the post-fix noise floor (~5e-7), tight enough
            // to catch the e12 sign regression, loose enough for cross-machine
            // f32 variation.
            assert!(
                diff < 1e-5,
                "compound xy+xz+xw+yz integration drift: v={v:?} \
                 integrated={via_integrated:?} closed={via_closed:?} \
                 diff={diff}",
            );
        }
    }

    /// 900 rotor multiplications (60 Hz × 15 s): norm-squared must stay near
    /// 1, else vertices grow/shrink under sustained spin.
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
        // Post-fix |R|² ≈ 1 + 2.3e-6 over 900 raw multiplies; pre-fix the
        // bug pushed it to 25.4. 1e-5 is a comfortable gate.
        assert!(
            (n2 - 1.0).abs() < 1e-5,
            "rotor norm drifted after 900 compositions: |R|² = {n2}",
        );
    }

    /// A unit-radius vertex must stay at radius 1 after sustained integration.
    /// Raw composition (no `.normalize()`) to catch algebraic drift in the
    /// geometric product itself.
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
            // Pre-fix the vertex grew to ~16.4; post-fix drift is ~2e-6.
            assert!(
                (l_rot - l0).abs() < 1e-5,
                "vertex length drift over 900 steps: {v0:?} (|v|={l0}) -> {v_rotated:?} (|Rv|={l_rot})",
            );
        }
    }

    /// Same 900-step integration but with `.normalize()` each step, mirroring
    /// `loam_physics::euclidean_r4::integrate_orientation`.
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
            // Pre-fix (with normalize) the vertex shrank to ~0.65: normalizing
            // a wrong rotor projects onto the wrong manifold. Post-fix ~0.
            assert!(
                (l_rot - l0).abs() < 1e-5,
                "normalized-path vertex length drift over 900 steps: \
                 {v0:?} (|v|={l0}) -> {v_rotated:?} (|Rv|={l_rot})",
            );
        }
    }

    /// Pins the Clifford left-contraction sign convention (`B ⌋ v`, not the
    /// physics `ω × r`) so future `Bivector5`/`Bivector6` stay consistent.
    #[test]
    fn bivector4_contract_vec_is_clifford_left_contraction() {
        // e_xy ⌋ e_x = -e_y, e_xy ⌋ e_y = +e_x.
        let b = Bivector4::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_vec4_close_tol(b.contract_vec(Vec4::X), -Vec4::Y, 1e-6);
        assert_vec4_close_tol(b.contract_vec(Vec4::Y), Vec4::X, 1e-6);
        // e_zw ⌋ e_z = -e_w; e_zw ⌋ e_w = +e_z.
        let b = Bivector4::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        assert_vec4_close_tol(b.contract_vec(Vec4::Z), -Vec4::W, 1e-6);
        assert_vec4_close_tol(b.contract_vec(Vec4::W), Vec4::Z, 1e-6);
        // Vanishes when v lies outside B's plane.
        let b = Bivector4::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0); // e_xy
        assert_vec4_close_tol(b.contract_vec(Vec4::Z), Vec4::ZERO, 1e-6);
        assert_vec4_close_tol(b.contract_vec(Vec4::W), Vec4::ZERO, 1e-6);
    }

    /// `Plane4::ALL[i].unit_bivector()` and `Bivector4::basis(i)` both agree
    /// with `Bivector4`'s field ordering.
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

    /// `From<Rotor4> for [f32; 8]` packs as `[s, xy, xz, xw, yz, yw, zw,
    /// xyzw]`, the order GPU uniform buffers consume.
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
