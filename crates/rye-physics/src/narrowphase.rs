//! Narrowphase collision dispatch table.
//!
//! `Narrowphase<S>` is a `HashMap` keyed by `(ColliderKind, ColliderKind)` with entries that
//! are function pointers. When a broadphase candidate pair is tested, the narrowphase looks up
//! the entry for the bodies' collider kinds and calls it.
//!
//! This is the core extension point for adding new collider types, new spaces, or new collision
//! algorithms without modifying existing code. To add H³ horosphere colliders: define
//! `Collider::Horosphere`, add `ColliderKind::Horosphere`, and register `sphere_horosphere` /
//! `horosphere_horosphere` functions. Nothing else changes.

use std::collections::HashMap;

use crate::body::RigidBody;
use crate::collider::ColliderKind;
use crate::integrator::PhysicsSpace;
use crate::response::Contact;

/// A narrowphase collision function. Returns `Some(contact)` if bodies `a` and `b` overlap,
/// `None` otherwise. Always called with `a.kind()` matching the key's first component.
pub type NarrowphaseFn<S> = fn(a: &RigidBody<S>, b: &RigidBody<S>, space: &S) -> Option<Contact<S>>;

/// Registry of narrowphase functions, keyed by the collider kinds of both bodies.
pub struct Narrowphase<S: PhysicsSpace> {
    dispatch: HashMap<(ColliderKind, ColliderKind), NarrowphaseFn<S>>,
}

impl<S: PhysicsSpace> Default for Narrowphase<S> {
    fn default() -> Self {
        Self {
            dispatch: HashMap::new(),
        }
    }
}

impl<S: PhysicsSpace> Narrowphase<S> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a narrowphase function for a specific collider pair. Registering a new pair is
    /// additive; registering over an existing pair replaces it.
    pub fn register(&mut self, a: ColliderKind, b: ColliderKind, f: NarrowphaseFn<S>) {
        self.dispatch.insert((a, b), f);
    }

    /// Look up and call the narrowphase function for this pair. Returns `None` if no function
    /// is registered.
    pub fn test(&self, a: &RigidBody<S>, b: &RigidBody<S>, space: &S) -> Option<Contact<S>>
    where
        S::Vector: std::ops::Mul<f32, Output = S::Vector>,
    {
        let key = (a.collider.kind(), b.collider.kind());
        if let Some(&f) = self.dispatch.get(&key) {
            return f(a, b, space);
        }
        // Try the reversed order, symmetry lets us register only one direction per pair if the
        // function handles both.
        let reversed = (b.collider.kind(), a.collider.kind());
        if let Some(&f) = self.dispatch.get(&reversed) {
            // Flip bodies so the registered function sees the kinds it expects; flip the
            // contact normal on the way out. The contact point is in world space and does not
            // need to be flipped.
            return f(b, a, space).map(|c| Contact {
                normal: c.normal * -1.0,
                point: c.point,
                penetration: c.penetration,
                restitution: c.restitution,
            });
        }
        None
    }
}
