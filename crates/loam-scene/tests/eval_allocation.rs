//! A baked collider grid evaluates the scene SDF once per cell, so the
//! evaluators must never touch the heap. Pinned with a counting global
//! allocator, which is process-wide and therefore lives in its own test binary
//! with exactly one test in it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use glam::{Vec3, Vec4};
use loam_math::{EuclideanR3, HyperbolicH3};
use loam_scene::{Scene, Scene4, SceneNode, SceneNode4};

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;

#[test]
fn evaluating_a_scene_never_touches_the_heap() {
    let scene = Scene::new(
        SceneNode::sphere(Vec3::new(0.1, 0.0, 0.05), 0.22)
            .smooth_union(SceneNode::box_(Vec3::new(0.2, 0.15, 0.25)), 0.08)
            .union(SceneNode::plane(Vec3::Y, -0.3))
            .subtract(SceneNode::sphere(Vec3::new(-0.15, 0.08, 0.0), 0.18))
            .intersect(SceneNode::cube(0.7)),
    );
    let scene4 = Scene4::new(
        SceneNode4::hypersphere(Vec4::new(0.1, 0.0, -0.05, 0.0), 0.5)
            .union(SceneNode4::halfspace(Vec4::Y, -0.4))
            .subtract(SceneNode4::hypersphere(Vec4::new(0.3, 0.1, 0.0, 0.1), 0.2)),
    );

    // Warm every lazily initialised path (formatting, TLS) before arming the
    // counter, so what it observes is only the evaluation loop.
    let mut checksum = scene.eval(&EuclideanR3, Vec3::ZERO)
        + scene.eval(&HyperbolicH3, Vec3::ZERO)
        + scene4.eval(Vec3::ZERO, 0.0, true);

    let before = ALLOCATIONS.load(Ordering::Relaxed);
    const STEPS: i32 = 16;
    for ix in 0..STEPS {
        for iy in 0..STEPS {
            for iz in 0..STEPS {
                let p = Vec3::new(
                    ix as f32 / STEPS as f32 - 0.5,
                    iy as f32 / STEPS as f32 - 0.5,
                    iz as f32 / STEPS as f32 - 0.5,
                );
                checksum += scene.eval(&EuclideanR3, p);
                checksum += scene.eval(&HyperbolicH3, p * 0.5);
                let (dist, kind) = scene4.eval_at(p, 0.25, true);
                checksum += dist + kind as f32;
            }
        }
    }
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;

    assert!(
        checksum.is_finite(),
        "guard against the loop being optimised away",
    );
    assert_eq!(
        allocations,
        0,
        "{} evaluations allocated {allocations} times",
        STEPS * STEPS * STEPS * 3,
    );
}
