//! Linear render-graph harness. Owns a [`Vec<Box<dyn RenderNode>>`] and runs each
//! node's `execute` in order against a single view.
//!
//! No dependency tracking, no cross-node resource sharing, no parallel execution. The
//! trait is intentionally a one-method `execute(rd, view)`: complex apps that need
//! scheduling or aliasing should compose nodes manually rather than waiting for this
//! graph to grow features.

use crate::device::RenderDevice;
use anyhow::Result;

/// One render-pass-or-equivalent unit of work the graph runs.
pub trait RenderNode {
    /// Static label used for tracing / debug output. Returned by reference because most
    /// impls hand back a string literal.
    fn name(&self) -> &'static str;

    /// Run the node against the given device + target view. Errors abort the rest of the
    /// graph.
    fn execute(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()>;
}

/// Linear sequence of [`RenderNode`]s. Construction is fluent ([`RenderGraph::add_node`]);
/// execution runs them in order against the same view ([`RenderGraph::run`]).
#[derive(Default)]
pub struct RenderGraph {
    nodes: Vec<Box<dyn RenderNode>>,
}

impl RenderGraph {
    /// Empty graph; pair with chained [`Self::add_node`] calls.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a node and return self by value so calls chain.
    pub fn add_node<N: RenderNode + 'static>(mut self, node: N) -> Self {
        self.nodes.push(Box::new(node));
        self
    }

    /// Run every node in insertion order, propagating the first error the graph encounters
    /// and aborting subsequent nodes.
    pub fn run(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        for n in &mut self.nodes {
            n.execute(rd, view)?;
        }
        Ok(())
    }
}
