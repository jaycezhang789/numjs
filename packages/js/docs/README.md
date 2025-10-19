# Documentation Index

Welcome to the extended documentation for `@jayce789/numjs`. The material is grouped by topic so you can deep dive into the subsystems you care about—backends, sparse support, WebGPU acceleration, and future roadmap items. Every guide is kept in sync with the latest published package.

- **Migration Guide** – [`From NumPy to NumJS`](./tutorials/from-numpy-migration.md) walks through the one-to-one API mapping and outlines a workflow for porting existing Python notebooks or services.
- **Backend Architecture** – [`Choosing between WASM and N-API`](./tutorials/backends.md) explains how the dual-backend loader works, how the distribution packages are organised, and how to override the automatic detection.
- **WebGPU Acceleration** – [`WebGPU Tutorial`](./tutorials/webgpu.md) shows how to initialise the experimental GPU pipeline in browsers or Node.js and how to chain GPU kernels effectively.
- **Interactive Playground** – [`StackBlitz / CodeSandbox guide`](./interactive/README.md) lists ready-to-run sandboxes, the included demo scripts, and tips for sharing sessions with your team.
- **Future Roadmap** – [`Autograd, Randomness, and more`](./future.md) describes planned features, design constraints, and how you can influence the prioritisation.
- **Sparse Matrix Design** – [`SparseMatrix Architecture`](./design/sparse-matrix.md) documents the CSR-first data model, backend dispatch, and the SuiteSparse integration plan.

> We welcome feedback and corrections through GitHub Issues or pull requests. Each guide links back to the relevant example in the `examples/` directory so you can try the code locally.
