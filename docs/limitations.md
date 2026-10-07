# Scope, limitations, and tradeoffs

## What the demonstration establishes

The repository has a real numerical learning loop and working release mechanics. Its models learn weights; its drift and accuracy numbers are calculated; its controlled regression modifies model weights; its registry verifies evidence before changing a pointer. The generated report is a saved evaluation result.

The task is intentionally small enough to inspect and run on an ordinary CPU. All data is synthetic, generated locally, and licensed as part of the project. There is no private employer code, customer data, pretrained model download, real CAD dataset, GPU measurement, or production traffic claim.

## Scientific limitations

- The three shape classes are simple, centered surface point clouds with modest variation. Real CAD models, occlusion, partial scans, topology, assemblies, and irregular sampling are absent.
- The baseline is deliberately sensitive to axes and units. Its comparison illustrates the effect of geometric invariance, not superiority over established 3D-learning methods.
- Only 54 latent groups are held out. Class/slice recalls use 18 groups each and are correspondingly coarse.
- The seed-42 fixture was used while developing features, thresholds, and tests. It is a regression benchmark, not an untouched external validation set. An independently sampled validation corpus is necessary before making a generalization claim.
- Group-bootstrap bounds summarize paired accuracy differences on this fixture. They do not account for choosing a model, feature set, or threshold after observing the fixture, and no multiple-testing correction is attempted.
- PSI is a descriptive, binned drift metric sensitive to sample size and binning. The threshold of 4.0 is deliberately permissive for the synthetic stress suite. It is not a universal drift standard.

## Engineering limitations

- Storage is local and POSIX-only (`fcntl.flock`, atomic rename, directory fsync). Distributed filesystem semantics, Windows support, service authentication, multi-tenancy, quotas, and object-store conditional writes are outside scope.
- Content addresses establish artifact consistency, not authenticity. Anyone authorized to rewrite code and the registry can manufacture an entirely new internally consistent history. There is no cryptographic signing or trusted training attestation.
- The registry binds recorded source/fitting hashes but does not prove the weights were produced by the documented trainer. Model derivations in the failure drills are intentionally manual and labeled as such.
- Report comparison is exact within the current numerical environment. A report generated on another platform may be rejected due to low-order floating-point differences; regenerate it where promotion runs.
- Object files become read-only after creation, and reads check their hashes. An interrupted transaction can leave orphan objects. No garbage collector, retention policy, serving process, or remote deployment is implemented.
- Rollback is an undo operation and can undo itself on the next call. It is not a command for selecting arbitrary historical versions.
- All records are processed in memory. The full fixture is hundreds of clouds, not a large-scale data pipeline. Published CPU timings describe this workload only.

## Why these choices

Inspectable JSON weights keep the artifact boundary small. A simple softmax classifier makes the evaluation and release machinery visible, rather than hiding it behind training infrastructure. A local registry makes crash consistency and concurrency testable without external accounts. Fixed seeds, numerical dependency pinning, and checked-in evidence make the claims straightforward to reproduce and challenge.
