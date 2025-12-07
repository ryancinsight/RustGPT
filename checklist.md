# DiffusionBlock Enhancement Sprint Checklist

Phase 1 (Audit/Planning) Complete

- [x] Read/analyze diffusion_block.rs
- [x] Review project docs (gap_audit, README)
- [x] Audit: math correctness (DDPM exact), invariants (clip/sanitize), backward full, tests good
- [x] Research: diffusion transformers (Diffusion-LM/LLaDA), speculation (Speculative Diffusion Sampling: draft small/verify large for faster reverse process)

Current Phase 2 (10-50%): Implement plan

- [ ] Update gap_audit.md with findings (no speculation, discrete stub, theorems missing)
- [ ] Design: add SpecDraftBlock (small), speculative_sample() with tree accept/reject
- [ ] Implement speculation in DiffusionBlock.sample()
- [ ] Add property tests (diffusion math, EMA equiv)
- [ ] Formalize theorems (stability, convergence SNR)
- [ ] GPU/rayon perf opts

Phase 3: Verify/docs

- [ ] Benchmarks vs baseline (training logs)
- [ ] Update rustdoc/theorems
- [ ] Close sprint, new task

## TransformerBlock Audit/Enhancement Sprint
- [x] Audit complete, gaps to gap_audit (TB-001..006)
- [ ] Integrate TransformerWorkspace (TB-001 Major perf)
- [ ] Prop/unit tests theorems/adaptive/MoE (TB-002/3/4)
- [ ] Extract WindowAdapter/GradPartitioner traits (TB-005)
- [ ] Par forward/RwLock→low-contention (TB-006)
- [ ] Bench validate + docs sync

## Diffusion Training Gradient NaN Fix
- [x] Analyze gradient NaN error in diffusion training
- [x] Identify root cause (numerical instability in V-prediction gradient scaling)
- [x] Check gradient computation and sanitization
- [x] Implement fixes for numerical stability (bounds checking, input validation, post-scaling sanitization)
- [x] Test compilation fixes
- [x] Update gap audit with findings
