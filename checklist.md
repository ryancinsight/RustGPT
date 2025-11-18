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
