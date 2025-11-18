# DiffusionBlock Backlog (Long-term)

## High Priority
- [ ] Full discrete masked diffusion impl (Diffusion-LM/LLaDA style token diffusion)
- [ ] GPU acceleration (wgpu/bytemuck integration)
- [ ] Advanced noise schedules (VP, EDM, improved cosine)
- [ ] Property-based tests: diffusion math invariants, sampling equiv (DDIM vs posterior)

## Medium Priority
- [ ] Formal theorems/proofs in rustdoc: stability (SNR bounds), convergence (min-SNR), v-pred equiv
- [ ] EMA full verification tests (sampling equiv main weights)
- [ ] Benchmarks: diffusion vs transformer_block (loss curves, sample quality)
- [ ] Rayon outer loops (batch forward/backward)

## Low Priority
- [ ] Integration with eprop (eligibility traces for diffusion)
- [ ] MoE experts specialized for timesteps
- [ ] Curriculum learning (timestep_strategy advanced)
- [ ] Discrete + continuous hybrid

Prioritized by math correctness → perf → features.
