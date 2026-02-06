use serde::{Deserialize, Serialize};

/// Policy for selecting the number of DDIM reverse steps at sampling time.
///
/// This exists to avoid hardcoding a magic default (e.g. 100) while still allowing
/// CLI overrides and checkpoint-stable defaults.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DdimStepsPolicy {
    /// Use an explicit fixed number of steps.
    Fixed(usize),

    /// Choose a step count from model/usage context.
    ///
    /// Heuristic: start from ~T/10 (so 100 for T=1000) and scale softly with
    /// max sequence length and prompt ratio, then clamp.
    Auto { min_steps: usize, max_steps: usize },
}

impl Default for DdimStepsPolicy {
    fn default() -> Self {
        DdimStepsPolicy::Auto {
            min_steps: 16,
            max_steps: 256,
        }
    }
}

impl DdimStepsPolicy {
    pub fn resolve(&self, total_timesteps: usize, max_length: usize, prompt_len: usize) -> usize {
        let total = total_timesteps.max(1);

        match *self {
            DdimStepsPolicy::Fixed(k) => k.max(1).min(total),
            DdimStepsPolicy::Auto {
                min_steps,
                max_steps,
            } => {
                // Base: ~T/10 (100 for T=1000). This preserves the old behavior scale
                // without hardcoding an exact constant.
                let base = (total as f32 / 10.0).round().max(1.0);

                // Scale with sequence length (sqrt keeps it gentle).
                let len_scale = ((max_length.max(1) as f32) / 256.0).sqrt().clamp(0.5, 2.0);

                // Slightly increase steps if prompt occupies most of the sequence.
                let prompt_ratio = if max_length > 0 {
                    (prompt_len as f32 / max_length as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let prompt_scale = 1.0 + 0.25 * prompt_ratio;

                let mut steps = (base * len_scale * prompt_scale).round() as usize;

                let min_s = min_steps.max(1);
                let max_s = max_steps.max(min_s);
                steps = steps.clamp(min_s, max_s);
                steps.min(total)
            }
        }
    }
}

/// Map a step index in `[0, steps-1]` to a diffusion timestep in `[0, total_timesteps-1]`.
pub fn map_step_to_timestep(step_idx: usize, steps: usize, total_timesteps: usize) -> usize {
    let steps = steps.max(1);
    let total = total_timesteps.max(1);
    if steps <= 1 {
        return 0;
    }
    let denom = (steps - 1) as f32;
    let frac = (step_idx as f32) / denom;
    let t = (frac * (total - 1) as f32).round() as isize;
    t.clamp(0, (total - 1) as isize) as usize
}
