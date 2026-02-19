use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::common::rng;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Default)]
pub enum DepthDistribution {
    /// Uniformly sample an integer depth in [min_steps, max_steps].
    #[default]
    Uniform,
}

/// Mixture-of-Depths: sample a compute depth (number of refinement steps)
/// during training to encourage depth diversity and reduce expected compute.
///
/// This is intentionally simple and deterministic-seed friendly.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MixtureOfDepthsConfig {
    #[serde(default)]
    pub enabled: bool,

    #[serde(default = "default_min_steps")]
    pub min_steps: usize,

    #[serde(default = "default_max_steps")]
    pub max_steps: usize,

    #[serde(default)]
    pub distribution: DepthDistribution,
}

fn default_min_steps() -> usize {
    1
}

fn default_max_steps() -> usize {
    0
}

impl Default for MixtureOfDepthsConfig {
    fn default() -> Self {
        Self {
            // Enabled by default: this acts like a mild stochastic-depth mechanism for
            // a latent refinement loop, but it only applies during training.
            enabled: true,
            min_steps: default_min_steps(),
            // 0 means "use the caller's max".
            max_steps: default_max_steps(),
            distribution: DepthDistribution::default(),
        }
    }
}

impl MixtureOfDepthsConfig {
    /// Sample a max depth for the current forward pass.
    ///
    /// `hard_max` is the model's configured limit (e.g. max_supervision_steps).
    pub fn sample_depth_cap(&self, hard_max: usize) -> usize {
        if !self.enabled {
            return hard_max;
        }

        let effective_max = if self.max_steps == 0 {
            hard_max
        } else {
            self.max_steps.min(hard_max)
        };

        let min_steps = self.min_steps.min(effective_max).max(1);
        let max_steps = effective_max.max(min_steps);

        match self.distribution {
            DepthDistribution::Uniform => {
                let mut r = rng::get_rng();
                r.random_range(min_steps..=max_steps)
            }
        }
    }
}
