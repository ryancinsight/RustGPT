//! EDM-style helpers for diffusion in embedding space.

/// Default EDM `sigma_data` used for preconditioning.
///
/// Common image-model defaults are ~0.5; for embedding-space diffusion we default to 1.0.
pub const EDM_SIGMA_DATA_DEFAULT: f32 = 1.0;

/// Serde default hook for `DiffusionBlockConfig::edm_sigma_data`.
#[inline]
pub fn diffusion_edm_sigma_data_default() -> f32 {
    EDM_SIGMA_DATA_DEFAULT
}

/// Convert VP-style cumulative alpha (`\bar{\alpha}`) to an EDM sigma.
///
/// Uses $\sigma^2 = \frac{1-\bar{\alpha}}{\bar{\alpha}}$.
#[inline]
pub fn sigma_from_alpha_bar(alpha_bar: f32) -> f32 {
    let alpha_bar = alpha_bar.clamp(1e-12, 1.0);
    (((1.0 - alpha_bar) / alpha_bar).max(0.0)).sqrt()
}

/// EDM preconditioning coefficients from $(\sigma, \sigma_{data})$.
///
/// Returns $(c_{in}, c_{skip}, c_{out})$.
#[inline]
pub fn precond_scales_from_sigma(sigma: f32, sigma_data: f32) -> (f32, f32, f32) {
    let sigma_data = sigma_data.max(1e-6);
    let denom = (sigma * sigma + sigma_data * sigma_data).max(1e-12);
    let c_in = 1.0 / denom.sqrt();
    let c_skip = (sigma_data * sigma_data) / denom;
    let c_out = (sigma * sigma_data) / denom.sqrt();
    (c_in, c_skip, c_out)
}

/// EDM loss weight for denoised (x0) objective.
///
/// From Karras et al. (EDM):
/// $w(\sigma) = \frac{\sigma^2 + \sigma_{data}^2}{(\sigma\,\sigma_{data})^2}$.
///
/// We clamp inputs to avoid singularities at very small $\sigma$.
pub fn loss_weight_from_sigma(sigma: f32, sigma_data: f32) -> f32 {
    let sigma = sigma.max(1e-6);
    let sigma_data = sigma_data.max(1e-6);
    let num = sigma * sigma + sigma_data * sigma_data;
    let den = (sigma * sigma) * (sigma_data * sigma_data);
    (num / den).max(0.0)
}
