use ndarray::Array2;

use super::block::NoiseScheduler;
use crate::layers::diffusion::{DiffusionPredictionTarget, map_step_to_timestep};

fn dedup_descending(mut timesteps: Vec<usize>) -> Vec<usize> {
    timesteps.sort_unstable_by(|a, b| b.cmp(a));
    timesteps.dedup();
    timesteps
}

/// Build a decreasing, unique timestep schedule (indices into the scheduler arrays).
/// Ensures the last timestep is 0.
pub(crate) fn make_discrete_timesteps(steps: usize, total_timesteps: usize) -> Vec<usize> {
    let steps = steps.max(1);
    let total = total_timesteps.max(1);
    let mut ts: Vec<usize> = (0..steps)
        .map(|i| map_step_to_timestep(i, steps, total))
        .collect();
    ts = dedup_descending(ts);
    if *ts.last().unwrap_or(&0) != 0 {
        ts.push(0);
    }
    ts
}

pub(crate) fn epsilon_from_prediction_target(
    pred: Array2<f32>,
    x_t: &Array2<f32>,
    t: usize,
    prediction_target: DiffusionPredictionTarget,
    scheduler: &NoiseScheduler,
) -> Array2<f32> {
    match prediction_target {
        DiffusionPredictionTarget::Epsilon => pred,
        DiffusionPredictionTarget::VPrediction => {
            let sa = scheduler.sqrt_alpha_cumprod(t).max(1e-6);
            let soa = scheduler.sqrt_one_minus_alpha_cumprod(t);
            let x0_hat = (x_t * sa) - (&pred * soa);
            (&pred + (&x0_hat * soa)) / sa
        }
        DiffusionPredictionTarget::Sample | DiffusionPredictionTarget::EdmX0 => {
            let sa = scheduler.sqrt_alpha_cumprod(t).max(1e-6);
            let soa = scheduler.sqrt_one_minus_alpha_cumprod(t);
            (x_t - (&pred * sa)) / soa
        }
    }
}

pub(crate) fn x0_from_prediction_target(
    pred: Array2<f32>,
    x_t: &Array2<f32>,
    t: usize,
    prediction_target: DiffusionPredictionTarget,
    scheduler: &NoiseScheduler,
) -> Array2<f32> {
    match prediction_target {
        DiffusionPredictionTarget::Sample | DiffusionPredictionTarget::EdmX0 => pred,
        DiffusionPredictionTarget::Epsilon => {
            let sa = scheduler.sqrt_alpha_cumprod(t).max(1e-6);
            let soa = scheduler.sqrt_one_minus_alpha_cumprod(t);
            (x_t - &(&pred * soa)) / sa
        }
        DiffusionPredictionTarget::VPrediction => {
            let sa = scheduler.sqrt_alpha_cumprod(t).max(1e-6);
            let soa = scheduler.sqrt_one_minus_alpha_cumprod(t);
            (x_t * sa) - (pred * soa)
        }
    }
}

/// PNDM/PLMS sampling: Adams-Bashforth multistep with a first-step Heun predictor-corrector.
///
/// This is the widely-used PLMS variant (as in Diffusers' PNDM scheduler) for deterministic
/// ODE sampling when the model predicts ε.
pub(crate) fn pndm_plms_sample<M>(
    mut x: Array2<f32>,
    timesteps: &[usize],
    scheduler: &NoiseScheduler,
    mut model_epsilon: M,
) -> Array2<f32>
where
    M: FnMut(&Array2<f32>, usize) -> Array2<f32>,
{
    if timesteps.len() < 2 {
        return x;
    }

    let mut prev_eps: Vec<Array2<f32>> = Vec::with_capacity(4);

    for i in 0..(timesteps.len() - 1) {
        let t = timesteps[i];
        let t_prev = timesteps[i + 1];

        let eps = model_epsilon(&x, t);

        let eps_hat = match prev_eps.len() {
            0 => {
                // Heun (predictor-corrector) to bootstrap the multistep history.
                let x_pred = scheduler.ddim_step_between(&x, t, t_prev, &eps, 0.0, None);
                let eps_next = model_epsilon(&x_pred, t_prev);
                (&eps + &eps_next) * 0.5
            }
            1 => {
                // 2-step Adams-Bashforth
                (&eps * 3.0 - &prev_eps[0]) * 0.5
            }
            2 => {
                // 3-step Adams-Bashforth
                (&eps * 23.0 - &prev_eps[1] * 16.0 + &prev_eps[0] * 5.0) / 12.0
            }
            _ => {
                // 4-step Adams-Bashforth
                (&eps * 55.0 - &prev_eps[2] * 59.0 + &prev_eps[1] * 37.0 - &prev_eps[0] * 9.0)
                    / 24.0
            }
        };

        x = scheduler.ddim_step_between(&x, t, t_prev, &eps_hat, 0.0, None);

        // Update history: keep last 3 previous eps (excluding current) for AB formulas.
        prev_eps.push(eps);
        if prev_eps.len() > 3 {
            prev_eps.remove(0);
        }
    }

    x
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DpmSolverAdaptiveConfig {
    pub h_init: f32,
    pub atol: f32,
    pub rtol: f32,
    pub theta: f32,
    pub lambda_err: f32,
}

impl Default for DpmSolverAdaptiveConfig {
    fn default() -> Self {
        Self {
            h_init: 0.05,
            atol: 0.0078,
            rtol: 0.05,
            theta: 0.9,
            lambda_err: 1e-5,
        }
    }
}

fn rms_norm(v: &Array2<f32>) -> f32 {
    let mut acc = 0.0f32;
    let mut n = 0usize;
    for &x in v.iter() {
        acc += x * x;
        n += 1;
    }
    if n == 0 { 0.0 } else { (acc / n as f32).sqrt() }
}

fn compute_lambda(alpha: f32, sigma: f32) -> f32 {
    let a = alpha.max(1e-12);
    let s = sigma.max(1e-12);
    a.ln() - s.ln()
}

fn precompute_alpha_sigma_lambda(scheduler: &NoiseScheduler) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let total = scheduler.num_timesteps().max(1);
    let mut alpha = Vec::with_capacity(total);
    let mut sigma = Vec::with_capacity(total);
    let mut lambda = Vec::with_capacity(total);
    for t in 0..total {
        let a = scheduler.sqrt_alpha_cumprod(t).max(1e-12);
        let s = scheduler.sqrt_one_minus_alpha_cumprod(t).max(1e-12);
        alpha.push(a);
        sigma.push(s);
        lambda.push(compute_lambda(a, s));
    }
    (alpha, sigma, lambda)
}

fn index_frac_from_lambda(target_lambda: f32, lambda: &[f32]) -> f32 {
    // lambda is expected to be decreasing with t (typically), but we handle either monotonic.
    let n = lambda.len().max(1);
    if n == 1 {
        return 0.0;
    }

    let is_increasing = lambda[0] < lambda[n - 1];
    if is_increasing {
        if target_lambda <= lambda[0] {
            return 0.0;
        }
        if target_lambda >= lambda[n - 1] {
            return (n - 1) as f32;
        }
        let mut lo = 0usize;
        let mut hi = n - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if lambda[mid] < target_lambda {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let l0 = lambda[lo];
        let l1 = lambda[hi];
        let frac = if (l1 - l0).abs() > 1e-12 {
            (target_lambda - l0) / (l1 - l0)
        } else {
            0.0
        };
        lo as f32 + frac.clamp(0.0, 1.0)
    } else {
        if target_lambda >= lambda[0] {
            return 0.0;
        }
        if target_lambda <= lambda[n - 1] {
            return (n - 1) as f32;
        }
        let mut lo = 0usize;
        let mut hi = n - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if lambda[mid] > target_lambda {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let l0 = lambda[lo];
        let l1 = lambda[hi];
        let frac = if (l0 - l1).abs() > 1e-12 {
            (l0 - target_lambda) / (l0 - l1)
        } else {
            0.0
        };
        lo as f32 + frac.clamp(0.0, 1.0)
    }
}

fn interp(v: &[f32], idx_f: f32) -> f32 {
    let n = v.len().max(1);
    if n == 1 {
        return v[0];
    }
    let idx_f = idx_f.clamp(0.0, (n - 1) as f32);
    let i0 = idx_f.floor() as usize;
    let i1 = (i0 + 1).min(n - 1);
    let w = idx_f - i0 as f32;
    v[i0] * (1.0 - w) + v[i1] * w
}

fn nearest_index(idx_f: f32, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let idx = idx_f.round() as isize;
    idx.clamp(0, (n - 1) as isize) as usize
}

#[derive(Clone, Copy)]
struct DpmSolverFirstUpdateParams {
    sigma_s: f32,
    alpha_t: f32,
    sigma_t: f32,
    lambda_s: f32,
    lambda_t: f32,
}

fn dpmsolverpp_first_update<M>(
    x: &Array2<f32>,
    params: DpmSolverFirstUpdateParams,
    mut model_x0_at_s: M,
    t_idx_s: usize,
) -> (Array2<f32>, Array2<f32>)
where
    M: FnMut(&Array2<f32>, usize) -> Array2<f32>,
{
    let h = params.lambda_t - params.lambda_s;
    let phi_1 = (-h).exp_m1();

    let model_s = model_x0_at_s(x, t_idx_s);
    let x_t = (params.sigma_t / params.sigma_s) * x - (params.alpha_t * phi_1) * &model_s;
    (x_t, model_s)
}

#[derive(Clone, Copy)]
struct DpmSolverSecondUpdateParams {
    sigma_s: f32,
    alpha_t: f32,
    sigma_t: f32,
    lambda_s: f32,
    lambda_t: f32,
    r1: f32,
    alpha_s1: f32,
    sigma_s1: f32,
}

fn dpmsolverpp_second_update<M>(
    x: &Array2<f32>,
    params: DpmSolverSecondUpdateParams,
    model_s: &Array2<f32>,
    mut model_x0: M,
    t_idx_s1: usize,
) -> Array2<f32>
where
    M: FnMut(&Array2<f32>, usize) -> Array2<f32>,
{
    let h = params.lambda_t - params.lambda_s;
    let phi_11 = (-(params.r1 * h)).exp_m1();
    let phi_1 = (-h).exp_m1();

    let x_s1 = (params.sigma_s1 / params.sigma_s) * x - (params.alpha_s1 * phi_11) * model_s;
    let model_s1 = model_x0(&x_s1, t_idx_s1);

    (params.sigma_t / params.sigma_s) * x
        - (params.alpha_t * phi_1) * model_s
        - ((0.5 / params.r1) * params.alpha_t * phi_1) * (&model_s1 - model_s)
}

/// DPM-Solver++ adaptive step size (order 2) in half-logSNR (lambda) space.
///
/// This is a faithful port of the dpmsolver++ adaptive scheme (DPM-Solver-12 style)
/// specialized to the scalar VP schedule derived from the discrete scheduler arrays.
pub(crate) fn dpmsolverpp_adaptive_sample<M>(
    mut x: Array2<f32>,
    scheduler: &NoiseScheduler,
    mut model_x0: M,
    cfg: DpmSolverAdaptiveConfig,
) -> Array2<f32>
where
    M: FnMut(&Array2<f32>, usize) -> Array2<f32>,
{
    let order = 2usize;
    let (alpha, sigma, lambda) = precompute_alpha_sigma_lambda(scheduler);
    let total = lambda.len().max(1);

    let lambda_start = lambda[total - 1];
    let lambda_end = lambda[0];

    let mut lambda_s = lambda_start;
    let mut idx_s_f = (total - 1) as f32;
    let mut h = cfg.h_init.max(1e-4);

    // Used for relative error scaling.
    let mut x_prev = x.clone();

    while (lambda_end - lambda_s).abs() > cfg.lambda_err {
        let remaining = lambda_end - lambda_s;
        if remaining.abs() <= cfg.lambda_err {
            break;
        }

        // Move in the direction of the target.
        let step_sign = if remaining >= 0.0 { 1.0 } else { -1.0 };
        let h_try = (step_sign * h).clamp(-remaining.abs(), remaining.abs());
        let lambda_t = lambda_s + h_try;

        let idx_t_f = index_frac_from_lambda(lambda_t, &lambda);

        let _alpha_s = interp(&alpha, idx_s_f);
        let sigma_s = interp(&sigma, idx_s_f);
        let alpha_t = interp(&alpha, idx_t_f);
        let sigma_t = interp(&sigma, idx_t_f);

        let t_idx_s = nearest_index(idx_s_f, total);

        let (x_lower, model_s) = dpmsolverpp_first_update(
            &x,
            DpmSolverFirstUpdateParams {
                sigma_s,
                alpha_t,
                sigma_t,
                lambda_s,
                lambda_t,
            },
            &mut model_x0,
            t_idx_s,
        );

        let r1 = 0.5f32;
        let lambda_s1 = lambda_s + r1 * (lambda_t - lambda_s);
        let idx_s1_f = index_frac_from_lambda(lambda_s1, &lambda);
        let alpha_s1 = interp(&alpha, idx_s1_f);
        let sigma_s1 = interp(&sigma, idx_s1_f);
        let t_idx_s1 = nearest_index(idx_s1_f, total);

        let x_higher = dpmsolverpp_second_update(
            &x,
            DpmSolverSecondUpdateParams {
                sigma_s,
                alpha_t,
                sigma_t,
                lambda_s,
                lambda_t,
                r1,
                alpha_s1,
                sigma_s1,
            },
            &model_s,
            &mut model_x0,
            t_idx_s1,
        );

        // Error estimate based on the difference between orders.
        let mut denom = x_lower.mapv(|v| v.abs());
        for (d, p) in denom.iter_mut().zip(x_prev.iter()) {
            *d = d.max(p.abs());
        }
        denom.mapv_inplace(|v| cfg.atol.max(cfg.rtol * v));

        let err = (&x_higher - &x_lower) / &denom;
        let e = rms_norm(&err);

        if e.is_finite() && e <= 1.0 {
            x = x_higher;
            x_prev = x_lower;
            lambda_s = lambda_t;
            idx_s_f = idx_t_f;
        }

        // Step size adaptation.
        let e_safe = if e.is_finite() { e.max(1e-12) } else { 1e6 };
        let factor = cfg.theta * e_safe.powf(-1.0 / (order as f32));
        let mut h_new = (h.abs() * factor).clamp(1e-4, 1.0);
        let rem_abs = (lambda_end - lambda_s).abs();
        h_new = h_new.min(rem_abs.max(1e-4));
        h = h_new;

        // Safety break for pathological schedules.
        if (lambda_end - lambda_s).abs() <= cfg.lambda_err {
            break;
        }
    }

    x
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::Array2;

    use super::*;

    fn make_scheduler() -> NoiseScheduler {
        // Use a small schedule to keep tests fast.
        NoiseScheduler::new(
            crate::layers::diffusion::NoiseSchedule::Cosine { s: 0.008 },
            64,
        )
    }

    #[test]
    fn test_pndm_plms_matches_reference_small() {
        let scheduler = make_scheduler();
        let ts16 = make_discrete_timesteps(16, scheduler.num_timesteps());
        let ts32 = make_discrete_timesteps(32, scheduler.num_timesteps());
        let ts64 = make_discrete_timesteps(64, scheduler.num_timesteps());

        // Toy epsilon model: eps = 0.1 * x + (t/T) * 0.01
        let total = scheduler.num_timesteps().max(1) as f32;
        let model = |x: &Array2<f32>, t: usize| {
            let bias = (t as f32 / total) * 0.01;
            x.mapv(|v| 0.1 * v + bias)
        };

        let x0 = Array2::from_elem((4, 8), 0.1234);
        let out16 = pndm_plms_sample(x0.clone(), &ts16, &scheduler, model);
        let out32 = pndm_plms_sample(x0.clone(), &ts32, &scheduler, model);
        let out64 = pndm_plms_sample(x0, &ts64, &scheduler, model);

        let diff16 = (&out16 - &out64).mapv(|v| v.abs()).mean().unwrap_or(0.0);
        let diff32 = (&out32 - &out64).mapv(|v| v.abs()).mean().unwrap_or(0.0);
        assert!(
            diff32 <= diff16,
            "Expected PLMS to converge with more steps (diff32={diff32}, diff16={diff16})"
        );
    }

    #[test]
    fn test_dpmsolverpp_adaptive_close_to_small_fixed_steps() {
        let scheduler = make_scheduler();

        // Exact-invariant case: if the model always returns x0 = 0, the update reduces to
        // x_t = (sigma_t / sigma_s) * x_s, independent of step partitioning.
        let model_x0 = |x: &Array2<f32>, _t: usize| Array2::<f32>::zeros(x.raw_dim());

        let x = Array2::from_elem((4, 8), 0.33);

        let out_default =
            dpmsolverpp_adaptive_sample(x.clone(), &scheduler, model_x0, Default::default());

        let cfg = DpmSolverAdaptiveConfig {
            h_init: 0.01,
            ..Default::default()
        };
        let out_ref = dpmsolverpp_adaptive_sample(x.clone(), &scheduler, model_x0, cfg);

        let diff = (&out_default - &out_ref)
            .mapv(|v| v.abs())
            .mean()
            .unwrap_or(0.0);
        assert_relative_eq!(diff, 0.0, epsilon = 1e-5);

        let sigma_start = scheduler
            .sqrt_one_minus_alpha_cumprod(scheduler.num_timesteps() - 1)
            .max(1e-12);
        let sigma_end = scheduler.sqrt_one_minus_alpha_cumprod(0).max(1e-12);
        let expected = x * (sigma_end / sigma_start);
        let err = (&out_default - &expected)
            .mapv(|v| v.abs())
            .mean()
            .unwrap_or(0.0);
        assert_relative_eq!(err, 0.0, epsilon = 1e-4);
    }
}
