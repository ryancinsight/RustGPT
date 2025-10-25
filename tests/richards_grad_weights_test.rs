use approx::assert_abs_diff_eq;
use llm::richards::{RichardsCurve, Variant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParamKind { Nu, K, M, Beta, A, B, Scale, Shift }

fn learnable_kinds(curve: &RichardsCurve) -> Vec<ParamKind> {
    let mut kinds = Vec::new();
    if curve.nu_learnable { kinds.push(ParamKind::Nu); }
    if curve.k_learnable { kinds.push(ParamKind::K); }
    if curve.m_learnable { kinds.push(ParamKind::M); }
    if curve.beta_learnable { kinds.push(ParamKind::Beta); }
    if curve.a_learnable { kinds.push(ParamKind::A); }
    if curve.b_learnable { kinds.push(ParamKind::B); }
    if curve.scale_learnable { kinds.push(ParamKind::Scale); }
    if curve.shift_learnable { kinds.push(ParamKind::Shift); }
    kinds
}

fn set_param_by_index(curve: &mut RichardsCurve, index: usize, value: f64) {
    let mut i = 0usize;
    if curve.nu_learnable {
        if i == index { curve.learned_nu = Some(value); return; } else { i += 1; }
    }
    if curve.k_learnable {
        if i == index { curve.learned_k = Some(value); return; } else { i += 1; }
    }
    if curve.m_learnable {
        if i == index { curve.learned_m = Some(value); return; } else { i += 1; }
    }
    if curve.beta_learnable {
        if i == index { curve.learned_beta = Some(value); return; } else { i += 1; }
    }
    if curve.a_learnable {
        if i == index { curve.learned_a = Some(value); return; } else { i += 1; }
    }
    if curve.b_learnable {
        if i == index { curve.learned_b = Some(value); return; } else { i += 1; }
    }
    if curve.scale_learnable {
        if i == index { curve.learned_scale = Some(value); return; } else { i += 1; }
    }
    if curve.shift_learnable {
        if i == index { curve.learned_shift = Some(value); return; } else { /* i += 1; */ }
    }
}

fn get_param_by_index(curve: &RichardsCurve, index: usize) -> f64 {
    let mut i = 0usize;
    if curve.nu_learnable {
        if i == index { return curve.learned_nu.unwrap_or(1.0); } else { i += 1; }
    }
    if curve.k_learnable {
        if i == index { return curve.learned_k.unwrap_or(1.0); } else { i += 1; }
    }
    if curve.m_learnable {
        if i == index { return curve.learned_m.unwrap_or(0.0); } else { i += 1; }
    }
    if curve.beta_learnable {
        if i == index { return curve.learned_beta.unwrap_or(1.0); } else { i += 1; }
    }
    if curve.a_learnable {
        if i == index { return curve.learned_a.unwrap_or(1.0); } else { i += 1; }
    }
    if curve.b_learnable {
        if i == index { return curve.learned_b.unwrap_or(0.0); } else { i += 1; }
    }
    if curve.scale_learnable {
        if i == index { return curve.learned_scale.unwrap_or(1.0); } else { i += 1; }
    }
    if curve.shift_learnable {
        if i == index { return curve.learned_shift.unwrap_or(0.0); } else { /* i += 1; */ }
    }
    unreachable!("Index out of range for learnable parameters");
}

fn finite_difference_grad(curve: &RichardsCurve, x: f64, index: usize, eps: f64) -> f64 {
    let base = get_param_by_index(curve, index);

    let mut curve_plus = curve.clone();
    set_param_by_index(&mut curve_plus, index, base + eps);
    let f_plus = curve_plus.forward_scalar(x);

    let mut curve_minus = curve.clone();
    set_param_by_index(&mut curve_minus, index, base - eps);
    let f_minus = curve_minus.forward_scalar(x);

    (f_plus - f_minus) / (2.0 * eps)
}

fn init_learned_params(curve: &mut RichardsCurve) {
    if curve.nu_learnable { curve.learned_nu = Some(1.0); }
    if curve.k_learnable { curve.learned_k = Some(1.0); }
    if curve.m_learnable { curve.learned_m = Some(0.0); }
    if curve.beta_learnable { curve.learned_beta = Some(1.0); }
    if curve.a_learnable { curve.learned_a = Some(1.0); }
    if curve.b_learnable { curve.learned_b = Some(0.0); }
    if curve.scale_learnable { curve.learned_scale = Some(1.0); }
    if curve.shift_learnable { curve.learned_shift = Some(0.0); }
}

#[test]
fn grad_weights_matches_finite_difference_sigmoid() {
    let mut curve = RichardsCurve::new_learnable(Variant::Sigmoid);
    init_learned_params(&mut curve);

    let x = 0.37;
    let eps = 1e-4;
    let tol = 5e-4;

    let analytic = curve.grad_weights_scalar(x, 1.0);
    let kinds = learnable_kinds(&curve);

    assert_eq!(analytic.len(), kinds.len());

    for (j, kind) in kinds.iter().enumerate() {
        let numeric = finite_difference_grad(&curve, x, j, eps);
        if *kind == ParamKind::Beta {
            assert_abs_diff_eq!(analytic[j], 0.0, epsilon = 1e-8);
            assert_abs_diff_eq!(numeric, 0.0, epsilon = tol);
        } else {
            assert_abs_diff_eq!(analytic[j], numeric, epsilon = tol);
        }
    }
}

#[test]
fn grad_weights_matches_finite_difference_tanh() {
    let mut curve = RichardsCurve::new_learnable(Variant::Tanh);
    init_learned_params(&mut curve);

    let x = -0.53;
    let eps = 1e-4;
    let tol = 7e-4; // slightly looser due to outer/input scaling

    let analytic = curve.grad_weights_scalar(x, 1.0);
    let kinds = learnable_kinds(&curve);

    assert_eq!(analytic.len(), kinds.len());

    for (j, kind) in kinds.iter().enumerate() {
        let numeric = finite_difference_grad(&curve, x, j, eps);
        if *kind == ParamKind::Beta {
            assert_abs_diff_eq!(analytic[j], 0.0, epsilon = 1e-8);
            assert_abs_diff_eq!(numeric, 0.0, epsilon = tol);
        } else {
            assert_abs_diff_eq!(analytic[j], numeric, epsilon = tol);
        }
    }
}

#[test]
fn grad_weights_matches_finite_difference_none_variant_includes_a_b() {
    let mut curve = RichardsCurve::new_learnable(Variant::None);
    init_learned_params(&mut curve);

    let x = 0.11;
    let eps = 1e-4;
    let tol = 5e-4;

    let analytic = curve.grad_weights_scalar(x, 1.0);
    let kinds = learnable_kinds(&curve);

    assert_eq!(analytic.len(), kinds.len());

    for (j, kind) in kinds.iter().enumerate() {
        let numeric = finite_difference_grad(&curve, x, j, eps);
        if *kind == ParamKind::Beta {
            assert_abs_diff_eq!(analytic[j], 0.0, epsilon = 1e-8);
            assert_abs_diff_eq!(numeric, 0.0, epsilon = tol);
        } else {
            assert_abs_diff_eq!(analytic[j], numeric, epsilon = tol);
        }
    }
}