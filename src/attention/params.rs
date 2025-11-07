/// Parameter information tracking for attention layers
/// Provides detailed breakdown of parameter counts for different components
#[derive(Debug, Clone)]
pub struct PolyAttentionParamInfo {
    /// Parameter count per head (w_q, w_k, w_v)
    pub head_params_per_head: usize,
    /// Total head parameters (all heads)
    pub head_params_total: usize,
    /// Output projection parameters
    pub output_projection_params: usize,
    /// Polynomial parameters (a, b, scale)
    pub polynomial_params: usize,
    /// Gating parameters (w_g, alpha_g, beta_g)
    pub gating_params: usize,
    /// Richards curve parameters for gating
    pub gate_poly_params: usize,
    /// Threshold predictor parameters (if present)
    pub threshold_predictor_params: usize,
    /// CoPE parameters
    pub cope_params: usize,
    /// Total parameter count
    pub total_params: usize,
}

impl PolyAttentionParamInfo {
    /// Create a new parameter info instance with calculated parameter counts
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        head_params_per_head: usize,
        gate_poly_params: usize,
        threshold_predictor_params: usize,
        cope_params: usize,
    ) -> Self {
        let head_params_total = head_params_per_head * num_heads;
        let output_projection_params = embed_dim * embed_dim;
        let polynomial_params = 3; // a, b, scale
        let gating_params = embed_dim * num_heads + 2 * num_heads; // w_g + alpha_g + beta_g

        let total_params = head_params_total
            + output_projection_params
            + polynomial_params
            + gating_params
            + gate_poly_params
            + threshold_predictor_params
            + cope_params;

        Self {
            head_params_per_head,
            head_params_total,
            output_projection_params,
            polynomial_params,
            gating_params,
            gate_poly_params,
            threshold_predictor_params,
            cope_params,
            total_params,
        }
    }

    /// Get a detailed breakdown of parameter counts as a formatted string
    pub fn breakdown(&self) -> String {
        format!(
            "PolyAttention Parameter Breakdown:\n\
             • Head parameters per head: {}\n\
             • Total head parameters: {}\n\
             • Output projection: {}\n\
             • Polynomial parameters: {}\n\
             • Gating parameters: {}\n\
             • Gate polynomial: {}\n\
             • Threshold predictor: {}\n\
             • CoPE parameters: {}\n\
             • Total parameters: {}",
            self.head_params_per_head,
            self.head_params_total,
            self.output_projection_params,
            self.polynomial_params,
            self.gating_params,
            self.gate_poly_params,
            self.threshold_predictor_params,
            self.cope_params,
            self.total_params
        )
    }
}

impl Default for PolyAttentionParamInfo {
    fn default() -> Self {
        Self {
            head_params_per_head: 0,
            head_params_total: 0,
            output_projection_params: 0,
            polynomial_params: 0,
            gating_params: 0,
            gate_poly_params: 0,
            threshold_predictor_params: 0,
            cope_params: 0,
            total_params: 0,
        }
    }
}
