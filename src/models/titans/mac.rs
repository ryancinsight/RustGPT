use serde::{Deserialize, Serialize};
use ndarray::{Array2, s};
use crate::network::Layer;
use crate::attention::poly_attention::PolyAttention;
use crate::models::titans::memory::NeuralMemory;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Memory As Context (MAC) Architecture
///
/// "We treat the memory as a context to the current information."
/// Segment-based approach where memory processes past segment and output is concatenated
/// with current segment input to attention.
#[derive(Serialize, Deserialize, Debug)]
pub struct TitansMAC {
    // Core branch (Attention)
    pub core: PolyAttention,

    // Long-term Memory branch (NeuralMemory)
    pub memory: NeuralMemory,

    // Persistent Memory parameters (Learnable)
    // Dimension: (persistent_len, input_dim)
    pub persistent_memory: Array2<f32>,

    pub segment_len: usize,
    pub persistent_len: usize,
}

impl TitansMAC {
    pub fn new(
        core: PolyAttention,
        memory: NeuralMemory,
        persistent_len: usize,
        segment_len: usize,
    ) -> Self {
        let input_dim = core.embed_dim;
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        let p_vec: Vec<f32> = (0..persistent_len * input_dim).map(|_| normal.sample(&mut rng)).collect();
        let persistent_memory = Array2::from_shape_vec((persistent_len, input_dim), p_vec).unwrap();

        Self {
            core,
            memory,
            persistent_memory,
            segment_len,
            persistent_len,
        }
    }

    // Helper to retrieve and concat
    fn process_segment(&mut self, segment: &Array2<f32>) -> Array2<f32> {
        // 1. Retrieve h_t from Memory using input context (segment) as query.
        let h_t = self.memory.retrieve(segment);

        // 2. Concatenate [Persistent | h_t | Segment_t]
        // persistent: (P, D)
        // h_t: (S, D) (retrieved for each token in segment)
        // segment: (S, D)
        // Concat along sequence dimension (axis 0).

        let p_len = self.persistent_len;
        let s_len = segment.nrows();
        let d = segment.ncols();
        let total_len = p_len + s_len + s_len;

        let mut context_input = Array2::<f32>::zeros((total_len, d));

        context_input.slice_mut(s![0..p_len, ..]).assign(&self.persistent_memory);
        context_input.slice_mut(s![p_len..p_len+s_len, ..]).assign(&h_t);
        context_input.slice_mut(s![p_len+s_len..total_len, ..]).assign(segment);

        // 3. Pass to Attention
        let attention_output = self.core.forward(&context_input);

        // Output of attention corresponds to the concatenated sequence.
        // We typically care about the output corresponding to the *Segment*.
        // We extract the last S rows.

        let segment_output = attention_output.slice(s![p_len+s_len..total_len, ..]).to_owned();

        // 4. Update Memory using Attention output (segment part)
        self.memory.update(&segment_output);

        segment_output
    }
}

impl Layer for TitansMAC {
    fn layer_type(&self) -> &str {
        "TitansMAC"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // 1. Chunk sequence into segments.
        let seq_len = input.nrows();
        let input_dim = input.ncols();

        // We collect outputs for segments.
        let mut outputs = Vec::new();
        let mut processed = 0;

        while processed < seq_len {
            let end = std::cmp::min(processed + self.segment_len, seq_len);
            let segment = input.slice(s![processed..end, ..]).to_owned();

            let seg_out = self.process_segment(&segment);
            outputs.push(seg_out);

            processed = end;
        }

        // Concatenate outputs
        if outputs.is_empty() {
            return Array2::zeros((0, input_dim));
        }

        let total_rows: usize = outputs.iter().map(|a| a.nrows()).sum();
        let mut result = Array2::<f32>::zeros((total_rows, input_dim));

        let mut cursor = 0;
        for out in outputs {
            let rows = out.nrows();
            result.slice_mut(s![cursor..cursor+rows, ..]).assign(&out);
            cursor += rows;
        }

        result
    }

    fn backward(&mut self, grads: &Array2<f32>, _lr: f32) -> Array2<f32> {
        // TODO: Implement backward for MAC.
        Array2::zeros((grads.nrows(), grads.ncols()))
    }

    fn parameters(&self) -> usize {
        self.core.parameters() + self.memory.parameters() + self.persistent_memory.len()
    }

    fn weight_norm(&self) -> f32 {
         let mut sum_sq = 0.0;
         sum_sq += self.core.weight_norm().powi(2);
         sum_sq += self.memory.weight_norm().powi(2);
         sum_sq += self.persistent_memory.mapv(|x| x*x).sum();
         sum_sq.sqrt()
    }

    fn compute_gradients(&self, input: &Array2<f32>, _output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        // TODO: Implement
        (Array2::zeros(input.raw_dim()), Vec::new())
    }

    fn apply_gradients(&mut self, _gradients: &[Array2<f32>], _learning_rate: f32) -> crate::errors::Result<()> {
        // TODO: Implement
        Ok(())
    }

    fn zero_gradients(&mut self) {
        self.core.zero_gradients();
        self.memory.zero_gradients();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use crate::attention::poly_attention::PolyAttention;
    use crate::models::titans::memory::NeuralMemory;

    #[test]
    fn test_titans_mac_forward() {
        let input_dim = 16;
        let num_heads = 4;
        let memory_hidden_dim = 8;
        let segment_len = 4;
        let persistent_len = 2;

        let poly = PolyAttention::new(input_dim, num_heads, 3, 64, None);
        let memory = NeuralMemory::new(input_dim, input_dim, input_dim, memory_hidden_dim);

        let mut mac = TitansMAC::new(poly, memory, persistent_len, segment_len);

        // Input: (8, 16)
        let seq_len = 8;
        let input = Array2::<f32>::zeros((seq_len, input_dim));

        let output = mac.forward(&input);

        assert_eq!(output.dim(), (seq_len, input_dim));
    }
}
