use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EstimateInput {
    pub seq_len: usize,
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub poly_degree: usize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FlopsEstimate {
    pub flops_forward: u64,
    pub bytes_forward: u64,
}

pub fn estimate_transformer_block(inp: EstimateInput) -> FlopsEstimate {
    let hdim = inp.hidden_dim.max(inp.embed_dim);
    let seq = inp.seq_len as u64;
    let d = inp.embed_dim as u64;
    let heads = inp.num_heads as u64;
    let p = inp.poly_degree as u64;
    let attn = seq * d * heads * p * 4;
    let ffn = seq * d * (hdim as u64) * 2;
    let norms = seq * d * 4;
    let flops = attn + ffn + norms;
    let bytes = seq * d * 4 + seq * (hdim as u64) * 4;
    FlopsEstimate {
        flops_forward: flops,
        bytes_forward: bytes,
    }
}

pub fn estimate_diffusion_block(inp: EstimateInput, time_embed_dim: usize) -> FlopsEstimate {
    let base = estimate_transformer_block(inp);
    let seq = inp.seq_len as u64;
    let ted = time_embed_dim as u64;
    let time_mlp = ted * (ted.max(32)) * 2 + (ted.max(32)) * (inp.embed_dim as u64 * 4) * 2;
    let flops = base.flops_forward + time_mlp + seq * inp.embed_dim as u64 * 2;
    let bytes = base.bytes_forward + ted * 4 + (inp.embed_dim as u64) * 16;
    FlopsEstimate {
        flops_forward: flops,
        bytes_forward: bytes,
    }
}

pub fn estimate_trm(inp: EstimateInput, recursions: usize, steps: usize) -> FlopsEstimate {
    let base = estimate_transformer_block(inp);
    let r = recursions as u64;
    let s = steps as u64;
    let flops = base.flops_forward * (r + 1) * s;
    let bytes = base.bytes_forward * (r + 1) * s;
    FlopsEstimate {
        flops_forward: flops,
        bytes_forward: bytes,
    }
}
