use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum CoPEVariant {
    Standard,
    Gated,
    Factorized { rank: usize },
    Hierarchical { num_chunks: usize },
    Optimized { rank: usize },
    Path,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CoPEConfig {
    pub variant: CoPEVariant,
    pub max_pos: usize,
    pub window_size: Option<usize>,
}

impl Default for CoPEConfig {
    fn default() -> Self {
        Self {
            variant: CoPEVariant::Standard,
            max_pos: 2048,
            window_size: None,
        }
    }
}
