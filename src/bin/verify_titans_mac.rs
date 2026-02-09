use llm::domain::attention::poly_attention::PolyAttention;
use llm::domain::memory::titans::mac::TitansMAC;
use llm::domain::memory::titans::NeuralMemory;
use llm::domain::network::Layer;
use ndarray::{Array2, Axis};
use rand::Rng;

fn main() {
    // Configuration
    let input_dim = 16;
    let num_heads = 4;
    let memory_hidden_dim = 16;
    let segment_len = 4;
    let persistent_len = 4;
    let seq_len = 4; // 1 segment

    // Initialize components
    let poly = PolyAttention::new(input_dim, num_heads, 3, 64, None);
    let memory = NeuralMemory::new(input_dim, input_dim, input_dim, memory_hidden_dim);
    let mut mac = TitansMAC::new(poly, memory, persistent_len, segment_len);

    // Create random input
    let mut rng = rand::rng();
    let mut input_data = Vec::with_capacity(seq_len * input_dim);
    for _ in 0..seq_len * input_dim {
        input_data.push(rng.random::<f32>());
    }
    let input = Array2::from_shape_vec((seq_len, input_dim), input_data).unwrap();
    println!("Main Input[0,0]: {}", input[[0,0]]);

    // Clone for streaming BEFORE running batch
    let mut mac_stream = mac.clone();

    // 1. Batch Forward
    println!("Running Batch Forward...");
    let batch_output = mac.forward(&input);

    // Snapshot Batch Memory State
    let batch_mem_w1_sum = mac.memory.curr_memory.as_ref().unwrap().w1.sum();
    let batch_mom_w1_sum = mac.memory.momentum.as_ref().unwrap().w1.sum();
    println!("Batch Memory W1 Sum: {:.6}", batch_mem_w1_sum);
    println!("Batch Momentum W1 Sum: {:.6}", batch_mom_w1_sum);

    // 2. Streaming Forward
    println!("Running Streaming Forward...");
    let mut stream_output = Array2::<f32>::zeros((seq_len, input_dim));
    let mut captured_stream_scores_h0: Option<ndarray::Array1<f32>> = None;

    for i in 0..seq_len {
        let token = input.row(i).to_owned();
        
        let out_token = mac_stream.forward_step(&token);
        stream_output.row_mut(i).assign(&out_token);
        
        if i == seq_len - 1 {
             println!("Stream Step {} (Last) Out: {:?}", i, out_token);
             if let Some(ws) = &mac_stream.streaming_workspace {
                 // Capture scores for Head 0
                 // scores_buffer is [Context | Input]
                 // context_len = persistent_len (4) + memory (1) = 5.
                 // Total len = 5 + 1 = 6.
                 let total_len = persistent_len + 1 + 1;
                 let scores = ws.poly_context_workspace.scores_buffer.slice(ndarray::s![0..total_len]).to_owned();
                 captured_stream_scores_h0 = Some(scores);
             }
        }

        if (i + 1) % segment_len == 0 {
             println!("End of segment {}", (i + 1) / segment_len);
             if let Some(mem) = &mac_stream.memory.curr_memory {
                 println!("  Stream Memory W1 Sum: {:.6}", mem.w1.sum());
             }
             if let Some(mom) = &mac_stream.memory.momentum {
                 println!("  Stream Momentum W1 Sum: {:.6}", mom.w1.sum());
             }
        }
    }

    // 3. Compare Outputs
    let diff = &batch_output - &stream_output;
    
    println!("--- Segment 1 Comparison (0..4) ---");
    let diff_seg1 = diff.slice(ndarray::s![0..4, ..]);
    let max_diff_seg1 = diff_seg1.mapv(|x| x.abs()).iter().fold(0.0f32, |a, &b| a.max(b));
    println!("Max Diff Seg 1: {:.9}", max_diff_seg1);

    let max_diff = max_diff_seg1;
    println!("Max Difference: {:.9}", max_diff);

    // 4. Verify Head 0 Scores
    // Access Batch Data
    let batch_data = mac.cached_forward_data.as_ref().expect("No batch data")[0].clone();
    let scores_batch_h0 = &batch_data.poly_caches[0].scores_dump.as_ref().unwrap()[0];
    
    let scores_stream_h0 = captured_stream_scores_h0.as_ref().expect("No stream scores captured");

    println!("\n--- Head 0 Debug ---");
    println!("Batch Scores: {:?}", scores_batch_h0);
    println!("Stream Scores: {:?}", scores_stream_h0);
    println!("Diff: {:.4e}", (scores_batch_h0 - scores_stream_h0).abs().sum());

    // 4. Manual Q/K Projection Check (for Head 0)
    // PolyAttention calculates head_dim = embed_dim / num_heads = 16 / 4 = 4.
    let head_dim = 4; 
    let head_idx = 0;
    let start = head_idx * head_dim;
    let end = start + head_dim;

    println!("\n=== Manual Projection Check (Head {}) ===", head_idx);
    
    // Q Projection
    // Batch Q comes from w_q * input. 
    // We want the Q for the last input token.
    let input_last = input.row(seq_len - 1);
    let q_proj = mac.core.w_q.t().dot(&input_last); // (16)
    let q_h0 = q_proj.slice(ndarray::s![start..end]);
    println!("Manual Q_h0: {:?}", q_h0);

    // K Projection (Input)
    let k_proj_in = mac.core.w_k.t().dot(&input_last); // (16)
    let k_in_h0 = k_proj_in.slice(ndarray::s![start..end]);
    println!("Manual K_in_h0: {:?}", k_in_h0);

    // K Projection (Last Context - Memory)
    // The last item in context is Neural Memory (h_t for the last segment).
    // In Batch, this is input_seq[persistent_len] (index 4).
    let _memory_row = mac.persistent_memory.row(0); // This is persistent[0]. 
    // Wait, context structure in Batch path for the last token:
    // [Persistent (4) | Memory (1) | Input (1)]
    // Indices: 0..4 (Persistent), 4 (Memory), 5 (Input)
    
    // Let's check Persistent[0]
    let k_proj_p0 = mac.core.w_k.t().dot(&mac.persistent_memory.row(0));
    let k_p0_h0 = k_proj_p0.slice(ndarray::s![start..end]);
    println!("Manual K_p0_h0: {:?}", k_p0_h0);

    // Score Calculation (Head 0, Input attending to Input)
    let dk_scale = 1.0 / (head_dim as f32).sqrt();
    let score_in_raw = q_h0.dot(&k_in_h0) * dk_scale;
    println!("Manual Score In Raw (Q*K*scale): {}", score_in_raw);

    // CoPE for Input (Pos 0)
    // Input is at relative pos 0.
    // PolyAttention uses mac.core.cope.pos_embeddings
    // We need to access private field `cope` via `core`.
    // Actually, we can't access private fields easily if they are not pub.
    // But we can check if `Stream Scores In` matches `score_in_raw + CoPE`.
    
    println!("=== End Manual Check ===\n");

    assert!((scores_batch_h0 - scores_stream_h0).abs().sum() < 1e-4, "Head 0 Scores Divergence");

    if max_diff < 1e-4 {
        println!("SUCCESS: Batch and Streaming outputs match!");
    } else {
        println!("FAILURE: Significant divergence detected.");
    }
}
