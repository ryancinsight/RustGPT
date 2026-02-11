use llm::domain::attention::poly_attention::PolyAttention;
use llm::domain::attention::position::config::{CoPEConfig, CoPEVariant};
use llm::domain::memory::titans::NeuralMemory;
use llm::domain::memory::titans::mac::TitansMAC;
use llm::domain::network::Layer;
use ndarray::{Array2, Axis};
use rand::Rng;

fn main() {
    // Configuration
    let input_dim = 16;
    let num_heads = 4;
    let memory_hidden_dim = 16;
    let segment_len = 4;
    let persistent_len = 2;
    let seq_len = 4; // 1 segment

    // Initialize components with CoPEConfig
    let cope_config = CoPEConfig {
        variant: CoPEVariant::Standard,
        max_pos: 64,
        window_size: None,
    };
    let poly = PolyAttention::new(input_dim, num_heads, 3, cope_config);
    let memory = NeuralMemory::new(input_dim, input_dim, input_dim, memory_hidden_dim);
    let mut mac = TitansMAC::new(poly, memory, persistent_len, segment_len);

    // Create random input
    let mut rng = rand::rng();
    let mut input_data = Vec::with_capacity(seq_len * input_dim);
    for _ in 0..seq_len * input_dim {
        input_data.push(rng.random::<f32>());
    }
    let input = Array2::from_shape_vec((seq_len, input_dim), input_data).unwrap();
    println!("Main Input[0,0]: {}", input[[0, 0]]);
    println!("Titan Memory configuration loaded");

    // Clone for streaming BEFORE running batch
    let mut mac_stream = mac.clone();

    // --- Memory Retrieval Verification ---
    println!("--- Memory Retrieval Verification (Step 0) ---");
    let h_t_batch_all = mac.memory.retrieve(&input);
    let h_t_batch_0 = h_t_batch_all.row(0);

    let h_t_stream_0 = mac_stream.memory.retrieve_step(&input.row(0).to_owned());

    println!("Batch h_t[0]: {:?}", h_t_batch_0);
    println!("Stream h_t[0]: {:?}", h_t_stream_0);
    let diff_ht = (&h_t_batch_0 - &h_t_stream_0).mapv(|x| x.abs()).sum();
    println!("h_t Diff: {:.6e}", diff_ht);
    if diff_ht > 1e-5 {
        panic!("h_t mismatch at step 0!");
    }
    println!("--- End Memory Verification ---");

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
    let mut captured_stream_scores_last_head: Option<ndarray::Array1<f32>> = None;

    for i in 0..seq_len {
        let token = input.row(i).to_owned();

        let out_token = mac_stream.forward_step(&token);
        stream_output.row_mut(i).assign(&out_token);

        if i == seq_len - 1 {
            println!("Stream Step {} (Last) Out: {:?}", i, out_token);
            if let Some(ws) = &mac_stream.streaming_workspace {
                // Capture scores for the LAST HEAD (since the buffer is overwritten per head)
                // scores_buffer is [Context | Input]
                // context_len = persistent_len (4) + memory (1) = 5.
                // Total len = 5 + 1 = 6.
                let total_len = persistent_len + 1 + 1;
                let scores = ws
                    .poly_context_workspace
                    .scores_buffer
                    .slice(ndarray::s![0..total_len])
                    .to_owned();
                captured_stream_scores_last_head = Some(scores);
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

    println!("--- Token 0 Comparison ---");
    println!("Batch[0]: {:?}", batch_output.row(0));
    println!("Stream[0]: {:?}", stream_output.row(0));
    let diff_0 = diff.row(0).mapv(|x| x.abs()).sum();
    println!("Diff[0] Sum: {:.6e}", diff_0);

    println!("--- Segment 1 Comparison (0..4) ---");
    let diff_seg1 = diff.slice(ndarray::s![0..4, ..]);
    let max_diff_seg1 = diff_seg1
        .mapv(|x| x.abs())
        .iter()
        .fold(0.0f32, |a, &b| a.max(b));
    println!("Max Diff Seg 1: {:.9}", max_diff_seg1);

    let max_diff = max_diff_seg1;
    println!("Max Difference: {:.9}", max_diff);

    // 4. Verify Head 3 (Last Head) Scores
    // Access Batch Data
    let batch_data = mac.cached_forward_data.as_ref().expect("No batch data")[0].clone();
    // Head 3 is index 3
    // We want the scores for the LAST token of the segment, which corresponds to index `segment_len - 1`
    let scores_batch_h3 = &batch_data.poly_caches[segment_len - 1]
        .scores_dump
        .as_ref()
        .unwrap()[3];

    let scores_stream_h3 = captured_stream_scores_last_head
        .as_ref()
        .expect("No stream scores captured");

    println!("\n--- Head 3 (Last Head) Debug ---");
    println!("Batch Scores: {:?}", scores_batch_h3);
    println!("Stream Scores: {:?}", scores_stream_h3);
    println!(
        "Diff: {:.4e}",
        (scores_batch_h3 - scores_stream_h3).abs().sum()
    );

    // 4. Manual Q/K Projection Check (for Head 3)
    // PolyAttention calculates head_dim = embed_dim / num_heads = 16 / 4 = 4.
    let head_dim = 4;
    let head_idx = 3;
    let start = head_idx * head_dim;
    let end = start + head_dim;

    println!("\n=== Manual Projection Check (Head {}) ===", head_idx);

    // Q Projection
    // Batch Q comes from w_q * input.
    // We want the Q for the last input token.
    let input_last = input.row(seq_len - 1);
    let q_proj = mac.core.w_q.t().dot(&input_last); // (16)
    let q_h3 = q_proj.slice(ndarray::s![start..end]);
    println!("Manual Q_h3: {:?}", q_h3);

    // K Projection (Input)
    let k_proj_in = mac.core.w_k.t().dot(&input_last); // (16)
    let k_in_h3 = k_proj_in.slice(ndarray::s![start..end]);
    println!("Manual K_in_h3: {:?}", k_in_h3);

    // K Projection (Last Context - Memory)
    // The last item in context is Neural Memory (h_t for the last segment).
    // In Batch, this is input_seq[persistent_len] (index 4).

    // Let's check Persistent[0]
    let k_proj_p0 = mac.core.w_k.t().dot(&mac.persistent_memory.row(0));
    let k_p0_h3 = k_proj_p0.slice(ndarray::s![start..end]);
    println!("Manual K_p0_h3: {:?}", k_p0_h3);

    // Score Calculation (Head 3, Input attending to Input)
    let dk_scale = 1.0 / (head_dim as f32).sqrt();
    let score_in_raw = q_h3.dot(&k_in_h3) * dk_scale;
    println!("Manual Score In Raw (Q*K*scale): {}", score_in_raw);

    println!("=== End Manual Check ===\n");

    assert!(
        (scores_batch_h3 - scores_stream_h3).abs().sum() < 1e-4,
        "Head 3 Scores Divergence"
    );

    if max_diff < 1e-4 {
        println!("SUCCESS: Batch and Streaming outputs match!");
    } else {
        println!("FAILURE: Significant divergence detected.");
    }
}
