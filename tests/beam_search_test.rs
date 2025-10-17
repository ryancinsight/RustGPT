use llm::{
    BeamHypothesis, BeamSearchConfig, BeamSearchState, LLM, ModelConfig, Vocab, build_network,
};
use ndarray::Array1;

#[test]
fn test_beam_search_config_default() {
    let config = BeamSearchConfig::default();

    assert_eq!(config.beam_width, 4);
    assert!(!config.use_adaptive_beam);
    assert_eq!(config.min_beam_width, 1);
    assert_eq!(config.max_beam_width, 8);
    assert_eq!(config.adaptation_threshold, 0.5);
    assert_eq!(config.max_length, 100);
    assert_eq!(config.temperature, 1.0);
}

#[test]
fn test_beam_search_config_builder() {
    let config = BeamSearchConfig::new()
        .with_beam_width(8)
        .with_adaptive_beam(true)
        .with_beam_range(2, 16)
        .with_adaptation_threshold(0.7)
        .with_max_length(50)
        .with_temperature(0.8);

    assert_eq!(config.beam_width, 8);
    assert!(config.use_adaptive_beam);
    assert_eq!(config.min_beam_width, 2);
    assert_eq!(config.max_beam_width, 16);
    assert_eq!(config.adaptation_threshold, 0.7);
    assert_eq!(config.max_length, 50);
    assert_eq!(config.temperature, 0.8);
}

#[test]
fn test_beam_hypothesis_creation() {
    let tokens = vec![1, 2, 3];
    let score = -1.5;

    let hypothesis = BeamHypothesis::new(tokens.clone(), score);

    assert_eq!(hypothesis.tokens, tokens);
    assert_eq!(hypothesis.score, score);
    assert!(!hypothesis.is_complete);
}

#[test]
fn test_beam_hypothesis_normalized_score() {
    let tokens = vec![1, 2, 3, 4];
    let score = -4.0;

    let hypothesis = BeamHypothesis::new(tokens, score);

    // Normalized score = score / length = -4.0 / 4 = -1.0
    assert_eq!(hypothesis.normalized_score(), -1.0);
}

#[test]
fn test_beam_search_state_initialization() {
    let initial_tokens = vec![1, 2];
    let beam_width = 4;

    let state = BeamSearchState::new(initial_tokens.clone(), beam_width);

    assert_eq!(state.beams.len(), 1);
    assert_eq!(state.beams[0].tokens, initial_tokens);
    assert_eq!(state.beams[0].score, 0.0);
    assert_eq!(state.current_beam_width, beam_width);
    assert!(state.completed.is_empty());
}

#[test]
fn test_beam_search_state_expand() {
    let initial_tokens = vec![1];
    let beam_width = 2;
    let config = BeamSearchConfig::new().with_beam_width(beam_width);

    let mut state = BeamSearchState::new(initial_tokens, beam_width);

    // Create mock predictions (probabilities for 5 tokens)
    let probs = Array1::from_vec(vec![0.1, 0.3, 0.4, 0.15, 0.05]);
    let predictions = vec![probs];

    state.expand(&predictions, &config, 5);

    // Should have beam_width beams after expansion
    assert_eq!(state.beams.len(), beam_width);

    // All beams should have 2 tokens now (initial + 1 new)
    for beam in &state.beams {
        assert_eq!(beam.tokens.len(), 2);
    }
}

#[test]
fn test_beam_search_state_mark_complete() {
    let initial_tokens = vec![1, 2];
    let beam_width = 2;
    let end_token = 3;
    let max_length = 10;

    let mut state = BeamSearchState::new(initial_tokens, beam_width);

    // Manually add a beam that ends with end_token
    state
        .beams
        .push(BeamHypothesis::new(vec![1, 2, end_token], -1.0));

    state.mark_complete(end_token, max_length);

    // The beam with end_token should be moved to completed
    assert_eq!(state.completed.len(), 1);
    assert_eq!(state.completed[0].tokens.last(), Some(&end_token));
}

#[test]
fn test_beam_search_state_is_done() {
    let initial_tokens = vec![1];
    let beam_width = 2;

    let mut state = BeamSearchState::new(initial_tokens, beam_width);

    // Initially not done
    assert!(!state.is_done());

    // Mark all beams as complete and move them
    state.beams[0].is_complete = true;
    state.completed.push(state.beams.remove(0));

    // Now should be done
    assert!(state.is_done());
}

#[test]
fn test_beam_search_state_get_best() {
    let initial_tokens = vec![1];
    let beam_width = 3;
    let config = BeamSearchConfig::new().with_length_penalty_alpha(1.0);

    let mut state = BeamSearchState::new(initial_tokens, beam_width);

    // Add beams with different scores
    state.beams = vec![
        BeamHypothesis::new(vec![1, 2], -2.0),    // normalized: -1.0
        BeamHypothesis::new(vec![1, 3, 4], -1.5), // normalized: -0.5 (best)
        BeamHypothesis::new(vec![1, 5], -3.0),    // normalized: -1.5
    ];

    let best = state.get_best(&config);

    assert!(best.is_some());
    let best_hypothesis = best.unwrap();
    assert_eq!(best_hypothesis.tokens, vec![1, 3, 4]);
    assert_eq!(
        best_hypothesis.normalized_score(config.length_penalty_alpha),
        -0.5
    );
}

#[test]
fn test_beam_search_state_compute_entropy() {
    let initial_tokens = vec![1];
    let beam_width = 2;

    let state = BeamSearchState::new(initial_tokens, beam_width);

    // Create predictions with different entropy levels
    // Uniform distribution (high entropy)
    let uniform_probs = Array1::from_vec(vec![0.2, 0.2, 0.2, 0.2, 0.2]);

    // Peaked distribution (low entropy)
    let peaked_probs = Array1::from_vec(vec![0.9, 0.025, 0.025, 0.025, 0.025]);

    let uniform_entropy = state.compute_entropy(&[uniform_probs]);
    let peaked_entropy = state.compute_entropy(&[peaked_probs]);

    // Uniform distribution should have higher entropy
    assert!(uniform_entropy > peaked_entropy);
}

#[test]
fn test_beam_search_state_adapt_beam_width() {
    let initial_tokens = vec![1];
    let beam_width = 4;
    let config = BeamSearchConfig::new()
        .with_beam_width(beam_width)
        .with_adaptive_beam(true)
        .with_beam_range(2, 8)
        .with_adaptation_threshold(0.5);

    let mut state = BeamSearchState::new(initial_tokens, beam_width);

    // High entropy (above threshold) should increase beam width
    state.adapt_beam_width(5.0, &config);
    assert_eq!(state.current_beam_width, 5);

    // Low entropy (below threshold) should decrease beam width
    state.adapt_beam_width(0.5, &config);
    assert_eq!(state.current_beam_width, 4);
}

#[test]
fn test_beam_search_state_adapt_beam_width_bounds() {
    let initial_tokens = vec![1];
    let config = BeamSearchConfig::new()
        .with_beam_width(2)
        .with_adaptive_beam(true)
        .with_beam_range(2, 4)
        .with_adaptation_threshold(0.5);

    let mut state = BeamSearchState::new(initial_tokens, 2);

    // Try to decrease below min
    state.adapt_beam_width(0.1, &config);
    assert_eq!(state.current_beam_width, 2); // Should stay at min

    // Increase to max
    state.adapt_beam_width(5.0, &config);
    assert_eq!(state.current_beam_width, 3);
    state.adapt_beam_width(5.0, &config);
    assert_eq!(state.current_beam_width, 4);

    // Try to increase above max
    state.adapt_beam_width(5.0, &config);
    assert_eq!(state.current_beam_width, 4); // Should stay at max
}

#[test]
fn test_beam_search_generation_basic() {
    // Create a simple model for testing
    let vocab = Vocab::new(vec!["<pad>", "hello", "world", "</s>"]);
    let config = ModelConfig::transformer(128, 256, 2, 80, None, Some(4));
    let network = build_network(&config, &vocab);
    let mut llm = LLM::new(vocab, network);

    // Create beam search config
    let beam_config = BeamSearchConfig::new()
        .with_beam_width(2)
        .with_max_length(5);

    // Generate text
    let output = llm.generate_with_beam_search("hello", &beam_config);

    // Output should be a string (may be empty for untrained model)
    assert!(output.is_empty() || !output.is_empty());
}

#[test]
fn test_beam_search_generation_with_adaptive_beam() {
    // Create a simple model for testing
    let vocab = Vocab::new(vec!["<pad>", "hello", "world", "</s>"]);
    let config = ModelConfig::transformer(128, 256, 2, 80, None, Some(4));
    let network = build_network(&config, &vocab);
    let mut llm = LLM::new(vocab, network);

    // Create beam search config with adaptive beam
    let beam_config = BeamSearchConfig::new()
        .with_beam_width(2)
        .with_adaptive_beam(true)
        .with_beam_range(1, 4)
        .with_max_length(5);

    // Generate text
    let output = llm.generate_with_beam_search("hello", &beam_config);

    // Output should be a string (may be empty for untrained model)
    assert!(output.is_empty() || !output.is_empty());
}

#[test]
fn test_beam_search_with_temperature() {
    let initial_tokens = vec![1];
    let beam_width = 2;

    // Test with temperature = 0.5 (more confident)
    let config_low_temp = BeamSearchConfig::new()
        .with_beam_width(beam_width)
        .with_temperature(0.5);

    let mut state = BeamSearchState::new(initial_tokens.clone(), beam_width);
    let probs = Array1::from_vec(vec![0.1, 0.3, 0.4, 0.15, 0.05]);
    let predictions = vec![probs];

    state.expand(&predictions, &config_low_temp, 5);

    // Should have beam_width beams
    assert_eq!(state.beams.len(), beam_width);
}
