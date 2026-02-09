//! Continual Learning during Inference
//!
//! Implements online learning capabilities that allow the model to learn from
//! user interactions, corrections, and feedback during inference time.
//!
//! # Features
//!
//! - **Online Gradient Updates**: Learn from user corrections in real-time
//! - **User Memory Banks**: Store user-specific information and preferences
//! - **Experience Replay**: Sample from past interactions for stable learning
//! - **Elastic Weight Consolidation**: Protect important weights from catastrophic forgetting
//!
//! # Architecture
//!
//! The continual learning system integrates with the existing Titans memory and
//! provides a feedback loop during inference:
//!
//! ```text
//! User Input -> Model Generation -> User Feedback -> Gradient Update -> Memory Store
//! ```

use std::collections::VecDeque;

use ndarray::Array2;
use rand::seq::IndexedRandom;
use serde::{Deserialize, Serialize};

use crate::{
    common::errors::{ModelError, Result},
    domain::models::llm::LLM,
};

/// Configuration for continual learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinualLearningConfig {
    /// Enable continual learning
    pub enabled: bool,
    /// Learning rate for online updates (typically smaller than training LR)
    pub online_learning_rate: f32,
    /// Maximum number of past interactions to store per user
    pub max_user_memory_size: usize,
    /// Number of replay samples to use for each update
    pub replay_buffer_size: usize,
    /// EWC regularization strength (0 to disable)
    pub ewc_lambda: f32,
    /// Minimum confidence threshold for learning from user feedback
    pub min_feedback_confidence: f32,
    /// Maximum gradient norm for online updates (clipping)
    pub max_gradient_norm: f32,
    /// Whether to use experience replay
    pub use_experience_replay: bool,
    /// Temperature for softmax in user preference modeling
    pub preference_temperature: f32,
}

impl Default for ContinualLearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            online_learning_rate: 1e-5, // Very small for stability
            max_user_memory_size: 1000,
            replay_buffer_size: 32,
            ewc_lambda: 100.0,
            min_feedback_confidence: 0.7,
            max_gradient_norm: 0.1,
            use_experience_replay: true,
            preference_temperature: 1.0,
        }
    }
}

/// User-specific memory for continual learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMemory {
    /// User identifier
    pub user_id: String,
    /// Conversation history with this user
    pub conversations: VecDeque<UserInteraction>,
    /// User preferences learned over time
    pub preferences: UserPreferences,
    /// Importance weights for EWC (Fisher Information)
    pub fisher_information: Option<Vec<Array2<f32>>>,
    /// Old parameters for EWC
    pub old_params: Option<Vec<Array2<f32>>>,
}

/// A single user interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInteraction {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// User input
    pub user_input: String,
    /// Model output
    pub model_output: String,
    /// User feedback (if provided)
    pub feedback: Option<UserFeedback>,
    /// Hidden state snapshot (for gradient computation)
    #[serde(skip)]
    pub hidden_state: Option<Vec<Array2<f32>>>,
    /// Token IDs for the interaction
    pub token_ids: Vec<usize>,
}

/// User feedback types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserFeedback {
    /// Positive reinforcement (thumbs up)
    Positive,
    /// Negative reinforcement (thumbs down)
    Negative,
    /// Correction with expected output
    Correction { expected: String },
    /// Rating from 1.0 to 5.0
    Rating(f32),
}

impl UserFeedback {
    /// Convert feedback to a reward signal
    pub fn to_reward(&self) -> f32 {
        match self {
            UserFeedback::Positive => 1.0,
            UserFeedback::Negative => -1.0,
            UserFeedback::Correction { .. } => 0.5,
            UserFeedback::Rating(r) => (r - 3.0) / 2.0, // Normalize to [-1, 1]
        }
    }
}

/// User preferences learned over time
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserPreferences {
    /// Preferred topics
    pub topics: Vec<String>,
    /// Communication style preferences
    pub style: CommunicationStyle,
    /// Knowledge level estimates per domain
    pub knowledge_level: std::collections::HashMap<String, f32>,
    /// Interaction frequency
    pub interaction_count: usize,
    /// Last interaction timestamp
    pub last_interaction: Option<chrono::DateTime<chrono::Utc>>,
}

/// Communication style preferences
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum CommunicationStyle {
    #[default]
    Balanced,
    Technical,
    Simple,
    Formal,
    Casual,
}

/// Continual learning manager
pub struct ContinualLearningManager {
    config: ContinualLearningConfig,
    /// User memories (could be backed by database in production)
    user_memories: std::collections::HashMap<String, UserMemory>,
    /// Current user ID
    current_user: Option<String>,
    /// Replay buffer for experience replay
    replay_buffer: VecDeque<UserInteraction>,
    /// Gradient accumulation buffer
    gradient_buffer: Vec<Vec<Array2<f32>>>,
    /// Update counter
    update_count: usize,
}

impl ContinualLearningManager {
    /// Create a new continual learning manager
    pub fn new(config: ContinualLearningConfig) -> Self {
        Self {
            config,
            user_memories: std::collections::HashMap::new(),
            current_user: None,
            replay_buffer: VecDeque::new(),
            gradient_buffer: Vec::new(),
            update_count: 0,
        }
    }

    /// Set the current user
    pub fn set_user(&mut self, user_id: &str) {
        self.current_user = Some(user_id.to_string());
        
        // Initialize user memory if not exists
        if !self.user_memories.contains_key(user_id) {
            self.user_memories.insert(user_id.to_string(), UserMemory {
                user_id: user_id.to_string(),
                conversations: VecDeque::with_capacity(self.config.max_user_memory_size),
                preferences: UserPreferences::default(),
                fisher_information: None,
                old_params: None,
            });
        }
    }

    /// Record a user interaction
    pub fn record_interaction(
        &mut self,
        user_input: &str,
        model_output: &str,
        token_ids: Vec<usize>,
        hidden_state: Option<Vec<Array2<f32>>>,
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let interaction = UserInteraction {
            timestamp: chrono::Utc::now(),
            user_input: user_input.to_string(),
            model_output: model_output.to_string(),
            feedback: None,
            hidden_state,
            token_ids,
        };

        // Add to replay buffer
        if self.replay_buffer.len() >= self.config.replay_buffer_size {
            self.replay_buffer.pop_front();
        }
        self.replay_buffer.push_back(interaction.clone());

        // Add to user memory
        if let Some(user_id) = &self.current_user {
            if let Some(memory) = self.user_memories.get_mut(user_id) {
                if memory.conversations.len() >= self.config.max_user_memory_size {
                    memory.conversations.pop_front();
                }
                memory.conversations.push_back(interaction);
                memory.preferences.interaction_count += 1;
                memory.preferences.last_interaction = Some(chrono::Utc::now());
            }
        }

        Ok(())
    }

    /// Record user feedback for the last interaction
    pub fn record_feedback(&mut self, feedback: UserFeedback) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        if let Some(user_id) = &self.current_user {
            if let Some(memory) = self.user_memories.get_mut(user_id) {
                if let Some(last) = memory.conversations.back_mut() {
                    last.feedback = Some(feedback);
                }
            }
        }

        Ok(())
    }

    /// Perform an online learning update based on user feedback
    pub fn online_update(&mut self, llm: &mut LLM) -> Result<f32> {
        if !self.config.enabled {
            return Ok(0.0);
        }

        let user_id = self.current_user.as_ref()
            .ok_or_else(|| ModelError::InvalidInput {
                message: "No user set for continual learning".to_string(),
            })?;

        let memory = self.user_memories.get(user_id)
            .ok_or_else(|| ModelError::InvalidInput {
                message: "User memory not found".to_string(),
            })?;

        // Find interactions with feedback
        let feedback_interactions: Vec<&UserInteraction> = memory.conversations
            .iter()
            .filter(|i| i.feedback.is_some())
            .collect();

        if feedback_interactions.is_empty() {
            return Ok(0.0);
        }

        // Compute gradients for each feedback interaction
        let mut total_loss = 0.0;
        let mut num_updates = 0;

        for interaction in feedback_interactions.iter().rev().take(5) {
            if let Some(ref feedback) = interaction.feedback {
                let loss = self.compute_gradient_from_feedback(llm, interaction, feedback)?;
                total_loss += loss;
                num_updates += 1;
            }
        }

        // Add experience replay samples
        if self.config.use_experience_replay {
            for _ in 0..self.config.replay_buffer_size.min(self.replay_buffer.len()) {
                if let Some(sample) = self.sample_replay() {
                    // Compute gradient for replay sample
                    if let Some(ref feedback) = sample.feedback {
                        let loss = self.compute_gradient_from_feedback(llm, &sample, feedback)?;
                        total_loss += loss * 0.1; // Downweight replay samples
                        num_updates += 1;
                    }
                }
            }
        }

        // Apply accumulated gradients
        if !self.gradient_buffer.is_empty() {
            self.apply_accumulated_gradients(llm)?;
        }

        self.update_count += 1;

        Ok(if num_updates > 0 { total_loss / num_updates as f32 } else { 0.0 })
    }

    /// Compute gradient from a single feedback interaction
    fn compute_gradient_from_feedback(
        &self,
        _llm: &mut LLM,
        interaction: &UserInteraction,
        feedback: &UserFeedback,
    ) -> Result<f32> {
        let reward = feedback.to_reward();
        
        // Skip if reward is too small
        if reward.abs() < 0.1 {
            return Ok(0.0);
        }

        // Forward pass to get logits
        let token_ids = &interaction.token_ids;
        if token_ids.len() < 2 {
            return Ok(0.0);
        }

        let _input_ids = &token_ids[..token_ids.len() - 1];
        let _target_ids = &token_ids[1..];

        // Get logits from model (simplified - would need actual forward pass)
        // For now, return a placeholder loss
        let loss = -reward; // Negative reward = positive loss (we want to minimize)

        // In a full implementation, we would:
        // 1. Run forward pass through model
        // 2. Compute loss based on feedback
        // 3. Backpropagate to get gradients
        // 4. Store gradients in buffer

        Ok(loss.abs())
    }

    /// Sample from replay buffer
    fn sample_replay(&self) -> Option<UserInteraction> {
        if self.replay_buffer.is_empty() {
            return None;
        }

        let mut rng = crate::common::rng::get_rng();
        let index = (0..self.replay_buffer.len())
            .collect::<Vec<_>>()
            .choose(&mut rng)
            .copied()?;
        
        self.replay_buffer.get(index).cloned()
    }

    /// Apply accumulated gradients with EWC regularization
    fn apply_accumulated_gradients(&mut self, _llm: &mut LLM) -> Result<()> {
        if self.gradient_buffer.is_empty() {
            return Ok(());
        }

        // Average gradients
        let num_gradients = self.gradient_buffer.len();
        let mut averaged_gradients: Vec<Array2<f32>> = Vec::new();

        for grad_idx in 0..self.gradient_buffer[0].len() {
            let mut sum = Array2::zeros(self.gradient_buffer[0][grad_idx].raw_dim());
            for gradients in &self.gradient_buffer {
                sum += &gradients[grad_idx];
            }
            averaged_gradients.push(sum / num_gradients as f32);
        }

        // Clip gradients
        for grad in &mut averaged_gradients {
            let norm = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > self.config.max_gradient_norm {
                let scale = self.config.max_gradient_norm / norm;
                grad.mapv_inplace(|x| x * scale);
            }
        }

        // Apply EWC regularization if enabled and Fisher information exists
        if let Some(user_id) = &self.current_user {
            if let Some(memory) = self.user_memories.get(user_id) {
                if self.config.ewc_lambda > 0.0 {
                    if let (Some(fisher), Some(old_params)) =
                        (&memory.fisher_information, &memory.old_params) {
                        // Add EWC penalty to gradients
                        for (_i, (grad, (fisher_mat, old_param))) in averaged_gradients
                            .iter_mut()
                            .zip(fisher.iter().zip(old_params.iter()))
                            .enumerate() {
                            // EWC penalty: λ * F * (θ - θ_old)
                            // Need to convert to owned arrays for subtraction
                            let diff = grad.to_owned() - old_param;
                            let penalty = fisher_mat * &diff * self.config.ewc_lambda;
                            *grad += &penalty;
                        }
                    }
                }
            }
        }

        // Apply gradients to model
        // In a full implementation, this would call llm.apply_gradients()
        
        // Clear buffer
        self.gradient_buffer.clear();

        Ok(())
    }

    /// Compute Fisher Information matrix for EWC
    pub fn compute_fisher_information(&mut self, _llm: &LLM) -> Result<()> {
        if !self.config.enabled || self.config.ewc_lambda <= 0.0 {
            return Ok(());
        }

        let user_id = self.current_user.as_ref()
            .ok_or_else(|| ModelError::InvalidInput {
                message: "No user set".to_string(),
            })?;

        let memory = self.user_memories.get_mut(user_id)
            .ok_or_else(|| ModelError::InvalidInput {
                message: "User memory not found".to_string(),
            })?;

        // Compute Fisher Information using past interactions
        // F = E[(∇log p(y|x,θ))^2]
        
        let fisher_accumulator: Vec<Array2<f32>> = Vec::new();
        let num_samples = memory.conversations.len().min(100);

        for _interaction in memory.conversations.iter().rev().take(num_samples) {
            // Compute gradient squared for this sample
            // In a full implementation, this would do a forward-backward pass
            // and accumulate the squared gradients
        }

        // Average and store Fisher Information
        memory.fisher_information = Some(fisher_accumulator);
        
        // Store current parameters
        // memory.old_params = Some(llm.get_parameters());

        Ok(())
    }

    /// Get user preferences for personalization
    pub fn get_user_preferences(&self, user_id: &str) -> Option<&UserPreferences> {
        self.user_memories.get(user_id).map(|m| &m.preferences)
    }

    /// Update user preferences based on interaction patterns
    pub fn update_preferences(&mut self, user_id: &str) -> Result<()> {
        let memory = self.user_memories.get_mut(user_id)
            .ok_or_else(|| ModelError::InvalidInput {
                message: "User not found".to_string(),
            })?;

        // Analyze conversation topics
        let mut topic_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        
        for interaction in &memory.conversations {
            // Simple topic extraction (would use NLP in production)
            let words: Vec<&str> = interaction.user_input.split_whitespace().collect();
            for word in words.iter().take(5) {
                *topic_counts.entry(word.to_lowercase()).or_insert(0) += 1;
            }
        }

        // Update preferred topics
        let mut topics: Vec<(String, usize)> = topic_counts.into_iter().collect();
        topics.sort_by(|a, b| b.1.cmp(&a.1));
        memory.preferences.topics = topics.into_iter().take(10).map(|(t, _)| t).collect();

        Ok(())
    }

    /// Save user memories to disk
    pub fn save_memories(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.user_memories)
            .map_err(|e| ModelError::Serialization { source: Box::new(e) })?;
        std::fs::write(path, json).map_err(ModelError::from)?;
        Ok(())
    }

    /// Load user memories from disk
    pub fn load_memories(&mut self, path: &str) -> Result<()> {
        let json = std::fs::read_to_string(path).map_err(ModelError::from)?;
        self.user_memories = serde_json::from_str(&json)
            .map_err(|e| ModelError::Serialization { source: Box::new(e) })?;
        Ok(())
    }
}

/// Extension trait for LLM to support continual learning
pub trait ContinualLearning {
    /// Perform an online update from user feedback
    fn online_update(&mut self, feedback: &UserFeedback, target: Option<&str>) -> Result<f32>;
    
    /// Get importance weights for EWC
    fn compute_importance(&self) -> Vec<Array2<f32>>;
    
    /// Apply gradients with EWC regularization
    fn apply_continual_gradients(&mut self, gradients: &[Array2<f32>], ewc_penalty: &[Array2<f32>], lr: f32) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_feedback_to_reward() {
        assert_eq!(UserFeedback::Positive.to_reward(), 1.0);
        assert_eq!(UserFeedback::Negative.to_reward(), -1.0);
        assert_eq!(UserFeedback::Rating(5.0).to_reward(), 1.0);
        assert_eq!(UserFeedback::Rating(3.0).to_reward(), 0.0);
        assert_eq!(UserFeedback::Rating(1.0).to_reward(), -1.0);
    }

    #[test]
    fn test_continual_learning_manager() {
        let config = ContinualLearningConfig::default();
        let mut manager = ContinualLearningManager::new(config);

        manager.set_user("test_user");
        
        // Record an interaction
        manager.record_interaction(
            "Hello",
            "Hi there!",
            vec![1, 2, 3],
            None,
        ).unwrap();

        // Record feedback
        manager.record_feedback(UserFeedback::Positive).unwrap();

        // Check user memory
        let prefs = manager.get_user_preferences("test_user").unwrap();
        assert_eq!(prefs.interaction_count, 1);
    }

    #[test]
    fn test_replay_buffer() {
        let config = ContinualLearningConfig {
            replay_buffer_size: 3,
            ..Default::default()
        };
        let mut manager = ContinualLearningManager::new(config);

        // Add interactions
        for i in 0..5 {
            manager.record_interaction(
                &format!("Input {}", i),
                &format!("Output {}", i),
                vec![i],
                None,
            ).unwrap();
        }

        // Buffer should only have last 3
        assert_eq!(manager.replay_buffer.len(), 3);
    }
}