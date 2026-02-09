use llm::presentation::webui::{run_server, WebUiConfig};
use llm::infrastructure::persistence::FileModelStorage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing (logs to stdout; respects RUST_LOG)
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    // Default config (127.0.0.1:8080, serves embedded static index)
    let config = WebUiConfig::default();

    // File-based storage (models/ directory)
    let storage = FileModelStorage::new("models");

    println!("Starting RustGPT Web UI at http://{}:{}", config.host, config.port);

    run_server(config, storage).await?;

    Ok(())
}
