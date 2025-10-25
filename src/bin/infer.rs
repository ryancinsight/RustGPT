use std::io::Write;

use clap::Parser;
use llm::{LLM, MAX_SEQ_LEN};

#[derive(Parser, Debug)]
#[command(name = "infer")]
#[command(about = "Interactive chat using a saved RustGPT model")] 
struct Args {
    /// Path to the saved model (versioned .json or .bin)
    #[arg(short, long, default_value = "models/rustgpt.bin")]
    model: String,

    /// If provided, generate once for this prompt then exit
    #[arg(short, long)]
    prompt: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse CLI args
    let args = Args::parse();

    // Load model (versioned with integrity and compatibility checks)
    let mut llm = LLM::load_versioned(&args.model)?;
    println!(
        "Loaded model from {} (max seq len: {}).",
        &args.model,
        MAX_SEQ_LEN
    );

    // Single-shot generation if --prompt provided
    if let Some(p) = args.prompt {
        let out = llm.predict(&p);
        println!("Output: {}", out);
        return Ok(());
    }

    // Interactive chat loop
    println!("\n--- Interactive Chat ---");
    println!("Type a prompt and press Enter to generate text.");
    println!("Decoding: greedy | type 'exit' to quit.");

    let mut input = String::new();
    loop {
        input.clear();
        print!("\nYou: ");
        std::io::stdout().flush().unwrap();

        if std::io::stdin().read_line(&mut input).is_err() {
            eprintln!("Failed to read input");
            continue;
        }

        let prompt = input.trim();
        if prompt.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }
        if prompt.is_empty() {
            continue;
        }

        let response = llm.predict(prompt);
        println!("Model: {}", response);
    }

    Ok(())
}