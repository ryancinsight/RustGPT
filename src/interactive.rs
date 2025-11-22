use std::io::Write;
use crate::llm::LLM;

/// Run interactive mode for user input and model responses
pub fn run_interactive_mode(llm: &mut LLM) -> crate::Result<()> {
    println!("\n--- Interactive Mode ---");
    println!("Type a prompt and press Enter to generate text.");
    println!("Using speculative beam search (balanced preset: beam_width=4, lookahead=3)");
    println!("Type 'exit' to quit.");

    let mut input = String::new();
    loop {
        // Clear the input string
        input.clear();

        // Prompt for user input
        print!("\nEnter prompt: ");
        std::io::stdout().flush().unwrap();

        // Read user input
        std::io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");

        // Trim whitespace and check for exit command
        let trimmed_input = input.trim();
        if trimmed_input.eq_ignore_ascii_case("exit") {
            println!("Exiting interactive mode.");
            break;
        }

        // Generate prediction based on user input with "User:" prefix
        let formatted_input = format!("User: {}", trimmed_input);
        let prediction = llm.predict(&formatted_input);

        println!("Model output: {}", prediction);
    }

    Ok(())
}
