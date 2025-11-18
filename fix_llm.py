
import sys

file_path = r"d:\RustGPT\src\llm.rs"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Mutable methods
content = content.replace("lrm.transformer.attention.take_tau_metrics()", "lrm.attention_mut().take_tau_metrics()")
content = content.replace("lrm.transformer.attention.take_pred_norm()", "lrm.attention_mut().take_pred_norm()")
content = content.replace("lrm.transformer.attention.get_head_metrics_and_reset()", "lrm.attention_mut().get_head_metrics_and_reset()")
content = content.replace("lrm.transformer.attention.adapt_degree(", "lrm.attention_mut().adapt_degree(")

# Mutable field access
content = content.replace("&mut lrm.transformer.attention.head_selection_config", "&mut lrm.attention_mut().head_selection_config")

# Immutable methods/fields
content = content.replace("lrm.transformer.attention.head_selection_config", "lrm.attention().head_selection_config")
content = content.replace("lrm.transformer.attention.moh_num_active()", "lrm.attention().moh_num_active()")
content = content.replace("lrm.transformer.attention.compute_moh_penalty(", "lrm.attention().compute_moh_penalty(")
content = content.replace("lrm.transformer.attention.num_heads()", "lrm.attention().num_heads()")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Replacements done.")
