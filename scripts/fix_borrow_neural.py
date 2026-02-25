"""Fix the E0499 double borrow in neural.rs forward_gpu_kernel"""

import re

target = 'd:/RustGPT/src/domain/memory/titans/neural.rs'
content = open(target, 'r', encoding='utf-8').read()

# Find and replace the broken download block (lines 1254-1271)
old_block = '''            let mut q_t_buf = pool.allocate(key_dim * 4)?;
            {
                let mut q_t_cpu = vec![0.0f32; key_dim];
                ops.download(pool, &q_buf, &mut {
                    let mut tmp = vec![0.0f32; seq_len * key_dim];
                    ops.download(pool, &q_buf, &mut tmp)?;
                    q_t_cpu.copy_from_slice(&tmp[tok_off_k..tok_off_k + key_dim]);
                    tmp
                })?;
                // Re-upload just the token slice
                let mut q_t_cpu = vec![0.0f32; key_dim];
                {
                    let mut tmp = vec![0.0f32; seq_len * key_dim];
                    ops.download(pool, &q_buf, &mut tmp)?;
                    q_t_cpu.copy_from_slice(&tmp[tok_off_k..tok_off_k + key_dim]);
                }
                ops.upload(pool, &q_t_cpu, &mut q_t_buf)?;
            }'''

new_block = '''            let mut q_t_buf = pool.allocate(key_dim * 4)?;
            {
                let mut q_all_cpu = vec![0.0f32; seq_len * key_dim];
                ops.download(pool, &q_buf, &mut q_all_cpu)?;
                let q_slice = &q_all_cpu[tok_off_k..tok_off_k + key_dim];
                ops.upload(pool, q_slice, &mut q_t_buf)?;
            }'''

if old_block not in content:
    # Try normalizing CRLF
    content_lf = content.replace('\r\n', '\n')
    old_block_lf = old_block.replace('\r\n', '\n')
    if old_block_lf not in content_lf:
        print('ERROR: old block not found')
        # Try searching for substring
        lines = content_lf.split('\n')
        for i, ln in enumerate(lines):
            if 'ops.download(pool, &q_buf, &mut {' in ln:
                print(f'Found at line {i+1}: {repr(ln)}')
        exit(1)
    new_content = content_lf.replace(old_block_lf, new_block, 1)
    open(target, 'w', encoding='utf-8').write(new_content)
else:
    new_content = content.replace(old_block, new_block, 1)
    open(target, 'w', encoding='utf-8').write(new_content)

print('Fixed q_t extraction block.')
