pub fn bleu_1_2(reference: &[usize], candidate: &[usize]) -> (f32, f32) {
    let b1 = bleu_n(reference, candidate, 1);
    let b2 = bleu_n(reference, candidate, 2);
    (b1, b2)
}

fn bleu_n(reference: &[usize], candidate: &[usize], n: usize) -> f32 {
    if reference.is_empty() || candidate.is_empty() || n == 0 {
        return 0.0;
    }
    let ref_ngrams = ngrams(reference, n);
    let cand_ngrams = ngrams(candidate, n);
    let mut matches = 0usize;
    let total = cand_ngrams.len();
    use std::collections::HashMap;
    let mut ref_counts: HashMap<Vec<usize>, usize> = HashMap::new();
    for g in ref_ngrams {
        *ref_counts.entry(g).or_insert(0) += 1;
    }
    let mut cand_counts: HashMap<Vec<usize>, usize> = HashMap::new();
    for g in cand_ngrams {
        *cand_counts.entry(g).or_insert(0) += 1;
    }
    for (g, c_cnt) in cand_counts.iter() {
        let r_cnt = *ref_counts.get(g).unwrap_or(&0);
        matches += c_cnt.min(&r_cnt);
    }
    if total == 0 {
        0.0
    } else {
        matches as f32 / total as f32
    }
}

fn ngrams(tokens: &[usize], n: usize) -> Vec<Vec<usize>> {
    let len = tokens.len();
    if len < n {
        return Vec::new();
    }
    let mut res = Vec::with_capacity(len - n + 1);
    for i in 0..=(len - n) {
        res.push(tokens[i..i + n].to_vec());
    }
    res
}

pub fn corpus_bleu_1_2(references: &[Vec<usize>], candidates: &[Vec<usize>]) -> (f32, f32) {
    let mut b1_sum = 0.0f32;
    let mut b2_sum = 0.0f32;
    let count = references.len().min(candidates.len());
    if count == 0 {
        return (0.0, 0.0);
    }
    for i in 0..count {
        let (b1, b2) = bleu_1_2(&references[i], &candidates[i]);
        b1_sum += b1;
        b2_sum += b2;
    }
    (b1_sum / count as f32, b2_sum / count as f32)
}
