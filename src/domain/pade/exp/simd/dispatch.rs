use ndarray::Array2;

use super::super::PadeExp;

impl PadeExp {
    /// SIMD-accelerated vectorized exponential computation.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn exp_simd(input: &Array2<f64>) -> Array2<f64> {
        if Self::has_avx512() {
            Self::exp_simd_avx512(input)
        } else if Self::has_avx2() {
            Self::exp_simd_avx2(input)
        } else {
            Self::exp_array(input)
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    fn has_avx512() -> bool {
        false
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    fn has_avx2() -> bool {
        cfg!(target_feature = "avx2")
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    fn exp_simd_avx512(input: &Array2<f64>) -> Array2<f64> {
        Self::exp_array(input)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    fn exp_simd_avx2(input: &Array2<f64>) -> Array2<f64> {
        let mut output = Array2::zeros(input.dim());
        const SIMD_CHUNK_SIZE: usize = 256;

        if let (Some(out_slice), Some(in_slice)) = (output.as_slice_mut(), input.as_slice()) {
            if input.len() > SIMD_CHUNK_SIZE {
                use rayon::prelude::*;
                out_slice
                    .par_iter_mut()
                    .zip(in_slice.par_iter())
                    .for_each(|(out, &x)| *out = Self::exp(x));
            } else {
                Self::process_simd_chunks(out_slice, in_slice);
            }
        } else {
            for (out, &x) in output.iter_mut().zip(input.iter()) {
                *out = Self::exp(x);
            }
        }

        output
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    fn process_simd_chunks(out_slice: &mut [f64], in_slice: &[f64]) {
        const SIMD_CHUNK_SIZE: usize = 64;

        out_slice
            .chunks_mut(SIMD_CHUNK_SIZE)
            .zip(in_slice.chunks(SIMD_CHUNK_SIZE))
            .for_each(|(out_chunk, in_chunk)| {
                in_chunk
                    .iter()
                    .zip(out_chunk.iter_mut())
                    .for_each(|(&x, out)| *out = Self::exp(x));
            });
    }

    /// Fallback for non-x86 architectures.
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    #[inline]
    pub fn exp_simd(input: &Array2<f64>) -> Array2<f64> {
        Self::exp_array(input)
    }
}
