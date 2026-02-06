use ndarray::Array2;

use super::super::PadeExp;

impl PadeExp {
    /// Vectorized exponential computation for ndarray arrays.
    #[inline]
    pub fn exp_array(input: &Array2<f64>) -> Array2<f64> {
        let mut output = Array2::zeros(input.dim());

        if let (Some(out_slice), Some(in_slice)) = (output.as_slice_mut(), input.as_slice()) {
            if input.len() > 2048 {
                use rayon::prelude::*;
                out_slice
                    .par_iter_mut()
                    .zip(in_slice.par_iter())
                    .for_each(|(out, &x)| *out = Self::exp(x));
            } else {
                Self::process_chunks_iterator(out_slice, in_slice);
            }
        } else {
            for (out, &x) in output.iter_mut().zip(input.iter()) {
                *out = Self::exp(x);
            }
        }

        output
    }

    /// Lazy iterator-based exponential computation (zero-allocation for caller).
    #[inline]
    pub fn exp_iter<'a, I>(iter: I) -> impl Iterator<Item = f64> + 'a
    where
        I: Iterator<Item = f64> + 'a,
    {
        iter.map(Self::exp)
    }

    /// Zero-copy in-place exponential transformation.
    #[inline]
    pub fn exp_array_inplace(array: &mut Array2<f64>) {
        let len = array.len();
        if let Some(slice) = array.as_slice_mut() {
            if len > 2048 {
                use rayon::prelude::*;
                slice.par_iter_mut().for_each(|x| *x = Self::exp(*x));
            } else {
                Self::process_chunks_iterator_inplace(slice);
            }
        } else {
            for x in array.iter_mut() {
                *x = Self::exp(*x);
            }
        }
    }

    #[inline]
    fn process_chunks_iterator(out_slice: &mut [f64], in_slice: &[f64]) {
        const CHUNK_SIZE: usize = 64;

        out_slice
            .chunks_mut(CHUNK_SIZE)
            .zip(in_slice.chunks(CHUNK_SIZE))
            .for_each(|(out_chunk, in_chunk)| {
                in_chunk
                    .iter()
                    .zip(out_chunk.iter_mut())
                    .for_each(|(&x, out)| *out = Self::exp(x));
            });
    }

    #[inline]
    fn process_chunks_iterator_inplace(out_slice: &mut [f64]) {
        const CHUNK_SIZE: usize = 64;

        out_slice.chunks_mut(CHUNK_SIZE).for_each(|chunk| {
            chunk.iter_mut().for_each(|x| *x = Self::exp(*x));
        });
    }
}
