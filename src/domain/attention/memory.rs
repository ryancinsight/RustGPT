use std::cell::RefCell;

use ndarray::Array2;

thread_local! {
    #[allow(clippy::missing_const_for_thread_local)]
    static TLS_SCORES: RefCell<Option<Array2<f32>>> = const { RefCell::new(None) }; // (N, N)
    #[allow(clippy::missing_const_for_thread_local)]
    static TLS_WORK:   RefCell<Option<Array2<f32>>> = const { RefCell::new(None) }; // (N, N)
    #[allow(clippy::missing_const_for_thread_local)]
    static TLS_YH:     RefCell<Option<Array2<f32>>> = const { RefCell::new(None) }; // (N, d_h)
    #[allow(clippy::missing_const_for_thread_local)]
    static TLS_PHI:    RefCell<Option<ndarray::Array1<f32>>> = const { RefCell::new(None) }; // (w)
    #[allow(clippy::missing_const_for_thread_local)]
    static TLS_ACC_F64: RefCell<Option<Vec<f64>>> = const { RefCell::new(None) }; // (d_h)
    #[allow(clippy::missing_const_for_thread_local)]
    static TLS_QPE: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Get or create a thread-local scratch buffer for attention scores (N×N matrices)
#[inline]
pub fn with_tls_scores<R>(n: usize, f: impl FnOnce(&mut Array2<f32>) -> R) -> R {
    TLS_SCORES.with(|cell| {
        let mut opt = cell.borrow_mut();
        let need = match &*opt {
            Some(a) => a.shape() != [n, n],
            None => true,
        };
        if need {
            *opt = Some(Array2::<f32>::zeros((n, n)));
        }
        let mat = opt.as_mut().unwrap();
        f(mat)
    })
}

/// Get or create a thread-local scratch buffer for intermediate work matrices (N×N)
#[inline]
pub fn with_tls_work<R>(n: usize, f: impl FnOnce(&mut Array2<f32>) -> R) -> R {
    TLS_WORK.with(|cell| {
        let mut opt = cell.borrow_mut();
        let need = match &*opt {
            Some(a) => a.shape() != [n, n],
            None => true,
        };
        if need {
            *opt = Some(Array2::<f32>::zeros((n, n)));
        }
        let mat = opt.as_mut().unwrap();
        f(mat)
    })
}

/// Get or create a thread-local scratch buffer for head outputs (N×d_h matrices)
#[inline]
pub fn with_tls_yh<R>(n: usize, d: usize, f: impl FnOnce(&mut Array2<f32>) -> R) -> R {
    TLS_YH.with(|cell| {
        let mut opt = cell.borrow_mut();
        let need = match &*opt {
            Some(a) => a.shape() != [n, d],
            None => true,
        };
        if need {
            *opt = Some(Array2::<f32>::zeros((n, d)));
        }
        let mat = opt.as_mut().unwrap();
        f(mat)
    })
}

#[inline]
pub fn with_tls_phi<R>(len: usize, f: impl FnOnce(&mut ndarray::Array1<f32>) -> R) -> R {
    TLS_PHI.with(|cell| {
        let mut opt = cell.borrow_mut();
        let need = match &*opt {
            Some(a) => a.len() != len,
            None => true,
        };
        if need {
            *opt = Some(ndarray::Array1::<f32>::zeros(len));
        }
        let vec = opt.as_mut().unwrap();
        f(vec)
    })
}

#[inline]
pub fn with_tls_acc_f64<R>(len: usize, f: impl FnOnce(&mut [f64]) -> R) -> R {
    TLS_ACC_F64.with(|cell| {
        let mut opt = cell.borrow_mut();
        let need = match &*opt {
            Some(v) => v.len() != len,
            None => true,
        };
        if need {
            *opt = Some(vec![0.0f64; len]);
        }
        let buf = opt.as_mut().unwrap();
        f(buf.as_mut_slice())
    })
}

#[inline]
pub fn with_tls_qpe<R>(len: usize, f: impl FnOnce(&mut Vec<f32>) -> R) -> R {
    TLS_QPE.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() != len {
            buf.resize(len, 0.0);
        }
        f(&mut buf)
    })
}
