#![feature(portable_simd)]
#![feature(stdarch_x86_avx512)]

mod jacobi_conjugation;
mod givens_qr_factorization;
mod svd;
mod utils;
pub mod traits;
pub use svd::{svd};
pub use utils::{svd_mat};

// Tests
#[cfg(test)]
mod tests;