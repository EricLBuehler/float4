//! Packed pair of F4E2M1 values in a single byte.
//!
//! This module provides the [`F4E2M1x2`] type, which stores two [`F4E2M1`] values
//! in a single byte. This matches the layout used by NVIDIA's `__nv_fp4x2_e2m1`
//! and enables correct `DeviceRepr` usage in `cudarc`.

use crate::F4E2M1;

/// Two [`F4E2M1`] values packed into a single byte.
///
/// The lower nibble (bits 3:0) holds the first value and the upper nibble
/// (bits 7:4) holds the second value. This layout matches NVIDIA's
/// `__nv_fp4x2_e2m1` format.
///
/// # Examples
///
/// ```
/// use float4::{F4E2M1, F4E2M1x2};
///
/// let lo = F4E2M1::from_f64(1.5);
/// let hi = F4E2M1::from_f64(-2.0);
/// let packed = F4E2M1x2::new(lo, hi);
///
/// assert_eq!(packed.lo().to_f64(), 1.5);
/// assert_eq!(packed.hi().to_f64(), -2.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct F4E2M1x2(u8);

const _: () = assert!(std::mem::size_of::<F4E2M1x2>() == 1);

impl F4E2M1x2 {
    /// Two positive zeros.
    pub const ZERO: Self = Self(0x00);

    /// Packs two [`F4E2M1`] values into a single byte.
    ///
    /// `lo` occupies bits 3:0 and `hi` occupies bits 7:4.
    #[inline(always)]
    pub const fn new(lo: F4E2M1, hi: F4E2M1) -> Self {
        Self((lo.to_bits() & 0x0F) | ((hi.to_bits() & 0x0F) << 4))
    }

    /// Wraps a raw byte directly.
    #[inline(always)]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Returns the raw byte.
    #[inline(always)]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// Extracts the lower nibble as an [`F4E2M1`].
    #[inline(always)]
    pub const fn lo(self) -> F4E2M1 {
        F4E2M1::from_bits(self.0 & 0x0F)
    }

    /// Extracts the upper nibble as an [`F4E2M1`].
    #[inline(always)]
    pub const fn hi(self) -> F4E2M1 {
        F4E2M1::from_bits((self.0 >> 4) & 0x0F)
    }

    /// Creates a packed pair by converting two `f32` values to [`F4E2M1`].
    ///
    /// Each value is independently rounded to the nearest representable
    /// F4E2M1 value using round-to-nearest-even.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1x2;
    ///
    /// let packed = F4E2M1x2::from_f32_pair(1.5, -2.0);
    /// assert_eq!(packed.lo().to_f64(), 1.5);
    /// assert_eq!(packed.hi().to_f64(), -2.0);
    /// ```
    #[inline]
    pub fn from_f32_pair(a: f32, b: f32) -> Self {
        Self::new(F4E2M1::from_f64(a as f64), F4E2M1::from_f64(b as f64))
    }

    /// Extracts both values as an `(f32, f32)` pair.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1x2;
    ///
    /// let packed = F4E2M1x2::from_f32_pair(3.0, -0.5);
    /// let (a, b) = packed.to_f32_pair();
    /// assert_eq!(a, 3.0);
    /// assert_eq!(b, -0.5);
    /// ```
    #[inline]
    pub fn to_f32_pair(self) -> (f32, f32) {
        (self.lo().to_f64() as f32, self.hi().to_f64() as f32)
    }
}

impl From<(F4E2M1, F4E2M1)> for F4E2M1x2 {
    #[inline]
    fn from((lo, hi): (F4E2M1, F4E2M1)) -> Self {
        Self::new(lo, hi)
    }
}

impl From<F4E2M1x2> for (F4E2M1, F4E2M1) {
    #[inline]
    fn from(packed: F4E2M1x2) -> Self {
        (packed.lo(), packed.hi())
    }
}

impl std::fmt::Display for F4E2M1x2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "F4E2M1x2({}, {})",
            self.lo().to_f64(),
            self.hi().to_f64()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_roundtrip() {
        for byte in 0..=u8::MAX {
            let packed = F4E2M1x2::from_bits(byte);
            let reconstructed =
                (packed.lo().to_bits() & 0x0F) | ((packed.hi().to_bits() & 0x0F) << 4);
            assert_eq!(reconstructed, byte, "roundtrip failed for 0x{byte:02X}");
        }
    }

    #[test]
    fn nvidia_packing_layout() {
        let packed = F4E2M1x2::from_bits(0xA5);
        assert_eq!(packed.lo().to_bits(), 0x5);
        assert_eq!(packed.hi().to_bits(), 0xA);
    }

    #[test]
    fn zero_constant() {
        assert_eq!(F4E2M1x2::ZERO.to_bits(), 0x00);
        assert_eq!(F4E2M1x2::ZERO.lo().to_f64(), 0.0);
        assert_eq!(F4E2M1x2::ZERO.hi().to_f64(), 0.0);
    }

    #[test]
    fn new_constructor() {
        let packed = F4E2M1x2::new(F4E2M1::from_bits(0x5), F4E2M1::from_bits(0xA));
        assert_eq!(packed.to_bits(), 0xA5);
    }

    #[test]
    fn from_into_tuple() {
        let lo = F4E2M1::from_bits(0x3);
        let hi = F4E2M1::from_bits(0xC);
        let packed = F4E2M1x2::from((lo, hi));
        let (lo2, hi2): (F4E2M1, F4E2M1) = packed.into();
        assert_eq!(lo2.to_bits(), lo.to_bits());
        assert_eq!(hi2.to_bits(), hi.to_bits());
    }

    #[test]
    fn exhaustive_new_roundtrip() {
        for lo_bits in 0u8..16 {
            for hi_bits in 0u8..16 {
                let lo = F4E2M1::from_bits(lo_bits);
                let hi = F4E2M1::from_bits(hi_bits);
                let packed = F4E2M1x2::new(lo, hi);
                assert_eq!(
                    packed.lo().to_bits(),
                    lo_bits,
                    "lo mismatch for new({lo_bits:#X}, {hi_bits:#X})"
                );
                assert_eq!(
                    packed.hi().to_bits(),
                    hi_bits,
                    "hi mismatch for new({lo_bits:#X}, {hi_bits:#X})"
                );
            }
        }
    }

    #[test]
    fn default_is_zero() {
        let d = F4E2M1x2::default();
        assert_eq!(d, F4E2M1x2::ZERO);
        assert_eq!(d.to_bits(), 0x00);
    }

    #[test]
    fn display() {
        let packed = F4E2M1x2::new(F4E2M1::from_f64(1.5), F4E2M1::from_f64(-2.0));
        assert_eq!(format!("{packed}"), "F4E2M1x2(1.5, -2)");
    }

    #[test]
    fn f32_pair_roundtrip() {
        // All exactly-representable F4E2M1 values as f32
        let representable: &[f32] = &[
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        for &a in representable {
            for &b in representable {
                let packed = F4E2M1x2::from_f32_pair(a, b);
                let (ra, rb) = packed.to_f32_pair();
                assert_eq!(ra, a, "lo mismatch for pair ({a}, {b})");
                assert_eq!(rb, b, "hi mismatch for pair ({a}, {b})");
            }
        }
    }
}
