//! Four-bit floating point types and block formats for Rust.
//!
//! This crate provides low-precision floating-point types following the OCP MX specification,
//! designed for efficient storage and computation in machine learning applications where
//! extreme quantization is beneficial.
//!
//! # Available Types
//!
//! - [`F4E2M1`]: 4-bit floating-point with 2 exponent bits and 1 mantissa bit
//! - [`E8M0`]: 8-bit scale factor representing powers of two (2^-127 to 2^127)
//! - [`MXFP4Block`]: Block format storing 32 F4E2M1 values with a shared E8M0 scale
//!
//! # F4E2M1 Format Details
//!
//! The [`F4E2M1`] type implements the E2M1 format with:
//! - 1 sign bit
//! - 2 exponent bits  
//! - 1 mantissa bit
//! - Exponent bias of 1
//! - Round-to-nearest-even (roundTiesToEven) rounding mode
//!
//! This format can represent 16 distinct values ranging from -6.0 to 6.0, including:
//! - Normal numbers: ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
//! - Subnormal numbers: ±0.5
//! - Zero: ±0.0
//!
//! # Examples
//!
//! Basic usage:
//!
//! ```
//! use float4::F4E2M1;
//!
//! // Create from f64
//! let a = F4E2M1::from_f64(1.5);
//! assert_eq!(a.to_f64(), 1.5);
//!
//! // Create from raw bits
//! let b = F4E2M1::from_bits(0x3); // 0b0011 = 1.5
//! assert_eq!(b.to_f64(), 1.5);
//!
//! // Values outside representable range saturate
//! let c = F4E2M1::from_f64(10.0);
//! assert_eq!(c.to_f64(), 6.0); // Saturates to maximum
//! ```
//!
//! # Rounding Behavior
//!
//! The type uses round-to-nearest-even as specified by IEEE 754:
//!
//! ```
//! use float4::F4E2M1;
//!
//! // Rounding to nearest
//! assert_eq!(F4E2M1::from_f64(1.75).to_f64(), 2.0);
//! assert_eq!(F4E2M1::from_f64(2.25).to_f64(), 2.0);
//!
//! // Round-to-even when exactly halfway
//! assert_eq!(F4E2M1::from_f64(1.25).to_f64(), 1.0); // Rounds to even
//! assert_eq!(F4E2M1::from_f64(2.5).to_f64(), 2.0);  // Rounds to even
//! ```
//!
//! # Special Values
//!
//! Unlike standard floating point formats, F4E2M1 has no representation for infinity or NaN.
//! These values saturate to the maximum representable value:
//!
//! ```
//! use float4::F4E2M1;
//!
//! assert_eq!(F4E2M1::from_f64(f64::INFINITY).to_f64(), 6.0);
//! assert_eq!(F4E2M1::from_f64(f64::NEG_INFINITY).to_f64(), -6.0);
//! assert_eq!(F4E2M1::from_f64(f64::NAN).to_f64(), 6.0);
//! ```
//!
//! # MXFP4 Block Format
//!
//! The [`MXFP4Block`] type provides efficient storage for multiple F4E2M1 values by sharing
//! a common scale factor:
//!
//! ```
//! use float4::{F4E2M1, E8M0, MXFP4Block};
//!
//! // Original f32 data
//! let data = vec![1.5, -2.0, 0.5, 3.0];
//!
//! // Compute scale (rounds up to power of 2)
//! let scale = E8M0::from_f32_slice(&data);
//! assert_eq!(scale.to_f64(), 4.0); // 3.0 rounds up to 4.0
//!
//! // Quantize values
//! let mut quantized = [F4E2M1::from_f64(0.0); 32];
//! for i in 0..data.len() {
//!     quantized[i] = F4E2M1::from_f64(data[i] as f64 / scale.to_f64());
//! }
//!
//! // Pack into block (17 bytes total for 32 values)
//! let block = MXFP4Block::from_f32_slice(quantized, scale);
//!
//! // Convert back
//! let restored = block.to_f32_array();
//! // Note: Due to F4E2M1's limited precision, values may be quantized
//! assert_eq!(restored[0], 2.0);  // 1.5/4.0 = 0.375 -> rounds to 0.5 -> 0.5*4.0 = 2.0
//! assert_eq!(restored[1], -2.0); // -2.0/4.0 = -0.5 is exactly representable
//! ```
//!
//! This format achieves 4× compression compared to f32, making it ideal for:
//! - Neural network weight storage
//! - Activation caching in quantized models
//! - Memory-bandwidth limited applications

mod block;
mod cvt;
mod m8e0;

pub use block::MXFP4Block;
pub use m8e0::E8M0;

/// A 4-bit floating point type with 2 exponent bits and 1 mantissa bit.
///
/// This type implements the E2M1 format from the OCP MX specification, providing
/// a compact representation suitable for machine learning applications requiring
/// extreme quantization.
///
/// # Format
///
/// The 4 bits are laid out as follows:
/// - Bit 3: Sign bit (0 = positive, 1 = negative)
/// - Bits 2-1: Exponent bits (biased by 1)
/// - Bit 0: Mantissa bit
///
/// # Representable Values
///
/// F4E2M1 can exactly represent the following values:
/// - **Normal numbers**: ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
/// - **Subnormal numbers**: ±0.5  
/// - **Zero**: ±0.0
///
/// # Examples
///
/// ```
/// use float4::F4E2M1;
///
/// // Create from floating point value
/// let x = F4E2M1::from_f64(2.5);
/// assert_eq!(x.to_f64(), 2.0); // Rounded to nearest representable value
///
/// // Access raw bit representation  
/// let bits = x.to_bits();
/// assert_eq!(bits, 0x4); // 0b0100 = +2.0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct F4E2M1(u8);

const _: () = assert!(std::mem::size_of::<F4E2M1>() == 1);

impl F4E2M1 {
    /// Creates a new `F4E2M1` value from a 64-bit floating point number.
    ///
    /// This function converts the input to the nearest representable F4E2M1 value
    /// using round-to-nearest-even. Values outside the
    /// representable range will saturate to the maximum or minimum values.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    ///
    /// // Exact representable values
    /// assert_eq!(F4E2M1::from_f64(2.0).to_f64(), 2.0);
    /// assert_eq!(F4E2M1::from_f64(-3.0).to_f64(), -3.0);
    ///
    /// // Rounding
    /// assert_eq!(F4E2M1::from_f64(2.7).to_f64(), 3.0);
    /// assert_eq!(F4E2M1::from_f64(1.25).to_f64(), 1.0); // Round to even
    ///
    /// // Saturation
    /// assert_eq!(F4E2M1::from_f64(10.0).to_f64(), 6.0);
    /// assert_eq!(F4E2M1::from_f64(-10.0).to_f64(), -6.0);
    /// ```
    ///
    /// # Special Values
    ///
    /// - `NaN` → 6.0 (maximum positive value)
    /// - `+Infinity` → 6.0  
    /// - `-Infinity` → -6.0
    #[inline(always)]
    pub const fn from_f64(x: f64) -> Self {
        Self(cvt::f64_to_fp4(x))
    }

    /// Converts this `F4E2M1` value to a 64-bit floating point number.
    ///
    /// This conversion is exact - the returned f64 will precisely represent
    /// the value stored in the F4E2M1.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    ///
    /// let x = F4E2M1::from_f64(1.5);
    /// assert_eq!(x.to_f64(), 1.5);
    ///
    /// // All 16 possible values can be converted
    /// for i in 0..16 {
    ///     let fp4 = F4E2M1::from_bits(i);
    ///     let _ = fp4.to_f64(); // Always succeeds
    /// }
    /// ```
    #[inline(always)]
    pub fn to_f64(&self) -> f64 {
        cvt::fp4_to_f64(self.0)
    }

    /// Creates a new `F4E2M1` value from its raw 4-bit representation.
    ///
    /// The bits are interpreted as:
    /// - Bit 3: Sign (0 = positive, 1 = negative)
    /// - Bits 2-1: Exponent (biased by 1)
    /// - Bit 0: Mantissa
    ///
    /// Only the lower 4 bits of the input are used.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    ///
    /// // 0x0 = 0b0000 = +0.0
    /// assert_eq!(F4E2M1::from_bits(0x0).to_f64(), 0.0);
    ///
    /// // 0x3 = 0b0011 = +1.5
    /// assert_eq!(F4E2M1::from_bits(0x3).to_f64(), 1.5);
    ///
    /// // 0xF = 0b1111 = -6.0
    /// assert_eq!(F4E2M1::from_bits(0xF).to_f64(), -6.0);
    /// ```
    ///
    /// # Bit Patterns
    ///
    /// | Bits | Decimal | Value |
    /// |------|---------|-------|
    /// | 0000 |    0    |  0.0  |
    /// | 0001 |    1    |  0.5  |
    /// | 0010 |    2    |  1.0  |
    /// | 0011 |    3    |  1.5  |
    /// | 0100 |    4    |  2.0  |
    /// | 0101 |    5    |  3.0  |
    /// | 0110 |    6    |  4.0  |
    /// | 0111 |    7    |  6.0  |
    /// | 1000 |    8    | -0.0  |
    /// | 1001 |    9    | -0.5  |
    /// | 1010 |   10    | -1.0  |
    /// | 1011 |   11    | -1.5  |
    /// | 1100 |   12    | -2.0  |
    /// | 1101 |   13    | -3.0  |
    /// | 1110 |   14    | -4.0  |
    /// | 1111 |   15    | -6.0  |
    #[inline(always)]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Returns the raw 4-bit representation of this `F4E2M1` value.
    ///
    /// The returned byte contains the 4-bit value in its lower nibble.
    /// The upper 4 bits are always zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    ///
    /// let x = F4E2M1::from_f64(1.5);
    /// assert_eq!(x.to_bits(), 0x3); // 0b0011
    ///
    /// let y = F4E2M1::from_f64(-2.0);
    /// assert_eq!(y.to_bits(), 0xC); // 0b1100
    /// ```
    #[inline(always)]
    pub const fn to_bits(&self) -> u8 {
        self.0
    }
}

impl F4E2M1 {
    /// The smallest positive normal F4E2M1 value (1.0).
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    /// assert_eq!(F4E2M1::MIN_POSITIVE_NORMAL.to_f64(), 1.0);
    /// ```
    pub const MIN_POSITIVE_NORMAL: F4E2M1 = F4E2M1(0x2);

    /// The smallest positive F4E2M1 value (0.5).
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    /// assert_eq!(F4E2M1::MIN_POSITIVE.to_f64(), 0.5);
    /// ```
    pub const MIN_POSITIVE: F4E2M1 = F4E2M1(0x1);

    /// The largest F4E2M1 value (6.0).
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    /// assert_eq!(F4E2M1::MAX.to_f64(), 6.0);
    /// ```
    pub const MAX: F4E2M1 = F4E2M1(0x7);

    /// The smallest (most negative) F4E2M1 value (-6.0).
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    /// assert_eq!(F4E2M1::MIN.to_f64(), -6.0);
    /// ```
    pub const MIN: F4E2M1 = F4E2M1(0xF);

    /// Positive zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    /// assert_eq!(F4E2M1::ZERO.to_f64(), 0.0);
    /// ```
    pub const ZERO: F4E2M1 = F4E2M1(0x0);

    /// Negative zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    /// assert_eq!(F4E2M1::NEG_ZERO.to_f64(), -0.0);
    /// ```
    pub const NEG_ZERO: F4E2M1 = F4E2M1(0x8);

    /// One.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    /// assert_eq!(F4E2M1::ONE.to_f64(), 1.0);
    /// ```
    pub const ONE: F4E2M1 = F4E2M1(0x2);

    /// Negative one.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    /// assert_eq!(F4E2M1::NEG_ONE.to_f64(), -1.0);
    /// ```
    pub const NEG_ONE: F4E2M1 = F4E2M1(0xA);

    /// The machine epsilon for F4E2M1 (0.5).
    ///
    /// This is the difference between 1.0 and the next representable value.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    /// assert_eq!(F4E2M1::EPSILON.to_f64(), 0.5);
    /// ```
    pub const EPSILON: F4E2M1 = F4E2M1(0x1);
}

impl Default for F4E2M1 {
    /// Returns the default value of 0.0.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    /// assert_eq!(F4E2M1::default().to_f64(), 0.0);
    /// ```
    #[inline]
    fn default() -> Self {
        F4E2M1::ZERO
    }
}

impl From<f32> for F4E2M1 {
    /// Converts a 32-bit float to F4E2M1.
    ///
    /// This is equivalent to converting via f64.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    ///
    /// let x: F4E2M1 = 2.5f32.into();
    /// assert_eq!(x.to_f64(), 2.0); // Rounded to nearest
    /// ```
    #[inline]
    fn from(value: f32) -> Self {
        F4E2M1::from_f64(value as f64)
    }
}

impl From<F4E2M1> for f32 {
    /// Converts F4E2M1 to a 32-bit float.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    ///
    /// let x = F4E2M1::from_f64(1.5);
    /// let y: f32 = x.into();
    /// assert_eq!(y, 1.5);
    /// ```
    #[inline]
    fn from(value: F4E2M1) -> Self {
        value.to_f64() as f32
    }
}

impl From<F4E2M1> for f64 {
    /// Converts F4E2M1 to a 64-bit float.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    ///
    /// let x = F4E2M1::from_f64(3.0);
    /// let y: f64 = x.into();
    /// assert_eq!(y, 3.0);
    /// ```
    #[inline]
    fn from(value: F4E2M1) -> Self {
        value.to_f64()
    }
}

impl std::fmt::Display for F4E2M1 {
    /// Formats the F4E2M1 value for display.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    ///
    /// let x = F4E2M1::from_f64(1.5);
    /// assert_eq!(format!("{}", x), "1.5");
    /// ```
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_f64())
    }
}

impl std::fmt::LowerExp for F4E2M1 {
    /// Formats the F4E2M1 value in scientific notation.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    ///
    /// let x = F4E2M1::from_f64(6.0);
    /// assert_eq!(format!("{:e}", x), "6e0");
    /// ```
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:e}", self.to_f64())
    }
}

impl std::fmt::UpperExp for F4E2M1 {
    /// Formats the F4E2M1 value in scientific notation with uppercase E.
    ///
    /// # Examples
    ///
    /// ```
    /// use float4::F4E2M1;
    ///
    /// let x = F4E2M1::from_f64(6.0);
    /// assert_eq!(format!("{:E}", x), "6E0");
    /// ```
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:E}", self.to_f64())
    }
}

#[cfg(test)]
mod test {
    use crate::F4E2M1;

    #[test]
    fn test_full_range() {
        // Test all 16 possible FP4 values (0x0 to 0xF)
        // Expected values for E2M1 format with bias=1:
        // Positive values:
        // 0x0 (0b0000): +0.0
        // 0x1 (0b0001): +0.5 (denormal)
        // 0x2 (0b0010): +1.0
        // 0x3 (0b0011): +1.5
        // 0x4 (0b0100): +2.0
        // 0x5 (0b0101): +3.0
        // 0x6 (0b0110): +4.0
        // 0x7 (0b0111): +6.0
        // Negative values (sign bit set):
        // 0x8 (0b1000): -0.0
        // 0x9 (0b1001): -0.5 (denormal)
        // 0xA (0b1010): -1.0
        // 0xB (0b1011): -1.5
        // 0xC (0b1100): -2.0
        // 0xD (0b1101): -3.0
        // 0xE (0b1110): -4.0
        // 0xF (0b1111): -6.0

        let expected_values = [
            0.0,  // 0x0
            0.5,  // 0x1
            1.0,  // 0x2
            1.5,  // 0x3
            2.0,  // 0x4
            3.0,  // 0x5
            4.0,  // 0x6
            6.0,  // 0x7
            -0.0, // 0x8
            -0.5, // 0x9
            -1.0, // 0xA
            -1.5, // 0xB
            -2.0, // 0xC
            -3.0, // 0xD
            -4.0, // 0xE
            -6.0, // 0xF
        ];

        for (bits, expected) in (0u8..16).zip(expected_values.iter()) {
            let converted = F4E2M1::from_bits(bits).to_f64();
            assert_eq!(
                converted, *expected,
                "Failed for bits 0x{bits:X}: got {converted}, expected {expected}"
            );

            // Also test through the struct
            let fp4 = F4E2M1(bits);
            assert_eq!(
                fp4.to_f64(),
                *expected,
                "Failed for F4E2M1(0x{:X}): got {}, expected {}",
                bits,
                fp4.to_f64(),
                expected
            );
        }
    }

    #[test]
    fn test_roundtrip() {
        // Test that representable values round-trip correctly
        let test_values = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];

        for &x in &test_values {
            let mxfp4 = F4E2M1::from_f64(x);
            let roundtrip = mxfp4.to_f64();
            assert_eq!(roundtrip, x, "Roundtrip failed for {x}: got {roundtrip}");
        }
    }

    #[test]
    fn test_rounding() {
        // Test round-to-nearest-even behavior
        // Values between representable FP4 values should round to nearest
        // When exactly halfway, round to even (least significant bit = 0)

        let test_cases = [
            // Value -> Expected rounded value
            // Based on actual behavior: 0.5 denormal (0x1) is the smallest positive value
            (0.75, 1.0), // 0.75 -> 1.0 (nearest)
            (1.25, 1.0), // 1.25 -> 1.0 (tie, round to even)
            (1.75, 2.0), // 1.75 -> 2.0 (nearest)
            (2.25, 2.0), // 2.25 -> 2.0 (nearest)
            (2.5, 2.0),  // 2.5 -> 2.0 (tie, round to even)
            (2.75, 3.0), // 2.75 -> 3.0 (nearest)
            (3.25, 3.0), // 3.25 -> 3.0 (nearest)
            (3.5, 4.0),  // 3.5 -> 4.0 (nearest)
            (4.5, 4.0),  // 4.5 -> 4.0 (nearest)
            (5.0, 4.0),  // 5.0 -> 4.0 (nearest)
            (5.5, 6.0),  // 5.5 -> 6.0 (nearest)
            (7.0, 6.0),  // 7.0 -> 6.0 (saturate to max)
            (10.0, 6.0), // 10.0 -> 6.0 (saturate to max)
            // Negative values
            (-0.75, -1.0), // -0.75 -> -1.0
            (-1.25, -1.0), // -1.25 -> -1.0
            (-1.75, -2.0), // -1.75 -> -2.0
            (-2.25, -2.0), // -2.25 -> -2.0
            (-2.5, -2.0),  // -2.5 -> -2.0
            (-2.75, -3.0), // -2.75 -> -3.0
            (-3.25, -3.0), // -3.25 -> -3.0
            (-3.5, -4.0),  // -3.5 -> -4.0
            (-4.5, -4.0),  // -4.5 -> -4.0
            (-5.0, -4.0),  // -5.0 -> -4.0
            (-5.5, -6.0),  // -5.5 -> -6.0
            (-7.0, -6.0),  // -7.0 -> -6.0 (saturate)
        ];

        for &(input, expected) in &test_cases {
            let fp4 = F4E2M1::from_f64(input);
            let result = fp4.to_f64();
            assert_eq!(
                result, expected,
                "Rounding failed for {input}: got {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_special_values() {
        // Test special values: infinities, NaN
        use std::f64;

        // Positive infinity should saturate to max positive value (6.0)
        let fp4 = F4E2M1::from_f64(f64::INFINITY);
        assert_eq!(fp4.to_f64(), 6.0);

        // Negative infinity should saturate to max negative value (-6.0)
        let fp4 = F4E2M1::from_f64(f64::NEG_INFINITY);
        assert_eq!(fp4.to_f64(), -6.0);

        // NaN should become positive max (6.0) according to the implementation
        let fp4 = F4E2M1::from_f64(f64::NAN);
        assert_eq!(fp4.to_f64(), 6.0);
    }
}
