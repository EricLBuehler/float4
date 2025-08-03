//! E8M0 floating-point format implementation.
//!
//! This module provides an implementation of NVIDIA's e8m0 format, an 8-bit
//! floating-point representation that stores only scale factors (powers of two).

/// An 8-bit floating-point type that represents scale factors as powers of two.
///
/// # Format Details
///
/// The e8m0 format is an 8-bit representation where:
/// - Values 0-254: Represent powers of two from 2^-127 to 2^127
/// - Value 255 (0xFF): Reserved for NaN
/// - No mantissa bits - only represents exact powers of two
/// - Exponent bias: 127
///
/// # Conversion Behavior
///
/// This implementation follows NVIDIA's CUDA specification:
/// - **Rounding mode**: round toward positive infinity.
/// - **Saturation mode**: clamp values to representable range (satfinite).
///
/// ## From f64 to E8M0
/// - NaN → 0xFF (NaN)
/// - Values ≤ 0 → 0x00 (2^-127, smallest positive value)
/// - Values are rounded UP to the next power of two
/// - Values > 2^127 → 0xFE (2^127, largest finite value)
///
/// ## From E8M0 to f64
/// - 0x00-0xFE → 2^(value - 127)
/// - 0xFF → NaN
///
/// # Examples
///
/// ```rust
/// use float4::E8M0;
///
/// // Exact powers of two convert precisely
/// let e = E8M0::from(4.0_f64);
/// assert_eq!(e.to_f64(), 4.0);
///
/// // Non-powers round UP to next power of two
/// let e = E8M0::from(3.0_f64);
/// assert_eq!(e.to_f64(), 4.0);  // rounds up
///
/// // Special values
/// assert!(E8M0::from(f64::NAN).to_f64().is_nan());
/// assert_eq!(E8M0::from(-1.0).to_f64(), 2f64.powi(-127));  // clamps to minimum
/// ```
///
/// # Reference
///
/// Based on NVIDIA's CUDA e8m0 specification:
/// <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8__e8m0.html>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct E8M0(u8);

const _: () = assert!(std::mem::size_of::<E8M0>() == 1);

impl E8M0 {
    /// The NaN (Not-a-Number) value for E8M0 format.
    ///
    /// Represented by the bit pattern 0xFF (255).
    pub const NAN: Self = Self(0xFF);

    /// Converts an f64 to E8M0 format.
    ///
    /// This implementation follows NVIDIA's specification with:
    /// - **Rounding**: `cudaRoundPosInf` - rounds toward positive infinity
    /// - **Saturation**: `__NV_SATFINITE` - clamps to representable range
    ///
    /// # Conversion Rules
    ///
    /// - NaN → E8M0::NAN (0xFF)
    /// - Values ≤ 0 → 0x00 (represents 2^-127)
    /// - Positive values are rounded UP to the next power of two
    /// - Values > 2^127 → 0xFE (represents 2^127)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use float4::E8M0;
    ///
    /// // Exact powers of two
    /// assert_eq!(E8M0::from(1.0).to_f64(), 1.0);   // 2^0
    /// assert_eq!(E8M0::from(2.0).to_f64(), 2.0);   // 2^1
    ///
    /// // Non-powers round UP
    /// assert_eq!(E8M0::from(1.5).to_f64(), 2.0);   // rounds to 2^1
    /// assert_eq!(E8M0::from(3.0).to_f64(), 4.0);   // rounds to 2^2
    ///
    /// // Edge cases
    /// assert_eq!(E8M0::from(0.0).to_f64(), 2f64.powi(-127));   // minimum
    /// assert_eq!(E8M0::from(-5.0).to_f64(), 2f64.powi(-127));  // negative → minimum
    /// assert_eq!(E8M0::from(f64::INFINITY).to_f64(), 2f64.powi(127));  // saturates
    /// ```
    #[inline(always)]
    pub fn from_f64(value: f64) -> Self {
        // NaN propagates.
        if value.is_nan() {
            return Self::NAN;
        }
        // Non-positive becomes the smallest code.
        if value <= 0.0 {
            return Self(0x00);
        }

        // log2 plus bias, then round *toward +∞* (cudaRoundPosInf).
        let exp = (value.log2()).ceil() as i32; // round-up

        // Use saturating add to prevent overflow
        let biased = exp.saturating_add(127);

        // __NV_SATFINITE behaviour: clamp to representable range.
        match biased {
            n if n < 0 => Self(0x00),   // underflow
            n if n > 254 => Self(0xFE), // overflow
            n => Self(n as u8),
        }
    }

    /// Converts this E8M0 value to an f64.
    ///
    /// # Returns
    ///
    /// - For bits 0x00-0xFE: Returns 2^(bits - 127)
    /// - For bits 0xFF: Returns NaN
    ///
    /// # Examples
    ///
    /// ```rust
    /// use float4::E8M0;
    ///
    /// assert_eq!(E8M0::from_bits(0x7F).to_f64(), 1.0);  // 2^(127-127) = 2^0 = 1
    /// assert_eq!(E8M0::from_bits(0x80).to_f64(), 2.0);  // 2^(128-127) = 2^1 = 2
    /// assert!(E8M0::NAN.to_f64().is_nan());
    /// ```
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        match self.0 {
            0xFF => f64::NAN,
            b => 2f64.powi(b as i32 - 127),
        }
    }

    /// Creates an E8M0 scale factor from a slice of f32 values.
    ///
    /// This function computes an appropriate scale factor for quantizing the given values.
    /// It finds the maximum absolute value in the slice and converts it to a power of two
    /// scale factor following E8M0 conversion rules.
    ///
    /// # Arguments
    ///
    /// * `values` - A slice of f32 values to compute the scale from
    ///
    /// # Returns
    ///
    /// An E8M0 scale factor that can represent the largest value in the slice when
    /// multiplied by the quantized values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use float4::E8M0;
    ///
    /// // Scale for values within a small range
    /// let values = [0.5, -0.75, 0.25];
    /// let scale = E8M0::from_f32_slice(&values);
    /// assert_eq!(scale.to_f64(), 1.0);  // rounds 0.75 up to 1.0
    ///
    /// // Scale for larger values
    /// let values = [1.0, 5.0, -3.5];
    /// let scale = E8M0::from_f32_slice(&values);
    /// assert_eq!(scale.to_f64(), 8.0);  // rounds 5.0 up to 8.0
    ///
    /// // Empty slice returns smallest scale
    /// let scale = E8M0::from_f32_slice(&[]);
    /// assert_eq!(scale.to_f64(), 2f64.powi(-127));
    /// ```
    #[inline(always)]
    pub fn from_f32_slice(values: &[f32]) -> Self {
        // Find maximum absolute value
        let max_abs = values.iter().map(|&x| x.abs()).fold(0.0f32, |a, b| {
            if b.is_nan() || a.is_nan() {
                f32::NAN
            } else if b > a {
                b
            } else {
                a
            }
        });

        // Convert to E8M0 (will round up to next power of 2)
        Self::from_f64(max_abs as f64)
    }

    /// Creates an E8M0 from raw bits.
    ///
    /// This performs no validation - the u8 value is directly used as the bit pattern.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use float4::E8M0;
    ///
    /// let e = E8M0::from_bits(0x7F);
    /// assert_eq!(e.to_f64(), 1.0);  // 2^(127-127) = 1
    ///
    /// let e = E8M0::from_bits(0xFF);
    /// assert!(e.to_f64().is_nan());  // 0xFF is NaN
    /// ```
    #[inline(always)]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Extracts the raw bits from an E8M0.
    ///
    /// Returns the underlying 8-bit representation.
    #[inline(always)]
    pub const fn to_bits(&self) -> u8 {
        self.0
    }
}

// ----- Conversions ---------------------------------------------------------

impl From<f32> for E8M0 {
    /// Converts an f32 to E8M0 format.
    ///
    /// This is equivalent to converting via f64.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use float4::E8M0;
    ///
    /// let e: E8M0 = 2.5f32.into();
    /// assert_eq!(e.to_f64(), 4.0); // rounds up to 2^2
    /// ```
    #[inline(always)]
    fn from(value: f32) -> Self {
        Self::from_f64(value as f64)
    }
}

impl From<f64> for E8M0 {
    #[inline(always)]
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

/// Converts E8M0 to f64.
///
/// This is a convenience trait implementation that calls [`E8M0::to_f64`].
impl From<E8M0> for f64 {
    #[inline(always)]
    fn from(v: E8M0) -> Self {
        v.to_f64()
    }
}

/// Creates an E8M0 from raw bits.
///
/// This performs no validation - the u8 value is directly used as the bit pattern.
///
/// # Examples
///
/// ```rust
/// use float4::E8M0;
///
/// let e = E8M0::from(0x7Fu8);
/// assert_eq!(e.to_f64(), 1.0);  // 2^(127-127) = 1
///
/// let e = E8M0::from(0xFFu8);
/// assert!(e.to_f64().is_nan());  // 0xFF is NaN
/// ```
impl From<u8> for E8M0 {
    #[inline(always)]
    fn from(b: u8) -> Self {
        Self::from_bits(b)
    }
}

/// Extracts the raw bits from an E8M0.
///
/// Returns the underlying 8-bit representation.
impl From<E8M0> for u8 {
    #[inline(always)]
    fn from(v: E8M0) -> Self {
        v.to_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8m0_conversion_roundtrip_and_edges() {
        // Exact expectations for a handful of edge / corner cases.
        let fixed = [
            (-1.0, 0x00u8), // negative → clamp down
            (0.0, 0x00),
            (2f64.powi(-127), 0x00), // smallest positive power of two
            (0.75, 0x7F),
            (1.0, 0x7F),
            (1.5, 0x80),
            (2.0, 0x80),
            (2f64.powi(127), 0xFE), // largest finite
            (f64::NAN, 0xFF),
        ];
        for &(x, bits) in &fixed {
            assert_eq!(E8M0::from(x).to_bits(), bits);
        }

        // Round-trip invariant for every representable power-of-two.
        for exp in -127..=127 {
            let x = 2f64.powi(exp);
            let code = E8M0::from(x);
            let y: f64 = code.into();
            assert_eq!(x, y);
        }

        // Values between powers of two must round *up*.
        let x = 1.3_f64;
        let y: f64 = E8M0::from(x).into();
        assert!(y >= x && y <= 2.0);
    }

    #[test]
    fn test_from_f32_slice() {
        // Test with typical values
        let values = [1.0f32, -2.0, 0.5];
        let scale = E8M0::from_f32_slice(&values);
        assert_eq!(scale.to_f64(), 2.0); // max abs is 2.0

        // Test with values that need rounding up
        let values = [1.5f32, -2.5, 0.5];
        let scale = E8M0::from_f32_slice(&values);
        assert_eq!(scale.to_f64(), 4.0); // 2.5 rounds up to 4.0

        // Test with small values
        let values = [0.1f32, -0.2, 0.15];
        let scale = E8M0::from_f32_slice(&values);
        assert_eq!(scale.to_f64(), 0.25); // 0.2 rounds up to 0.25

        // Test with large values
        let values = [100.0f32, -50.0, 75.0];
        let scale = E8M0::from_f32_slice(&values);
        assert_eq!(scale.to_f64(), 128.0); // 100 rounds up to 128

        // Test with NaN
        let values = [1.0f32, f32::NAN, -2.0];
        let scale = E8M0::from_f32_slice(&values);
        assert!(scale.to_f64().is_nan());

        // Test with infinity
        let values = [1.0f32, f32::INFINITY, -2.0];
        let scale = E8M0::from_f32_slice(&values);
        assert_eq!(scale.to_f64(), 2f64.powi(127)); // saturates to max

        // Test empty slice
        let scale = E8M0::from_f32_slice(&[]);
        assert_eq!(scale.to_f64(), 2f64.powi(-127)); // smallest positive value

        // Test with zeros
        let values = [0.0f32, -0.0, 0.0];
        let scale = E8M0::from_f32_slice(&values);
        assert_eq!(scale.to_f64(), 2f64.powi(-127)); // smallest positive value

        // Test single value
        let values = [3.0f32];
        let scale = E8M0::from_f32_slice(&values);
        assert_eq!(scale.to_f64(), 4.0); // 3.0 rounds up to 4.0
    }
}
