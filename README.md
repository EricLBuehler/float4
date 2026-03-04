# `float4`

**MXFP4-compatible 4-bit floating point types and block formats for Rust.**

This crate provides low-precision floating-point types following the OCP MX specification, designed for efficient storage and computation in machine learning applications where extreme quantization is beneficial.

## Available Types

- **`F4E2M1`**: 4-bit floating-point with 2 exponent bits and 1 mantissa bit
- **`F4E2M1x2`**: Packed pair of two F4E2M1 values in a single byte (NVIDIA `__nv_fp4x2_e2m1` compatible)
- **`E8M0`**: 8-bit scale factor representing powers of two (2^-127 to 2^127)
- **`MXFP4Block`**: Block format storing 32 F4E2M1 values with a shared E8M0 scale

## Features

- **Extreme compression**: 4× smaller than f32 with MXFP4Block format
- **IEEE 754 compliant rounding**: Round-to-nearest-even for F4E2M1
- **Power-of-two scales**: E8M0 provides exact scaling without rounding errors
- **Efficient block storage**: Pack multiple values with shared scale factor
- **NVIDIA compatible packing**: F4E2M1x2 matches `__nv_fp4x2_e2m1` layout for zero-copy CUDA interop
- **Pack/unpack utilities**: Convert between F4E2M1 slices and packed F4E2M1x2 vectors
- **Comprehensive API**: Conversions, constants, and trait implementations

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
float4 = "0.2"
```

## Example Usage

```rust
use float4::F4E2M1;

// Create from f64
let a = F4E2M1::from_f64(1.5);
assert_eq!(a.to_f64(), 1.5);

// Create from raw bits
let b = F4E2M1::from_bits(0x3); // 0b0011 = 1.5
assert_eq!(b.to_f64(), 1.5);

// Arithmetic operations (via f64 conversion)
let x = F4E2M1::from_f64(2.0);
let y = F4E2M1::from_f64(3.0);
let sum = F4E2M1::from_f64(x.to_f64() + y.to_f64());
assert_eq!(sum.to_f64(), 5.0); // May round to nearest representable value

// Constants
assert_eq!(F4E2M1::MAX.to_f64(), 6.0);
assert_eq!(F4E2M1::MIN.to_f64(), -6.0);
assert_eq!(F4E2M1::EPSILON.to_f64(), 0.5);
```

### Packed Pairs (F4E2M1x2)

Two F4E2M1 values packed into a single byte, matching NVIDIA's `__nv_fp4x2_e2m1` layout (lower nibble = first value, upper nibble = second value):

```rust
use float4::{F4E2M1, F4E2M1x2, pack, unpack};

// Pack two values into one byte
let pair = F4E2M1x2::new(F4E2M1::from_f64(1.5), F4E2M1::from_f64(-2.0));
assert_eq!(pair.lo().to_f64(), 1.5);
assert_eq!(pair.hi().to_f64(), -2.0);

// Convert from f32 pairs directly
let pair = F4E2M1x2::from_f32_pair(3.0, 0.5);
let (a, b) = pair.to_f32_pair();
assert_eq!(a, 3.0);
assert_eq!(b, 0.5);

// Pack a slice of F4E2M1 values into pairs
let values = vec![
    F4E2M1::from_f64(1.0),
    F4E2M1::from_f64(2.0),
    F4E2M1::from_f64(3.0),
    F4E2M1::from_f64(4.0),
];
let packed = pack(&values);   // [F4E2M1x2(1.0, 2.0), F4E2M1x2(3.0, 4.0)]
let unpacked = unpack(&packed); // [1.0, 2.0, 3.0, 4.0]
assert_eq!(values, unpacked);
```

### Block Format Example

```rust
use float4::{F4E2M1, E8M0, MXFP4Block};

// Original data
let data = vec![1.5, -2.0, 0.5, 3.0, 1.0, -0.5];

// Compute scale factor (rounds up to power of 2)
let scale = E8M0::from_f32_slice(&data);
assert_eq!(scale.to_f64(), 4.0); // 3.0 rounds up to 4.0

// Quantize to F4E2M1
let mut quantized = [F4E2M1::from_f64(0.0); 32];
for (i, &value) in data.iter().enumerate() {
    quantized[i] = F4E2M1::from_f64(value as f64 / scale.to_f64());
}

// Pack into block (17 bytes for 32 values vs 128 bytes for f32)
let block = MXFP4Block::from_f32_slice(quantized, scale);

// Retrieve values
let restored = block.to_f32_array();
assert_eq!(restored[0], 1.5);
assert_eq!(restored[1], -2.0);
```

## E8M0 Scale Factors

The E8M0 type represents scale factors as exact powers of two:

```rust
use float4::E8M0;

// Exact powers of two are preserved
let scale = E8M0::from(4.0);
assert_eq!(scale.to_f64(), 4.0);

// Non-powers round UP to next power of two
let scale = E8M0::from(3.0);
assert_eq!(scale.to_f64(), 4.0);  // 3.0 → 4.0

let scale = E8M0::from(5.0);
assert_eq!(scale.to_f64(), 8.0);  // 5.0 → 8.0

// Computing scale from data
let data = [1.5, -2.0, 0.5, 3.0];
let scale = E8M0::from_f32_slice(&data);
assert_eq!(scale.to_f64(), 4.0);  // max(|data|) = 3.0 → 4.0
```

Key characteristics:
- Range: 2^-127 to 2^127
- Always rounds UP (toward positive infinity)
- No rounding errors when scaling by powers of two
- Ideal for block quantization schemes

## Representable Values

F4E2M1 can exactly represent 16 distinct values:

| Value | Bit Pattern | Type |
|-------|-------------|------|
| 0.0   | 0000 | Zero |
| 0.5   | 0001 | Subnormal |
| 1.0   | 0010 | Normal |
| 1.5   | 0011 | Normal |
| 2.0   | 0100 | Normal |
| 3.0   | 0101 | Normal |
| 4.0   | 0110 | Normal |
| 6.0   | 0111 | Normal |
| -0.0  | 1000 | Negative zero |
| -0.5  | 1001 | Subnormal |
| -1.0  | 1010 | Normal |
| -1.5  | 1011 | Normal |
| -2.0  | 1100 | Normal |
| -3.0  | 1101 | Normal |
| -4.0  | 1110 | Normal |
| -6.0  | 1111 | Normal |

## Special Values

Unlike standard floating point formats, F4E2M1 has no representation for infinity or NaN. These values saturate to the maximum representable value:

```rust
use float4::F4E2M1;

assert_eq!(F4E2M1::from_f64(f64::INFINITY).to_f64(), 6.0);
assert_eq!(F4E2M1::from_f64(f64::NEG_INFINITY).to_f64(), -6.0);
assert_eq!(F4E2M1::from_f64(f64::NAN).to_f64(), 6.0);
```
