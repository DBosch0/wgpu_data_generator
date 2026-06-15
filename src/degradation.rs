use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DegradationConfig {
    // Gaussian noise
    pub noise_enabled: bool,
    pub noise_sigma_min: f32,
    pub noise_sigma_max: f32,

    // Motion blur (axis-aligned box blur)
    pub blur_enabled: bool,
    pub blur_kernel_min: u32,
    pub blur_kernel_max: u32,

    // JPEG compression round-trip
    pub jpeg_enabled: bool,
    pub jpeg_quality_min: u8,
    pub jpeg_quality_max: u8,

    // Downscale + upscale (nearest-neighbor)
    pub downscale_enabled: bool,
    pub downscale_factor_min: f32, // e.g. 1.5 means scale to 1/1.5 then back
    pub downscale_factor_max: f32,

    // Per-channel color jitter
    pub jitter_enabled: bool,
    pub jitter_scale_min: f32,
    pub jitter_scale_max: f32,

    // Vignette
    pub vignette_enabled: bool,
    pub vignette_strength_min: f32,
    pub vignette_strength_max: f32,
}

impl Default for DegradationConfig {
    fn default() -> Self {
        Self {
            noise_enabled: true,
            noise_sigma_min: 2.0,
            noise_sigma_max: 12.0,

            blur_enabled: true,
            blur_kernel_min: 1,
            blur_kernel_max: 5,

            jpeg_enabled: true,
            jpeg_quality_min: 50,
            jpeg_quality_max: 90,

            downscale_enabled: true,
            downscale_factor_min: 1.0,
            downscale_factor_max: 2.0,

            jitter_enabled: true,
            jitter_scale_min: 0.85,
            jitter_scale_max: 1.15,

            vignette_enabled: true,
            vignette_strength_min: 0.0,
            vignette_strength_max: 0.5,
        }
    }
}

/// Apply the full degradation pipeline to RGBA pixels.
/// Returns the pixel buffer (same dimensions — downscale is always followed by upscale).
pub fn apply_degradation(
    mut rgba: Vec<u8>,
    width: u32,
    height: u32,
    cfg: &DegradationConfig,
    rng: &mut impl Rng,
) -> Vec<u8> {
    // 1. Gaussian noise
    if cfg.noise_enabled {
        let sigma: f32 = rng.random_range(cfg.noise_sigma_min..=cfg.noise_sigma_max);
        let dist = Normal::new(0.0f32, sigma).unwrap();
        for pixel in rgba.chunks_exact_mut(4) {
            for c in 0..3 {
                let v = pixel[c] as f32 + dist.sample(rng);
                pixel[c] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }

    // 2. Motion blur (1D box blur in a random axis-aligned direction)
    if cfg.blur_enabled {
        let kernel = rng.random_range(cfg.blur_kernel_min..=cfg.blur_kernel_max);
        if kernel > 1 {
            let horizontal: bool = rng.random_bool(0.5);
            rgba = box_blur_1d(&rgba, width, height, kernel, horizontal);
        }
    }

    // 3. JPEG compression round-trip — the most impactful sim-to-real step
    if cfg.jpeg_enabled {
        let quality = rng.random_range(cfg.jpeg_quality_min..=cfg.jpeg_quality_max);
        rgba = jpeg_roundtrip(&rgba, width, height, quality);
    }

    // 4. Downscale + upscale (nearest-neighbor)
    if cfg.downscale_enabled {
        let factor: f32 = rng.random_range(cfg.downscale_factor_min..=cfg.downscale_factor_max);
        if factor > 1.01 {
            let sw = ((width as f32 / factor) as u32).max(1);
            let sh = ((height as f32 / factor) as u32).max(1);
            let small = nn_resize(&rgba, width, height, sw, sh);
            rgba = nn_resize(&small, sw, sh, width, height);
        }
    }

    // 5. Color jitter (per-channel multiplicative)
    if cfg.jitter_enabled {
        for pixel in rgba.chunks_exact_mut(4) {
            for c in 0..3 {
                let scale: f32 = rng.random_range(cfg.jitter_scale_min..=cfg.jitter_scale_max);
                let v = pixel[c] as f32 * scale;
                pixel[c] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }

    // 6. Vignette (radial falloff from center)
    if cfg.vignette_enabled {
        let strength: f32 =
            rng.random_range(cfg.vignette_strength_min..=cfg.vignette_strength_max);
        if strength > 0.0 {
            let cx = width as f32 * 0.5;
            let cy = height as f32 * 0.5;
            let max_r = (cx * cx + cy * cy).sqrt();
            for y in 0..height {
                for x in 0..width {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let r = (dx * dx + dy * dy).sqrt();
                    let factor = 1.0 - strength * (r / max_r).powi(2);
                    let idx = ((y * width + x) * 4) as usize;
                    for c in 0..3 {
                        let v = rgba[idx + c] as f32 * factor;
                        rgba[idx + c] = v.clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }

    rgba
}

fn box_blur_1d(src: &[u8], width: u32, height: u32, kernel: u32, horizontal: bool) -> Vec<u8> {
    let mut dst = src.to_vec();
    let half = (kernel / 2) as i32;

    if horizontal {
        for y in 0..height {
            for x in 0..width {
                let mut sum = [0f32; 3];
                let mut count = 0u32;
                for k in -half..=half {
                    let sx = (x as i32 + k).clamp(0, width as i32 - 1) as u32;
                    let idx = ((y * width + sx) * 4) as usize;
                    for c in 0..3 {
                        sum[c] += src[idx + c] as f32;
                    }
                    count += 1;
                }
                let idx = ((y * width + x) * 4) as usize;
                for c in 0..3 {
                    dst[idx + c] = (sum[c] / count as f32) as u8;
                }
            }
        }
    } else {
        for y in 0..height {
            for x in 0..width {
                let mut sum = [0f32; 3];
                let mut count = 0u32;
                for k in -half..=half {
                    let sy = (y as i32 + k).clamp(0, height as i32 - 1) as u32;
                    let idx = ((sy * width + x) * 4) as usize;
                    for c in 0..3 {
                        sum[c] += src[idx + c] as f32;
                    }
                    count += 1;
                }
                let idx = ((y * width + x) * 4) as usize;
                for c in 0..3 {
                    dst[idx + c] = (sum[c] / count as f32) as u8;
                }
            }
        }
    }
    dst
}

fn jpeg_roundtrip(src: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    // JPEG doesn't support alpha — convert RGBA → RGB for encode, then pad back
    let rgb: Vec<u8> = src
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();

    let mut encoded = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut encoded, quality)
        .encode(&rgb, width, height, image::ExtendedColorType::Rgb8)
        .expect("JPEG encode failed");

    image::load_from_memory_with_format(&encoded, image::ImageFormat::Jpeg)
        .expect("JPEG decode failed")
        .to_rgb8()
        .pixels()
        .flat_map(|p| [p[0], p[1], p[2], 255])
        .collect()
}

fn nn_resize(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let mut dst = vec![0u8; (dst_w * dst_h * 4) as usize];
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = ((dx as f32 / dst_w as f32) * src_w as f32) as u32;
            let sy = ((dy as f32 / dst_h as f32) * src_h as f32) as u32;
            let si = ((sy * src_w + sx) * 4) as usize;
            let di = ((dy * dst_w + dx) * 4) as usize;
            dst[di..di + 4].copy_from_slice(&src[si..si + 4]);
        }
    }
    dst
}
