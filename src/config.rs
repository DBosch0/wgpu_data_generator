use serde::{Deserialize, Serialize};

use crate::degradation::DegradationConfig;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RangeF32 {
    pub min: f32,
    pub max: f32,
}

impl RangeF32 {
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }
    pub fn sample(&self, rng: &mut impl rand::Rng) -> f32 {
        rng.random_range(self.min..=self.max)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CameraConfig {
    /// Horizontal rotation around Y axis, degrees
    pub yaw_range: RangeF32,
    /// Vertical angle from horizon, degrees
    pub pitch_range: RangeF32,
    /// Roll around the view axis, degrees
    pub roll_range: RangeF32,
    /// Distance from the face origin
    pub distance_range: RangeF32,
    /// Vertical field of view in degrees
    pub fov_deg: f32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            yaw_range: RangeF32::new(-30.0, 30.0),
            pitch_range: RangeF32::new(-20.0, 20.0),
            roll_range: RangeF32::new(-10.0, 10.0),
            distance_range: RangeF32::new(0.4, 0.8),
            fov_deg: 60.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LightRangeConfig {
    pub yaw_range: RangeF32,
    pub pitch_range: RangeF32,
    pub intensity_range: RangeF32,
    pub ambient_range: RangeF32,
}

impl Default for LightRangeConfig {
    fn default() -> Self {
        Self {
            yaw_range: RangeF32::new(-180.0, 180.0),
            pitch_range: RangeF32::new(10.0, 80.0),
            intensity_range: RangeF32::new(0.6, 1.4),
            ambient_range: RangeF32::new(0.05, 0.25),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MorphConfig {
    /// How many blend shape targets to activate simultaneously
    pub max_active_targets: usize,
    /// Per-active-target weight range
    pub weight_range: RangeF32,
}

impl Default for MorphConfig {
    fn default() -> Self {
        Self {
            max_active_targets: 4,
            weight_range: RangeF32::new(0.0, 0.7),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneratorConfig {
    pub output_dir: String,
    pub num_samples: usize,
    pub image_width: u32,
    pub image_height: u32,
    /// Path to ICT-FaceKit .glb (or any GLTF with morph targets)
    pub face_model_path: String,
    /// Number of textures to pre-load into GPU from res/textures/ at startup
    pub texture_pool_size: usize,
    /// Open a preview window while generating
    pub preview_window: bool,
    /// RNG seed (0 = use OS entropy)
    pub seed: u64,

    pub camera: CameraConfig,
    pub light: LightRangeConfig,
    pub morph: MorphConfig,
    pub degradation: DegradationConfig,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            output_dir: "output".to_string(),
            num_samples: 1000,
            image_width: 640,
            image_height: 480,
            face_model_path: "res/face/face.glb".to_string(),
            texture_pool_size: 50,
            preview_window: false,
            seed: 0,
            camera: CameraConfig::default(),
            light: LightRangeConfig::default(),
            morph: MorphConfig::default(),
            degradation: DegradationConfig::default(),
        }
    }
}

pub fn load_config(path: &std::path::Path) -> anyhow::Result<GeneratorConfig> {
    let text = std::fs::read_to_string(path)?;
    Ok(toml::from_str(&text)?)
}
