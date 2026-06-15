use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    pub direction: [f32; 4], // world-space direction toward light, w=padding
    pub color: [f32; 4],     // RGB color, w=intensity
    pub ambient: [f32; 4],   // RGB ambient color, w=ambient strength
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LightConfig {
    /// Horizontal angle around Y axis in degrees
    pub yaw: f32,
    /// Vertical angle from horizon in degrees (positive = above)
    pub pitch: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub ambient_strength: f32,
}

impl Default for LightConfig {
    fn default() -> Self {
        Self {
            yaw: 45.0,
            pitch: 45.0,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            ambient_strength: 0.1,
        }
    }
}

impl LightConfig {
    pub fn to_uniform(&self) -> LightUniform {
        let yaw = self.yaw.to_radians();
        let pitch = self.pitch.to_radians();
        let (sin_yaw, cos_yaw) = yaw.sin_cos();
        let (sin_pitch, cos_pitch) = pitch.sin_cos();

        // Unit vector pointing from scene toward the light source
        let dir = [
            cos_pitch * sin_yaw,
            sin_pitch,
            cos_pitch * cos_yaw,
            0.0,
        ];

        LightUniform {
            direction: dir,
            color: [
                self.color[0],
                self.color[1],
                self.color[2],
                self.intensity,
            ],
            ambient: [
                self.color[0] * self.ambient_strength,
                self.color[1] * self.ambient_strength,
                self.color[2] * self.ambient_strength,
                self.ambient_strength,
            ],
        }
    }
}

pub fn create_light_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("light_bind_group_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

pub fn create_light_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("light_bind_group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    })
}
