use std::path::Path;

use anyhow::Context;
use cgmath::Deg;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::Serialize;
use wgpu::util::DeviceExt;

use crate::{
    camera::{Camera, CameraUniform, Projection},
    config::GeneratorConfig,
    degradation::apply_degradation,
    gltf_model::{self, FaceMesh, GpuFaceMesh},
    landmark::{self, LandmarkDefinition, LandmarkPoint},
    lighting::{self, LightConfig, LightUniform},
    offscreen::{self, OffscreenTarget},
    texture::Texture,
};

#[derive(Serialize)]
pub struct SampleLabel {
    pub image_file: String,
    pub landmarks: Vec<LandmarkPoint>,
    pub camera_yaw_deg: f32,
    pub camera_pitch_deg: f32,
    pub camera_roll_deg: f32,
    pub camera_distance: f32,
    pub morph_weights: Vec<f32>,
    pub light_yaw_deg: f32,
    pub light_pitch_deg: f32,
    pub light_intensity: f32,
    pub light_ambient: f32,
}

struct PoolEntry {
    _diffuse: Texture,
    _normal: Texture,
    bind_group: wgpu::BindGroup,
}

pub struct Generator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    target: OffscreenTarget,
    face_mesh: FaceMesh,
    gpu_mesh: GpuFaceMesh,
    render_pipeline: wgpu::RenderPipeline,
    camera: Camera,
    projection: Projection,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    landmark_def: LandmarkDefinition,
    texture_pool: Vec<PoolEntry>,
    config: GeneratorConfig,
}

impl Generator {
    pub fn new(config: GeneratorConfig) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .context("No GPU adapter found")?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("generator_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
        ))?;

        let w = config.image_width;
        let h = config.image_height;

        let target = OffscreenTarget::new(&device, w, h);

        // Bind group layouts
        let texture_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("texture_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let camera_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let light_layout = lighting::create_light_bind_group_layout(&device);

        // Camera
        let camera = Camera::new((0.0, 0.0, 0.6), Deg(0.0), Deg(0.0));
        let projection = Projection::new(w, h, Deg(config.camera.fov_deg), 0.5, 200.0);
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Light
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("light_buffer"),
            contents: bytemuck::cast_slice(&[LightConfig::default().to_uniform()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let light_bind_group =
            lighting::create_light_bind_group(&device, &light_layout, &light_buffer);

        // Render pipeline
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("face_pipeline_layout"),
                bind_group_layouts: &[Some(&texture_layout), Some(&camera_layout), Some(&light_layout)],
                immediate_size: 0,
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("face_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/face.wgsl").into()),
        });

        let render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("face_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[gltf_model::FaceMeshVertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: offscreen::RENDER_FORMAT,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Texture::DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        // Load face model
        let model_path = Path::new(&config.face_model_path);
        let (face_mesh, gpu_mesh) =
            gltf_model::load_obj_face(model_path, &device, &queue, &texture_layout)?;

        let landmark_def = LandmarkDefinition {
            indices: face_mesh.landmark_indices.clone(),
            names: None,
        };

        // Texture pool (FFHQ-UV textures if available)
        let texture_pool =
            load_texture_pool(&device, &queue, &texture_layout, &config, &face_mesh)?;

        Ok(Self {
            device,
            queue,
            target,
            face_mesh,
            gpu_mesh,
            render_pipeline,
            camera,
            projection,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            light_buffer,
            light_bind_group,
            landmark_def,
            texture_pool,
            config,
        })
    }

    /// Render one sample. Returns (rgba_pixels, label).
    pub fn generate_sample(
        &mut self,
        rng: &mut impl Rng,
        image_file: String,
    ) -> anyhow::Result<(Vec<u8>, SampleLabel)> {
        let cfg = &self.config.clone();

        // 1. Randomize morph weights
        let num_targets = self.face_mesh.morph_targets.len();
        for w in self.face_mesh.morph_weights.iter_mut() {
            *w = 0.0;
        }
        if num_targets > 0 {
            let active = cfg.morph.max_active_targets.min(num_targets);
            let mut indices: Vec<usize> = (0..num_targets).collect();
            for i in 0..active {
                let j = rng.random_range(i..num_targets);
                indices.swap(i, j);
            }
            for &idx in &indices[..active] {
                self.face_mesh.morph_weights[idx] = cfg.morph.weight_range.sample(rng);
            }
        }
        gltf_model::upload_blended(&self.face_mesh, &self.gpu_mesh, &self.queue);
        let blended_positions = gltf_model::blend_positions(&self.face_mesh);

        // 2. Camera
        let yaw_deg = cfg.camera.yaw_range.sample(rng);
        let pitch_deg = cfg.camera.pitch_range.sample(rng);
        let roll_deg = cfg.camera.roll_range.sample(rng);
        let distance = cfg.camera.distance_range.sample(rng);
        self.camera.set_pose(
            cgmath::Deg(yaw_deg).into(),
            cgmath::Deg(pitch_deg).into(),
            cgmath::Deg(roll_deg).into(),
            distance,
            cgmath::Point3::new(0.0, 0.0, 9.5), // centroid of ICT-FaceKit landmarks
        );
        self.camera_uniform.update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));

        // 3. Light
        let light_yaw = cfg.light.yaw_range.sample(rng);
        let light_pitch = cfg.light.pitch_range.sample(rng);
        let light_intensity = cfg.light.intensity_range.sample(rng);
        let light_ambient = cfg.light.ambient_range.sample(rng);
        let light_uniform: LightUniform = LightConfig {
            yaw: light_yaw,
            pitch: light_pitch,
            color: [1.0, 0.98, 0.95],
            intensity: light_intensity,
            ambient_strength: light_ambient,
        }
        .to_uniform();
        self.queue.write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&[light_uniform]));

        // 4. Background color (random solid)
        let bg = wgpu::Color {
            r: rng.random::<f64>(),
            g: rng.random::<f64>(),
            b: rng.random::<f64>(),
            a: 1.0,
        };

        // 5. Texture from pool (or model's built-in)
        let bind_group_ref: &wgpu::BindGroup = if !self.texture_pool.is_empty() {
            let idx = rng.random_range(0..self.texture_pool.len());
            &self.texture_pool[idx].bind_group
        } else {
            &self.gpu_mesh.material.bind_group
        };

        // 6. Render
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame_encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("face_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.target.color_texture.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(bg),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.target.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(1, &self.camera_bind_group, &[]);
            pass.set_bind_group(2, &self.light_bind_group, &[]);
            pass.set_vertex_buffer(0, self.gpu_mesh.vertex_buffer.slice(..));

            pass.set_bind_group(0, bind_group_ref, &[]);
            pass.set_index_buffer(self.gpu_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.gpu_mesh.num_elements, 0, 0..1);
        }

        // 7. Readback + degrade
        let rgba = self.target.readback_rgba(&self.device, &self.queue, encoder);
        let rgba = apply_degradation(rgba, cfg.image_width, cfg.image_height, &cfg.degradation, rng);

        // 8. Landmark projection
        let view_proj: cgmath::Matrix4<f32> = self.camera_uniform.view_proj.into();
        let landmarks =
            landmark::project_landmarks(&blended_positions, &self.landmark_def, view_proj, cfg.image_width, cfg.image_height);

        let label = SampleLabel {
            image_file,
            landmarks,
            camera_yaw_deg: yaw_deg,
            camera_pitch_deg: pitch_deg,
            camera_roll_deg: roll_deg,
            camera_distance: distance,
            morph_weights: self.face_mesh.morph_weights.clone(),
            light_yaw_deg: light_yaw,
            light_pitch_deg: light_pitch,
            light_intensity,
            light_ambient,
        };

        Ok((rgba, label))
    }

    pub fn run(&mut self) -> anyhow::Result<()> {
        let cfg = self.config.clone();

        let out = Path::new(&cfg.output_dir);
        std::fs::create_dir_all(out.join("images"))?;
        std::fs::create_dir_all(out.join("labels"))?;

        let mut rng: StdRng = if cfg.seed == 0 {
            StdRng::from_os_rng()
        } else {
            StdRng::seed_from_u64(cfg.seed)
        };

        let mut dataset_index = Vec::with_capacity(cfg.num_samples);

        for i in 0..cfg.num_samples {
            let image_name = format!("{:06}.png", i);
            let label_name = format!("{:06}.json", i);

            let (rgba, label) =
                self.generate_sample(&mut rng, format!("images/{}", image_name))?;

            // Save image
            image::RgbaImage::from_raw(cfg.image_width, cfg.image_height, rgba)
                .context("RgbaImage creation failed")?
                .save(out.join("images").join(&image_name))?;

            // Save label
            std::fs::write(
                out.join("labels").join(&label_name),
                serde_json::to_string_pretty(&label)?,
            )?;

            dataset_index.push(serde_json::json!({
                "image": format!("images/{}", image_name),
                "label": format!("labels/{}", label_name),
            }));

            if i % 100 == 0 || i == cfg.num_samples - 1 {
                log::info!("Generated {}/{}", i + 1, cfg.num_samples);
            }
        }

        std::fs::write(out.join("dataset.json"), serde_json::to_string_pretty(&dataset_index)?)?;
        std::fs::write(out.join("config_used.toml"), toml::to_string_pretty(&cfg)?)?;

        log::info!("Done → {}", out.display());
        Ok(())
    }
}

fn load_texture_pool(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    config: &GeneratorConfig,
    _face_mesh: &FaceMesh,
) -> anyhow::Result<Vec<PoolEntry>> {
    use std::ffi::OsStr;

    let texture_dir = Path::new("res/textures");
    if !texture_dir.exists() {
        log::info!("res/textures/ not found — using model's built-in texture only");
        return Ok(Vec::new());
    }

    let mut paths: Vec<_> = std::fs::read_dir(texture_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| matches!(
            p.extension().and_then(OsStr::to_str),
            Some("png") | Some("jpg") | Some("jpeg")
        ))
        .collect();

    if paths.is_empty() {
        return Ok(Vec::new());
    }

    paths.shuffle(&mut rand::rng());
    paths.truncate(config.texture_pool_size);

    let mut pool = Vec::with_capacity(paths.len());
    for path in &paths {
        let img = image::open(path)
            .with_context(|| format!("Failed to load {}", path.display()))?;
        let diffuse = Texture::from_image(device, queue, &img, Some("pool_diffuse"), false)?;
        // Flat normal (blue) per entry — 1×1 so trivially cheap, avoids lifetime issues
        let normal = Texture::solid_color(device, queue, [128, 128, 255, 255], "pool_normal");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&normal.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&normal.sampler),
                },
            ],
        });

        pool.push(PoolEntry { _diffuse: diffuse, _normal: normal, bind_group });
    }

    log::info!("Loaded {} textures into pool", pool.len());
    Ok(pool)
}
