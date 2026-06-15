use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;
use cgmath::InnerSpace;
use wgpu::util::DeviceExt;

use crate::texture;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FaceMeshVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
}

impl FaceMeshVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 5] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3, 3 => Float32x3, 4 => Float32x3];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: core::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub struct MorphTarget {
    pub name: String,
    /// Per raw-OBJ-vertex position deltas (indexed same as FaceMesh::base_positions).
    pub position_deltas: Vec<[f32; 3]>,
}

pub struct FaceMesh {
    /// Raw OBJ positions (26,719 for ICT-FaceKit).
    pub base_positions: Vec<[f32; 3]>,
    /// Per unified GPU vertex: index into base_positions.
    pub vertex_pos_indices: Vec<u32>,
    /// Per unified GPU vertex: UV coordinate (Y-flipped for wgpu).
    pub tex_coords: Vec<[f32; 2]>,
    /// Triangle indices into the unified vertex space.
    pub indices: Vec<u32>,
    pub morph_targets: Vec<MorphTarget>,
    pub morph_weights: Vec<f32>,
    /// Raw OBJ position indices for the 68 dlib 300-W landmarks (from vertex_indices.json).
    pub landmark_indices: Vec<usize>,
}

pub struct GpuFaceMaterial {
    pub diffuse_texture: texture::Texture,
    pub normal_texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
}

pub struct GpuFaceMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
    pub material: GpuFaceMaterial,
}

// ─── OBJ parsing ──────────────────────────────────────────────────────────────

struct ObjData {
    raw_pos: Vec<[f32; 3]>,
    raw_uv: Vec<[f32; 2]>,
    /// Flat list of position indices per triangle vertex (after fan-triangulation).
    face_pos: Vec<u32>,
    /// Flat list of UV indices per triangle vertex.
    face_uv: Vec<u32>,
}

/// Parse positions, UVs, and face connectivity from an OBJ file.
/// Ignores normals, materials, groups, and objects — uses global vertex indexing.
fn parse_obj(path: &Path) -> anyhow::Result<ObjData> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read {}", path.display()))?;

    let mut raw_pos: Vec<[f32; 3]> = Vec::new();
    let mut raw_uv: Vec<[f32; 2]> = Vec::new();
    let mut face_pos: Vec<u32> = Vec::new();
    let mut face_uv: Vec<u32> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("v ") {
            let mut it = rest.split_ascii_whitespace();
            let x: f32 = it.next().context("v: missing x")?.parse()?;
            let y: f32 = it.next().context("v: missing y")?.parse()?;
            let z: f32 = it.next().context("v: missing z")?.parse()?;
            raw_pos.push([x, y, z]);
        } else if let Some(rest) = line.strip_prefix("vt ") {
            let mut it = rest.split_ascii_whitespace();
            let u: f32 = it.next().context("vt: missing u")?.parse()?;
            let v: f32 = it.next().context("vt: missing v")?.parse()?;
            raw_uv.push([u, 1.0 - v]); // Y-flip for wgpu
        } else if let Some(rest) = line.strip_prefix("f ") {
            // Each token is pos[/uv[/normal]] (1-indexed). Fan-triangulate polygons.
            let verts: Vec<(u32, u32)> = rest
                .split_ascii_whitespace()
                .map(|tok| {
                    let mut parts = tok.split('/');
                    let pi: u32 = parts.next().and_then(|s| s.parse::<u32>().ok())
                        .map(|n| n - 1)
                        .unwrap_or(0);
                    let ui: u32 = parts.next()
                        .filter(|s| !s.is_empty())
                        .and_then(|s| s.parse::<u32>().ok())
                        .map(|n| n - 1)
                        .unwrap_or(pi);
                    (pi, ui)
                })
                .collect();
            // Fan triangulation: (0,1,2), (0,2,3), (0,3,4), ...
            for i in 1..verts.len().saturating_sub(1) {
                face_pos.push(verts[0].0);
                face_uv.push(verts[0].1);
                face_pos.push(verts[i].0);
                face_uv.push(verts[i].1);
                face_pos.push(verts[i + 1].0);
                face_uv.push(verts[i + 1].1);
            }
        }
    }

    Ok(ObjData { raw_pos, raw_uv, face_pos, face_uv })
}

/// Read only the `v` lines from an OBJ (fast path for morph-target expression OBJs).
fn load_raw_positions(path: &Path) -> anyhow::Result<Vec<[f32; 3]>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read {}", path.display()))?;
    let mut positions = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.trim().strip_prefix("v ") {
            let mut it = rest.split_ascii_whitespace();
            let x: f32 = it.next().context("v: x")?.parse()?;
            let y: f32 = it.next().context("v: y")?.parse()?;
            let z: f32 = it.next().context("v: z")?.parse()?;
            positions.push([x, y, z]);
        }
    }
    Ok(positions)
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Load ICT-FaceKit (OBJ format) and all 53 FACS expression morph targets.
///
/// `neutral_path` — path to `generic_neutral_mesh.obj`.
/// Expression OBJs and `vertex_indices.json` are expected in the same directory.
pub fn load_obj_face(
    neutral_path: &Path,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<(FaceMesh, GpuFaceMesh)> {
    let obj = parse_obj(neutral_path)
        .with_context(|| format!("Failed to parse {}", neutral_path.display()))?;

    // Build a unified vertex array: each unique (pos_idx, uv_idx) pair → one GPU vertex.
    let mut vertex_pos_indices: Vec<u32> = Vec::new();
    let mut tex_coords: Vec<[f32; 2]> = Vec::new();
    let mut unified_indices: Vec<u32> = Vec::new();
    let mut index_map: HashMap<(u32, u32), u32> = HashMap::new();

    for (&pi, &ui) in obj.face_pos.iter().zip(obj.face_uv.iter()) {
        let key = (pi, ui);
        let unified_idx = *index_map.entry(key).or_insert_with(|| {
            let new_idx = vertex_pos_indices.len() as u32;
            vertex_pos_indices.push(pi);
            tex_coords.push(obj.raw_uv[ui as usize]);
            new_idx
        });
        unified_indices.push(unified_idx);
    }

    let num_raw = obj.raw_pos.len();
    let num_unified = vertex_pos_indices.len();
    let face_dir = neutral_path.parent().unwrap_or(Path::new("."));

    // Load landmark indices and expression list from vertex_indices.json.
    let json_path = face_dir.join("vertex_indices.json");
    let (landmark_indices, expressions) = if json_path.exists() {
        let text = std::fs::read_to_string(&json_path)?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let lm: Vec<usize> = v["idx_to_landmark_verts"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
            .unwrap_or_default();
        let exprs: Vec<String> = v["expressions"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default();
        (lm, exprs)
    } else {
        log::warn!("vertex_indices.json not found — landmark indices and expressions will be empty");
        (Vec::new(), Vec::new())
    };

    // Load expression morph targets.
    let mut morph_targets: Vec<MorphTarget> = Vec::with_capacity(expressions.len());
    for expr_name in &expressions {
        let expr_path = face_dir.join(format!("{}.obj", expr_name));
        if !expr_path.exists() {
            log::warn!("Expression OBJ not found, skipping: {}", expr_path.display());
            continue;
        }
        let expr_pos = load_raw_positions(&expr_path)
            .with_context(|| format!("Failed to load {}", expr_path.display()))?;

        if expr_pos.len() != num_raw {
            log::warn!(
                "Expression {} has {} verts but neutral has {} — skipping",
                expr_name, expr_pos.len(), num_raw
            );
            continue;
        }

        let deltas: Vec<[f32; 3]> = obj.raw_pos.iter().zip(expr_pos.iter())
            .map(|(b, e)| [e[0] - b[0], e[1] - b[1], e[2] - b[2]])
            .collect();

        morph_targets.push(MorphTarget { name: expr_name.clone(), position_deltas: deltas });
    }

    log::info!(
        "Loaded face model: {} raw positions, {} unified vertices, {} triangles, {} morph targets, {} landmarks",
        num_raw, num_unified, unified_indices.len() / 3,
        morph_targets.len(), landmark_indices.len()
    );

    let num_morph_targets = morph_targets.len();
    let face_mesh = FaceMesh {
        base_positions: obj.raw_pos,
        vertex_pos_indices,
        tex_coords,
        indices: unified_indices.clone(),
        morph_targets,
        morph_weights: vec![0.0; num_morph_targets],
        landmark_indices,
    };

    let blended = blend_vertices(&face_mesh);

    // Fallback material: skin-tone solid color diffuse, flat normal map.
    let diffuse_texture =
        texture::Texture::solid_color(device, queue, [220, 180, 150, 255], "diffuse_fallback");
    let normal_texture =
        texture::Texture::solid_color(device, queue, [128, 128, 255, 255], "normal_fallback");

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("face_material_bind_group"),
        layout: texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&normal_texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&normal_texture.sampler),
            },
        ],
    });

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("face_vertex_buffer"),
        contents: bytemuck::cast_slice(&blended),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });
    let num_elements = unified_indices.len() as u32;
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("face_index_buffer"),
        contents: bytemuck::cast_slice(&unified_indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let gpu_mesh = GpuFaceMesh {
        vertex_buffer,
        index_buffer,
        num_elements,
        material: GpuFaceMaterial { diffuse_texture, normal_texture, bind_group },
    };

    Ok((face_mesh, gpu_mesh))
}

/// Blend raw OBJ positions according to current morph weights. Fast path used for
/// landmark projection (no tangent computation).
pub fn blend_positions(face: &FaceMesh) -> Vec<[f32; 3]> {
    let mut positions = face.base_positions.clone();
    for (target, &w) in face.morph_targets.iter().zip(face.morph_weights.iter()) {
        if w == 0.0 {
            continue;
        }
        for (pos, d) in positions.iter_mut().zip(target.position_deltas.iter()) {
            pos[0] += w * d[0];
            pos[1] += w * d[1];
            pos[2] += w * d[2];
        }
    }
    positions
}

/// Full blend: positions + recomputed normals/tangents/bitangents. Used for GPU upload.
pub fn blend_vertices(face: &FaceMesh) -> Vec<FaceMeshVertex> {
    let blended_raw = blend_positions(face);

    let n_unified = face.vertex_pos_indices.len();
    let mut vertices: Vec<FaceMeshVertex> = (0..n_unified)
        .map(|i| FaceMeshVertex {
            position: blended_raw[face.vertex_pos_indices[i] as usize],
            tex_coords: face.tex_coords[i],
            normal: [0.0; 3],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        })
        .collect();

    let mut tri_counts = vec![0u32; n_unified];

    for tri in face.indices.chunks(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);

        let p0: cgmath::Vector3<f32> = vertices[i0].position.into();
        let p1: cgmath::Vector3<f32> = vertices[i1].position.into();
        let p2: cgmath::Vector3<f32> = vertices[i2].position.into();
        let dp1 = p1 - p0;
        let dp2 = p2 - p0;

        let face_normal = dp1.cross(dp2);
        let len = face_normal.magnitude();
        if len < 1e-10 {
            continue;
        }
        let face_normal = face_normal / len;

        let uv0: cgmath::Vector2<f32> = vertices[i0].tex_coords.into();
        let uv1: cgmath::Vector2<f32> = vertices[i1].tex_coords.into();
        let uv2: cgmath::Vector2<f32> = vertices[i2].tex_coords.into();
        let duv1 = uv1 - uv0;
        let duv2 = uv2 - uv0;
        let denom = duv1.x * duv2.y - duv1.y * duv2.x;

        let (tangent, bitangent) = if denom.abs() > 1e-8 {
            let r = 1.0 / denom;
            (
                (dp1 * duv2.y - dp2 * duv1.y) * r,
                (dp2 * duv1.x - dp1 * duv2.x) * -r,
            )
        } else {
            (cgmath::Vector3::unit_x(), cgmath::Vector3::unit_y())
        };

        for &idx in &[i0, i1, i2] {
            let v = &mut vertices[idx];
            let n: cgmath::Vector3<f32> = v.normal.into();
            v.normal = (n + face_normal).into();
            let t: cgmath::Vector3<f32> = v.tangent.into();
            v.tangent = (t + tangent).into();
            let b: cgmath::Vector3<f32> = v.bitangent.into();
            v.bitangent = (b + bitangent).into();
            tri_counts[idx] += 1;
        }
    }

    for (i, &count) in tri_counts.iter().enumerate() {
        if count > 0 {
            let inv = 1.0 / count as f32;
            let n: cgmath::Vector3<f32> = vertices[i].normal.into();
            let t: cgmath::Vector3<f32> = vertices[i].tangent.into();
            let b: cgmath::Vector3<f32> = vertices[i].bitangent.into();
            vertices[i].normal = n.normalize().into();
            vertices[i].tangent = (t * inv).into();
            vertices[i].bitangent = (b * inv).into();
        }
    }

    vertices
}

pub fn upload_blended(face: &FaceMesh, gpu: &GpuFaceMesh, queue: &wgpu::Queue) {
    let blended = blend_vertices(face);
    queue.write_buffer(&gpu.vertex_buffer, 0, bytemuck::cast_slice(&blended));
}
