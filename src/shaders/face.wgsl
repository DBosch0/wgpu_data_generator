// Bind group 0: material textures
@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;
@group(0) @binding(2) var t_normal: texture_2d<f32>;
@group(0) @binding(3) var s_normal: sampler;

// Bind group 1: camera
struct CameraUniform {
    view_pos:  vec4<f32>,
    view:      mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_proj:  mat4x4<f32>,
    inv_view:  mat4x4<f32>,
}
@group(1) @binding(0) var<uniform> camera: CameraUniform;

// Bind group 2: light
struct LightUniform {
    direction: vec4<f32>, // world-space direction toward light, w=padding
    color:     vec4<f32>, // RGB + intensity in w
    ambient:   vec4<f32>, // RGB ambient + strength in w
}
@group(2) @binding(0) var<uniform> light: LightUniform;

// Vertex input — matches FaceMeshVertex layout (60 bytes)
struct VertexInput {
    @location(0) position:   vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal:     vec3<f32>,
    @location(3) tangent:    vec3<f32>,
    @location(4) bitangent:  vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords:     vec2<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) world_normal:   vec3<f32>,
    @location(3) world_tangent:  vec3<f32>,
    @location(4) world_bitangent: vec3<f32>,
}

@vertex
fn vs_main(v: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Face is at world origin — no model matrix needed
    out.clip_position  = camera.view_proj * vec4<f32>(v.position, 1.0);
    out.world_position = v.position;
    out.tex_coords     = v.tex_coords;
    out.world_normal   = v.normal;
    out.world_tangent  = v.tangent;
    out.world_bitangent = v.bitangent;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample diffuse and normal map
    let albedo      = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let normal_map  = textureSample(t_normal,  s_normal,  in.tex_coords).xyz;

    // Decode normal map from [0,1] → [-1,1]
    let tbn_normal  = normalize(normal_map * 2.0 - 1.0);

    // Build TBN matrix and transform to world space
    let T = normalize(in.world_tangent);
    let B = normalize(in.world_bitangent);
    let N = normalize(in.world_normal);
    let tbn = mat3x3<f32>(T, B, N);
    let world_normal = normalize(tbn * tbn_normal);

    // Light direction (toward light, normalized)
    let light_dir = normalize(light.direction.xyz);

    // Diffuse
    let diff = max(dot(world_normal, light_dir), 0.0);
    let diffuse = light.color.rgb * light.color.w * diff;

    // Specular (Blinn-Phong)
    let view_dir  = normalize(camera.view_pos.xyz - in.world_position);
    let half_dir  = normalize(light_dir + view_dir);
    let spec      = pow(max(dot(world_normal, half_dir), 0.0), 32.0);
    let specular  = light.color.rgb * light.color.w * spec * 0.3;

    // Ambient
    let ambient = light.ambient.rgb;

    let final_color = (ambient + diffuse + specular) * albedo.rgb;
    return vec4<f32>(final_color, albedo.a);
}
