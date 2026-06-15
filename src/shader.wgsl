struct CameraUniform{
    view_pos: vec4<f32>,
    view: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> camera: CameraUniform;

struct VertexInput{
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
};

struct VertexOutput{
    @builtin(position) clip_position: vec4<f32>,  
    @location(0) tex_coords: vec2<f32>,
//   @location(1) world_position: vec3<f32>,
//   @location(2) world_view_position: vec3<f32>,
//   @location(3) world_light_position: vec3<f32>,
//   @location(4) world_normal: vec3<f32>,
//   @location(5) world_tangent: vec3<f32>,
//   @location(6) world_bitangent: vec3<f32>,
};

@vertex
fn vs_main(model:VertexInput) -> VertexOutput{
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.tex_coords = model.tex_coords;
    return out;
}

@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;
@group(0) @binding(2) var t_normal: texture_2d<f32>;
@group(0) @binding(3) var s_normal: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32>{
    let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let object_normal: vec4<f32> = textureSample(t_normal, s_normal, in.tex_coords);

    return object_color;
}