use serde::Serialize;

/// 68 vertex indices into the ICT-FaceKit mesh following the dlib/iBug 300-W convention.
///
/// Groups (0-indexed):
///   0-16  : jaw line (17 points)
///   17-21 : left eyebrow (5)
///   22-26 : right eyebrow (5)
///   27-35 : nose bridge + tip (9)
///   36-41 : left eye (6)
///   42-47 : right eye (6)
///   48-59 : outer mouth (12)
///   60-67 : inner mouth (8)
///
/// Source: ICT-FaceKit `FaceXModel/vertex_indices.json` → `idx_to_landmark_verts`.
/// These are 0-based indices into the raw OBJ vertex positions of `generic_neutral_mesh.obj`.
/// At runtime, landmark indices are loaded directly from vertex_indices.json by
/// `load_obj_face()` — this constant serves as documentation only.
pub const LANDMARK_INDICES_300W: [usize; 68] = [
    // jaw (0–16)
    1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966,
    3661, 4390, 3927, 3924, 2608, 3272, 4088, 3443,
    // left eyebrow (17–21)
    268, 493, 1914, 2044, 1401,
    // right eyebrow (22–26)
    3615, 4240, 4114, 2734, 2509,
    // nose (27–35)
    978, 4527, 4942, 4857, 1140, 2075, 1147, 4269, 3360,
    // left eye (36–41)
    1507, 1542, 1537, 1528, 1518, 1511,
    // right eye (42–47)
    3742, 3751, 3756, 3721, 3725, 3732,
    // outer mouth (48–59)
    5708, 5695, 2081, 0, 4275, 6200, 6213, 6346, 6461, 5518, 5957, 5841,
    // inner mouth (60–67)
    5702, 5711, 5533, 6216, 6207, 6470, 5517, 5966,
];

pub struct LandmarkDefinition {
    pub indices: Vec<usize>,
    pub names: Option<Vec<String>>,
}

impl LandmarkDefinition {
    pub fn dlib_300w() -> Self {
        Self {
            indices: LANDMARK_INDICES_300W.to_vec(),
            names: None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct LandmarkPoint {
    /// X pixel coordinate (origin top-left)
    pub x: f32,
    /// Y pixel coordinate (origin top-left)
    pub y: f32,
    /// NDC depth in [0,1] — useful for 3D training targets
    pub depth: f32,
    /// False if the point is behind the camera or outside the view frustum
    pub visible: bool,
}

/// Project 3D landmark positions through the view-projection matrix to 2D pixel coords.
///
/// `blended_positions` must contain the already-blended vertex positions (output of
/// `blend_vertices`). Each returned `LandmarkPoint` is in pixel space with (0,0) at
/// top-left, matching image coordinate conventions.
pub fn project_landmarks(
    blended_positions: &[[f32; 3]],
    definition: &LandmarkDefinition,
    view_proj: cgmath::Matrix4<f32>,
    width: u32,
    height: u32,
) -> Vec<LandmarkPoint> {
    definition
        .indices
        .iter()
        .map(|&idx| {
            let pos = blended_positions[idx];
            let clip = view_proj * cgmath::vec4(pos[0], pos[1], pos[2], 1.0);
            let w = clip.w;
            if w.abs() < 1e-6 {
                return LandmarkPoint { x: 0.0, y: 0.0, depth: 0.0, visible: false };
            }
            let ndc_x = clip.x / w;
            let ndc_y = clip.y / w;
            let depth = clip.z / w;
            // Convert NDC [-1,1] to pixel coords; Y is flipped (NDC +Y = image top)
            let px = (ndc_x + 1.0) * 0.5 * width as f32;
            let py = (1.0 - ndc_y) * 0.5 * height as f32;
            let visible = depth >= 0.0 && depth <= 1.0 && ndc_x.abs() <= 1.0 && ndc_y.abs() <= 1.0;
            LandmarkPoint { x: px, y: py, depth, visible }
        })
        .collect()
}
