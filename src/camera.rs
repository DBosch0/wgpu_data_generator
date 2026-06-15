use cgmath::{InnerSpace, Matrix4, Point3, Rad, SquareMatrix, Vector3, perspective};

pub struct Camera {
    pub position: Point3<f32>,
    yaw: Rad<f32>,
    pitch: Rad<f32>,
    roll: Rad<f32>,
    pub target: Point3<f32>,
}

impl Camera {
    pub fn new<V: Into<Point3<f32>>, Y: Into<Rad<f32>>, P: Into<Rad<f32>>>(
        position: V,
        yaw: Y,
        pitch: P,
    ) -> Self {
        Self {
            position: position.into(),
            yaw: yaw.into(),
            pitch: pitch.into(),
            roll: Rad(0.0),
            target: Point3::new(0.0, 0.0, 0.0),
        }
    }

    /// Position the camera at `distance` from `target`, looking at it from the
    /// given spherical angles (all in radians).
    pub fn set_pose(
        &mut self,
        yaw: Rad<f32>,
        pitch: Rad<f32>,
        roll: Rad<f32>,
        distance: f32,
        target: Point3<f32>,
    ) {
        let (sin_yaw, cos_yaw) = yaw.0.sin_cos();
        let (sin_pitch, cos_pitch) = pitch.0.sin_cos();
        let offset = Vector3::new(
            cos_pitch * sin_yaw,
            sin_pitch,
            cos_pitch * cos_yaw,
        ) * distance;
        self.position = target + offset;
        self.yaw = yaw;
        self.pitch = pitch;
        self.roll = roll;
        self.target = target;
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        // Orbit camera: always look toward self.target.
        // Roll tilts the up vector around the look direction.
        let forward = (cgmath::Vector3::new(
            self.target.x - self.position.x,
            self.target.y - self.position.y,
            self.target.z - self.position.z,
        ))
        .normalize();
        let world_up = Vector3::unit_y();
        let right = forward.cross(world_up).normalize();
        let (sin_roll, cos_roll) = self.roll.0.sin_cos();
        let up = world_up * cos_roll + right * sin_roll;

        Matrix4::look_to_rh(self.position, forward, up)
    }
}

pub struct Projection {
    aspect: f32,
    fovy: Rad<f32>,
    znear: f32,
    zfar: f32,
}

impl Projection {
    pub fn new<F: Into<Rad<f32>>>(width: u32, height: u32, fovy: F, znear: f32, zfar: f32) -> Self {
        Self {
            aspect: width as f32 / height as f32,
            fovy: fovy.into(),
            znear,
            zfar,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        perspective(self.fovy, self.aspect, self.znear, self.zfar)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_position: [f32; 4],
    view: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_position: [0.0; 4],
            view: cgmath::Matrix4::identity().into(),
            view_proj: cgmath::Matrix4::identity().into(),
            inv_proj: cgmath::Matrix4::identity().into(),
            inv_view: cgmath::Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        self.view_position = camera.position.to_homogeneous().into();
        let proj = projection.calc_matrix();
        let view = camera.calc_matrix();
        let view_proj = proj * view;
        self.view = view.into();
        self.view_proj = view_proj.into();
        self.inv_proj = proj.invert().unwrap().into();
        self.inv_view = view.invert().unwrap().into();
    }
}
