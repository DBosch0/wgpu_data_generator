use crate::texture::Texture;

pub const RENDER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub struct OffscreenTarget {
    pub color_texture: Texture,
    pub depth_texture: Texture,
    readback_buffer: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    bytes_per_row: u32,
}

impl OffscreenTarget {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let color_texture =
            Texture::create_render_target(device, width, height, RENDER_FORMAT, "offscreen_color");
        let depth_texture = Texture::create_depth_texture_sized(device, width, height);
        let (readback_buffer, bytes_per_row) =
            Texture::create_readback_buffer(device, width, height);

        Self {
            color_texture,
            depth_texture,
            readback_buffer,
            width,
            height,
            bytes_per_row,
        }
    }

    /// Encodes a texture→buffer copy into `encoder`, submits, polls until done,
    /// then maps and returns tight (unpadded) RGBA bytes.
    pub fn readback_rgba(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: wgpu::CommandEncoder,
    ) -> Vec<u8> {
        // Copy color texture into the readback buffer (padded rows)
        let mut enc = encoder;
        enc.copy_texture_to_buffer(
            self.color_texture.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            self.color_texture.size,
        );
        queue.submit(std::iter::once(enc.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll failed");

        // Map and strip row padding
        let slice = self.readback_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll failed");

        let row_bytes = (self.width * 4) as usize;
        let padded = self.bytes_per_row as usize;
        let mapped = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity(row_bytes * self.height as usize);
        for row in 0..self.height as usize {
            pixels.extend_from_slice(&mapped[row * padded..row * padded + row_bytes]);
        }
        drop(mapped);
        self.readback_buffer.unmap();

        pixels
    }
}
