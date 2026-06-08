#![allow(dead_code)]

use std::sync::Arc;

use winit::{application::ApplicationHandler, window::Window};

use crate::{GpuContext, SurfaceWrapper};

pub(crate) struct Application<'a> {
    title: &'static str,
    window: Option<Arc<Window>>,
    surface: Option<SurfaceWrapper<'a>>,
    context: GpuContext,
}

impl Application<'_> {
    pub(crate) fn new(title: &'static str, context: GpuContext) -> Self {
        Self {
            title,
            window: None,
            surface: None,
            context,
        }
    }
}

impl<'a> ApplicationHandler for Application<'a> {
    fn new_events(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _cause: winit::event::StartCause,
    ) {
        todo!()
    }

    fn resumed(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        todo!()
    }

    fn window_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        _event: winit::event::WindowEvent,
    ) {
        todo!()
    }

    fn suspended(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        todo!()
    }
}
