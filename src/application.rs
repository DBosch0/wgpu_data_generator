#![allow(dead_code)]

use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{Key, NamedKey},
    window::{self, Window},
};

use crate::{GpuContext, SurfaceWrapper};

struct ApplicationState;

impl ApplicationState {
    fn new() -> Self {
        // TODO State of the application
        Self
    }

    fn resize(&mut self) {}

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {}

    fn update(&mut self, event: WindowEvent) {}
}

pub(crate) struct Application<'a> {
    title: &'static str,
    window: Option<Arc<Window>>,
    surface: SurfaceWrapper<'a>,
    context: GpuContext,
    state: Option<ApplicationState>,
}

impl<'a> Application<'a> {
    pub(crate) fn new(
        title: &'static str,
        context: GpuContext,
        surface: SurfaceWrapper<'a>,
    ) -> Self {
        Self {
            title,
            window: None,
            surface,
            context,
            state: None,
        }
    }

    pub(crate) fn initialize(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = window::WindowAttributes::default().with_title(self.title);
        self.window = Some(Arc::new(
            event_loop
                .create_window(window_attributes)
                .expect("creating window"),
        ));

        self.surface.resume(
            &self.context,
            Arc::clone(self.window.as_ref().unwrap()),
            false,
        );
        self.state = Some(ApplicationState::new());
    }
}

impl<'a> ApplicationHandler for Application<'a> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        //after NewEvent::StartCause::Init this function is called. here we intialize
        //the window, surface, and application state
        self.initialize(event_loop);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(size) => {
                self.surface.resize(&self.context, size);
                self.state.as_mut().unwrap().resize();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            }
            | WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(s),
                        ..
                    },
                ..
            } if s == "r" => {
                println!("{:#?}", self.context.instance.generate_report());
            }
            WindowEvent::RedrawRequested => {
                if self.state.is_none() {
                    return;
                }

                let frame = self.surface.acquire(&self.context);
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(self.surface.config().view_formats[0]),
                    ..wgpu::wgt::TextureViewDescriptor::default()
                });

                self.state.as_mut().unwrap().render(
                    &view,
                    &self.context.device,
                    &self.context.queue,
                );

                self.window.as_ref().unwrap().pre_present_notify();
                frame.present();

                self.window.as_ref().unwrap().request_redraw();
            }
            _ => self.state.as_mut().unwrap().update(event),
        }
    }
}
