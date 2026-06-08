#![allow(dead_code)]
use winit::event_loop::EventLoop;

use crate::application::Application;

mod application;

async fn get_adapter_with_capabilities_or_from_env(
    instance: &wgpu::Instance,
    required_features: &wgpu::Features,
    required_downlevel_capabilities: &wgpu::DownlevelCapabilities,
) -> wgpu::Adapter {
    use wgpu::Backends;

    if std::env::var("WGPU_ADAPTER_NAME").is_ok() {
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(instance, None)
            .await
            .expect("No suitable GPU adapters found on system");
        let adapter_info = adapter.get_info();
        log::info!("Using {} ({:?})", adapter_info.name, adapter_info.backend);

        let adapter_features = adapter.features();
        assert!(
            adapter_features.contains(*required_features),
            "Adapter does not support the required features: {:?}",
            *required_features - adapter_features
        );

        let downlevel_capabilities = adapter.get_downlevel_capabilities();
        assert!(
            downlevel_capabilities.shader_model >= required_downlevel_capabilities.shader_model,
            "Adapter does not support the minimum shader model required to run: {:?}",
            required_downlevel_capabilities.shader_model
        );
        adapter
    } else {
        let adapters = instance.enumerate_adapters(Backends::all()).await;

        let mut chosen_adapter = None;
        for adapter in adapters {
            let required_features = *required_features;
            let adapter_features = adapter.features();
            if !adapter_features.contains(required_features)
                || adapter.get_downlevel_capabilities().shader_model
                    < required_downlevel_capabilities.shader_model
            {
                continue;
            } else {
                chosen_adapter = Some(adapter);
                break;
            }
        }

        let adapter = chosen_adapter.expect("No suitable GPU adapters found on the system!");
        let adapter_info = adapter.get_info();
        log::info!("Using {} ({:?})", adapter_info.name, adapter_info.backend);
        adapter
    }
}

pub(crate) struct SurfaceWrapper<'a> {
    pub(crate) surface: wgpu::Surface<'a>,
    pub(crate) config: wgpu::SurfaceConfiguration,
}

#[derive(Debug)]
pub(crate) struct GpuContext {
    pub(crate) instance: wgpu::Instance,
    pub(crate) adapter: wgpu::Adapter,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
}

impl GpuContext {
    async fn init_async() -> Self {
        log::info!("inializing the GPU");

        let instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        let instance = wgpu::Instance::new(instance_descriptor);
        let adapter = get_adapter_with_capabilities_or_from_env(
            &instance,
            &wgpu::Features::empty(),
            &wgpu::DownlevelCapabilities {
                flags: wgpu::DownlevelFlags::COMPUTE_SHADERS,
                ..Default::default()
            },
        )
        .await;
        let needed_limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());

        let (device, queue) = adapter
            .request_device(&wgpu::wgt::DeviceDescriptor {
                label: Some("Device Descriptor"),
                required_features: adapter.features(), //NOTE: if we need additional features add here
                required_limits: needed_limits,
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                memory_hints: wgpu::MemoryHints::MemoryUsage, //NOTE: could switch to performance if necessary
                trace: match std::env::var_os("WGPU_TRACE") {
                    Some(path) => wgpu::Trace::Directory(path.into()),
                    None => wgpu::Trace::Off,
                },
            })
            .await
            .expect("Unable to find suitable GPU");

        Self {
            instance,
            adapter,
            device,
            queue,
        }
    }
}

fn init_logger() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .filter_module("wgpu_core", log::LevelFilter::Info)
        .filter_module("wgpu_hall", log::LevelFilter::Error)
        .filter_module("naga", log::LevelFilter::Error)
        .parse_default_env()
        .init();
}

async fn start(title: &'static str) {
    init_logger();

    let backends = wgpu::Instance::enabled_backend_features();
    log::info!("Enabled Backends {:?}", backends);
    let context = GpuContext::init_async().await;
    log::info!("Create GpuContext");

    let mut application = Application::new(title, context);
    let event_loop = EventLoop::new().expect("Can construct an event loop");

    log::info!("Entering Event Loop");
    event_loop
        .run_app(&mut application)
        .expect("No Event Loop Errors");
}

fn main() {
    log::debug!("test");
    pollster::block_on(start("obj_viewer"))
}
