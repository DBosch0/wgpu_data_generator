#![allow(dead_code)]

use crate::application::Application;
use winit::event_loop::EventLoop;

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

fn init_logger() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .filter_module("wgpu_core", log::LevelFilter::Info)
        .filter_module("wgpu_hall", log::LevelFilter::Error)
        .filter_module("naga", log::LevelFilter::Error)
        .parse_default_env()
        .init();
}

fn run() -> anyhow::Result<()> {
    init_logger();

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = Application::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}

fn main() {
    run().unwrap()
}
