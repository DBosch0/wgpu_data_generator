use crate::application::Application;
use winit::event_loop::EventLoop;

mod application;
mod camera;
mod config;
mod degradation;
mod generator;
mod gltf_model;
mod landmark;
mod lighting;
mod model;
mod offscreen;
mod resource;
mod texture;

pub(crate) async fn get_adapter_with_capabilities_or_from_env(
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

fn run_windowed_demo() -> anyhow::Result<()> {
    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = Application::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}

fn run_headless(config_path: &std::path::Path) -> anyhow::Result<()> {
    let cfg = config::load_config(config_path)?;
    let mut g = generator::Generator::new(cfg)?;
    g.run()
}

fn main() {
    init_logger();

    let args: Vec<String> = std::env::args().collect();

    let result = if args.contains(&"--headless".to_string()) || args.contains(&"--preview".to_string()) {
        // Find --config <path>
        let config_path = args
            .iter()
            .position(|a| a == "--config")
            .and_then(|i| args.get(i + 1))
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("sample_config.toml"));

        if args.contains(&"--preview".to_string()) {
            // TODO Phase 7: run generator in parallel with a winit preview window.
            // For now, fall through to headless and note that preview is not yet wired.
            log::warn!("--preview: preview window not yet implemented, running headless");
        }

        run_headless(&config_path)
    } else {
        // Default: windowed cube demo
        run_windowed_demo()
    };

    if let Err(e) = result {
        log::error!("{e}");
        std::process::exit(1);
    }
}
