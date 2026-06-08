fn init_logger() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .filter_module("wgpu_core", log::LevelFilter::Info)
        .filter_module("wgpu_hall", log::LevelFilter::Error)
        .filter_module("naga", log::LevelFilter::Error)
        .parse_default_env()
        .init();
}

async fn start(_title: &'static str) {
    init_logger();

    let backends = wgpu::Instance::enabled_backend_features();
    log::info!("Enabled Backends {:?}", backends);
}

fn main() {
    log::debug!("test");
    pollster::block_on(start("obj_viewer"))
}
