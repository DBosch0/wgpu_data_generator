fn init_logger() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .filter_module("wgpu_core", log::LevelFilter::Info)
        .filter_module("wgpu_hall", log::LevelFilter::Error)
        .filter_module("naga", log::LevelFilter::Error)
        .parse_default_env()
        .build();
}

async fn start(title: &'static str) {
    todo!()
}

fn main() {
    pollster::block_on(start("obj_viewer"))
}
