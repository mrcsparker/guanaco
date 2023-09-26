use magnus::{define_module, function, prelude::*, Error};

pub mod camel;

fn get_accelerator() -> String {
    match llm_base::ggml::accelerator::get_accelerator() {
        llm_base::ggml::accelerator::Accelerator::CuBLAS => "cuda".to_owned(),
        llm_base::ggml::accelerator::Accelerator::CLBlast => "opencl".to_owned(),
        llm_base::ggml::accelerator::Accelerator::Metal => "metal".to_owned(),
        _ => "cpu".to_owned(),
    }
}

#[magnus::init]
fn init() -> Result<(), Error> {
    let namespace = define_module("Guanaco")?;
    namespace.define_singleton_method("get_accelerator", function!(get_accelerator, 0))?;

    camel::setup(namespace)?;

    Ok(())
}
