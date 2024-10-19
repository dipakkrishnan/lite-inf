use clap::Parser;

#[derive(Parser)]
struct CliArgs {
    model: String
}

use candle_core::quantized::gguf_file;
use candle_core::{Device};
use candle_transformers::models::quantized_llama as model;
use model::ModelWeights;
use anyhow::{Result};

const DEVICE: Device = Device::Cpu;

fn main() -> Result<()> {
    // below from https://github.com/huggingface/candle/issues/1301
    let model_path = "/Users/dipakkrishnan/.cache/huggingface/hub/models--bartowski--Llama-3.2-1B-Instruct-GGUF/snapshots/067b946cf014b7c697f3654f621d577a3e3afd1c/Llama-3.2-1B-Instruct-f16.gguf";
    let mut file;
    match std::fs::File::open(&model_path) {
        Ok(model_file) => file = model_file,
        Err(err) => {
            panic!("Error loading model from file during inference: {:?}", err);
        }
    }
    let start = std::time::Instant::now();

    let model_content = match gguf_file::Content::read(&mut file) {
        Ok(file_data) => file_data,
        Err(err) => {
            panic!("Error loading file data from model file: {:?}", err);
        }
    };
    println!(
        "loaded {:?} tensors in {:.2}s",
        model_content.tensor_infos.len(),
        start.elapsed().as_secs_f32(),
    );

    let _model = match ModelWeights::from_gguf(model_content, &mut file, &DEVICE) {
        Ok(model_weights) => model_weights,
        Err(err) => {
            panic!("Error extracting weights from gguf: {:?}", err);
        }
    };

    Ok(())
}
