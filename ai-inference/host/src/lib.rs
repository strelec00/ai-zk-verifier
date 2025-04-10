use multiply_methods::MULTIPLY_ELF;
use risc0_zkvm::{default_prover, ExecutorEnv, Receipt};
use sha2::{Sha256, Digest};
use image::DynamicImage;
use tract_onnx::prelude::*;
use tract_onnx::prelude::tract_ndarray::Array4;

pub fn verify_inference(image_data: &DynamicImage, model_hash: &[u8], result_label: &str) -> (Receipt, Vec<u8>) {
    let mut hasher = Sha256::new();

    let image_bytes = image_data.clone().into_bytes();
    hasher.update(&image_bytes);
    hasher.update(model_hash);
    hasher.update(result_label.as_bytes());
    let final_hash: [u8; 32] = hasher.finalize().into();

    let env = ExecutorEnv::builder()
        .write(&final_hash)
        .unwrap()
        .build()
        .unwrap();

    let prover = default_prover();
    let receipt = prover.prove(env, MULTIPLY_ELF).unwrap().receipt;

    (receipt, final_hash.to_vec())
}

// Softmax funkcija
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Računanje eksponeencijalnih vrijednosti i zbrajanje za normalizaciju
    let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let exp_sum: f32 = exp_values.iter().sum();

    // Normalizacija vrijednosti
    exp_values.iter().map(|&x| x / exp_sum).collect()
}

pub fn run_onnx_inference(image: &DynamicImage, model_data: &[u8]) -> String {
    // Resize bez gubitka proporcija, zatim crop (isto kao torchvision Resize(224, 224))
    let resized = image.resize(224, 224, image::imageops::FilterType::Triangle).to_rgb8();

    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    // Konverzija iz HWC -> CHW + normalizacija
    let mut float_data = vec![0f32; 3 * 224 * 224];
    for (y, x, pixel) in resized.enumerate_pixels() {
        for c in 0..3 {
            let value = pixel.0[c] as f32 / 255.0;
            let normalized = (value - mean[c]) / std[c];
            float_data[c * 224 * 224 + (y as usize) * 224 + (x as usize)] = normalized;
        }
    }

    let image_tensor = Tensor::from(Array4::from_shape_vec((1, 3, 224, 224), float_data).unwrap());

    let model = tract_onnx::onnx()
        .model_for_read(&mut &*model_data)
        .unwrap()
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 224, 224)))
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();

    let result = model.run(tvec!(image_tensor.into())).unwrap();
    let output_tensor = result[0].to_array_view::<f32>().unwrap();
    let logits = output_tensor.as_slice().unwrap();

    // Softmax
    let probs = softmax(logits);
    println!("Rust Softmax vjerojatnosti: {:?}", probs);

    let (label_index, confidence) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    println!("Rust predikcija: index={}, confidence={:.4}", label_index, confidence);

    // Prag povjerenja
    if *confidence < 0.799 {
        return "unknown".to_string();
    }

    // Mapiranje na klase (redoslijed mora biti isti kao ImageFolder class_to_idx)
    match label_index {
        0 => "macka".to_string(),
        1 => "pas".to_string(),
        _ => "unknown".to_string(),
    }
}
