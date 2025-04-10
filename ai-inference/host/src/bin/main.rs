use host::{run_onnx_inference, verify_inference};
use multiply_methods::MULTIPLY_ID;
use sha2::{Sha256, Digest};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Učitaj sliku i model
    let image_data = image::open("images/pas.jpg").expect("Failed to read image");
    let model_data = std::fs::read("model/model.onnx").expect("Failed to read model");

    // Lokalna inferenca
    let result_label = run_onnx_inference(&image_data, &model_data); // npr. "cat", "dog", "unknown"

    // Hash modela
    let model_hash = Sha256::digest(&model_data);

    // Verifikacija u ZKVM-u
    let (receipt, _) = verify_inference(&image_data, &model_hash, &result_label);

    // ✅ Verifikacija receipt-a
    match receipt.verify(MULTIPLY_ID) {
        Ok(_) => {
            println!("\nZK dokaz je verificiran!");
        }
        Err(e) => {
            eprintln!("Verification failed: {:?}", e);
            std::process::exit(1);
        }
    }

    println!("\n📷 Slika klasifikovana kao: {}\n", result_label);
}
