# ZKML Solana ⛓️‍💥

## Goal  
A zero-knowledge proof system using the EZKL library with ONNX-format models to generate proofs that machine learning inference was performed correctly. The Solana blockchain is used as a verifier.

## Description  

1. **AI Model Selection**  
   - Choose a simple yet effective AI model (e.g., a neural network for image classification or an existing model used in the lab).
   - Export the model to **ONNX format** for compatibility with the EZKL library.  
   - Analyze the computational complexity of inference.  

2. **Verifiable Computing**  
   - Use the **EZKL** library to generate zero-knowledge proofs of correct model inference.
   - Alternatively, use a **zkVM virtual machine** (e.g., Risc0) to generate proofs of correct inference execution.  

3. **Blockchain Integration**  
   - Implement a **smart contract** on a blockchain platform (Ethereum or a testnet in the lab).  
   - Verify the generated proof of correctness.  
   - Optionally, store the verified result and associated proof on the blockchain for auditing.  

4. **Decentralized Application (Optional)**  
   - Create a simple user interface where users can upload input data for inference and retrieve verified results.  

## Technologies  
- **ML Model**: PyTorch / TensorFlow -> ONNX format
- **Verifiable Computing**: EZKL, *zkVM, *Risc0  
- **Blockchain**: Solana  
- **Frontend (Optional)**: React, Next.js  

## How to Run  
Coming soon...  

## License  
MIT  
