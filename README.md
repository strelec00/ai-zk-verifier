# AI ZK Verifier ⛓️‍💥

## Description  
A zero-knowledge proof system using RISC Zero zkVM to create proofs that ML model interference was right. Solana blockchain is used as a verifier.

## Goal  
Develop and analyze a system where AI model inference results are computed and verified using verifiable computing techniques, with blockchain serving as a trust and audit layer.  

## Description  

1. **AI Model Selection**  
   - Choose a simple yet effective AI model (e.g., a neural network for image classification or an existing model used in the lab).  
   - Analyze the computational complexity of inference.  

2. **Verifiable Computing**  
   - Explore techniques such as **zk-SNARKs** or **zk-STARKs** for generating proofs of AI inference correctness.  
   - Alternatively, use a **zkVM virtual machine** (e.g., Risc0) to generate proofs of correct inference execution.  

3. **Blockchain Integration**  
   - Implement a **smart contract** on a blockchain platform (Ethereum or a testnet in the lab).  
   - Verify the generated proof of correctness.  
   - Optionally, store the verified result and associated proof on the blockchain for auditing.  

4. **Decentralized Application (Optional)**  
   - Create a simple user interface where users can upload input data for inference and retrieve verified results.  

## Technologies  
- **ML Model**: PyTorch / TensorFlow -> ONNX format
- **Verifiable Computing**: zkVM, Risc0  
- **Blockchain**: Solana, zkVM  
- **Frontend (Optional)**: React, Next.js  

## How to Run  
Coming soon...  

## License  
MIT  
