# AI ZK Verifier

## Zadatak  
Implementacija dokaza ispravnosti inferencije modela strojnog učenja na blockchainu.  

## 🎯 Cilj  
Implementirati i analizirati sustav u kojem se rezultati inferencije AI modela izračunavaju i verificiraju korištenjem tehnika provjerljivog računarstva, dok blockchain služi kao sloj povjerenja i revizije.  

## 📖 Opis  

1. **Odabir AI modela**  
   - Odabrati jednostavni, ali učinkovit AI model (npr. neuronsku mrežu za klasifikaciju slika ili neki već korišten u laboratoriju).  
   - Analizirati računalnu složenost inferencije.  

2. **Provjerljivo računarstvo**  
   - Istražiti tehnike poput **zk-SNARKs** ili **zk-STARKs** za generiranje dokaza ispravnosti AI inferencije.  
   - Alternativno, koristiti **zkVM virtualni stroj** (npr. Risc0) za generiranje dokaza o ispravnom izvršavanju inferencije.  

3. **Blockchain integracija**  
   - Implementirati **pametni ugovor** na blockchain platformi (Ethereum ili testnet u laboratoriju).  
   - Verificirati generirani dokaz ispravnosti.  
   - Opcionalno, pohraniti verificirani rezultat i povezani dokaz na blockchain radi revizije.  

4. **Decentralizirana aplikacija (opcionalno)**  
   - Kreirati jednostavno korisničko sučelje za prijenos ulaznih podataka i preuzimanje verificiranih rezultata.  

## 🛠️ Tehnologije  
- **AI model**: PyTorch / TensorFlow  
- **Provjerljivo računarstvo**: zk-SNARKs, zk-STARKs, Risc0  
- **Blockchain**: Solana, zkVM  
- **Frontend (opcionalno)**: React, Next.js  

## 🚀 Kako pokrenuti  
Dolazi uskoro...  

## 📜 Licenca  
MIT  
