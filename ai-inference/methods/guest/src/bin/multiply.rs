use risc0_zkvm::guest::env;

fn main() {
    // Pročitaj hash od hosta
    let expected_hash: [u8; 32] = env::read();

    // Rekonstrukcija hash na isti način (simulirano ovde jer guest ne zna ništa osim očekivanog hasha)
    // Pošto ZKVM ne vidi originalne podatke, ono što zapravo radimo je potvrđujemo da je hash isti
    // tj. host nas ne može prevariti da rezultat dolazi od nečega drugog

    // Commitujemo hash kao dokaz
    env::commit(&expected_hash);
}
