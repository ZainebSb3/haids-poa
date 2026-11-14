# Testbed reference configuration

**This is a reference template rebuilt from the parameters stated in the paper**
(Geth v1.13.15, Clique, chain ID 1337, single sealer, epoch 30,000, bootnode discovery).
It is not the original file used in April 2026. The gas limit and the pre-funded accounts
are placeholders.

Block period: the sessions of 8-9 April 2026 (Eclipse, Sybil) ran with `period: 5`;
all later sessions ran with `period: 15`.

Start a sealer (A1) after `geth init genesis.json`:

    geth --datadir ./A1 --networkid 1337 --port 30303 \
         --http --http.addr 0.0.0.0 --http.port 8545 \
         --http.api eth,net,web3,txpool,admin \
         --bootnodes <BOOTNODE_ENODE> \
         --unlock <SEALER_ADDRESS> --password ./password.txt --mine \
         --miner.etherbase <SEALER_ADDRESS> --allow-insecure-unlock

Run this only on an isolated network: the admin API and an unlocked account are exposed.
