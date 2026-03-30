# Defektr — Subnet for Manufacturing Visual Quality Control

<img src="media/DefektrLogo.png" alt="Defektr Logo" width="400" />

Defektr is a Bittensor subnet that produces edge-deployable AI models for manufacturing visual defect detection. Miners compete to build the best defect detection models, validators evaluate them on real benchmark images, and factories purchase the top-performing models.

**Hackathon:** Bittensor Subnet Ideathon — Round 2

---

## How It Works

```
1. Validator publishes a challenge spec on-chain (defect domain, task type, constraints)
2. Miners train defect detection models offline
3. Miners upload model.onnx + metadata.json to IPFS, commit metadata CID on-chain
4. Validator reads commits each epoch, downloads new/updated models to local cache
5. Validator runs each model on block-hash seeded benchmark images
6. Validator scores and calls set_weights() on-chain every epoch
7. Top miner earns TAO — winner-takes-all
8. New challenge block → validator publishes new spec → cycle repeats
```

**Key design:** No Axon/Synapse. Miners upload ONNX files to IPFS. Validator evaluates locally. Zero self-reporting — all metrics are validator-measured.

---

## Scoring System

```
Score = 0.50 × accuracy + 0.30 × speed + 0.20 × robustness
```

**Accuracy** = `0.60 × classification_reward + 0.40 × mask_iou_reward`
- `classification_reward` — normalized BCE on sigmoid confidence output
- `mask_iou_reward` — pixel IoU between predicted mask and ground truth (0.0 if no mask head)

**Speed** — linear decay from T_soft=200ms (full reward) to T_hard=1000ms (zero), measured on validator hardware

**Robustness** — fraction of augmented probe images where hard label is consistent across augmentations

**Edge deployability gate** — hard fail (score=0) if simulated FPS < min_fps (simulated at 15× CPU time to approximate RPi4)

**Copy detection** — if two miners score within 0.2% of each other, the later-registered one is penalized to 0

---

## Miner Architectures (Phase 1)

| Model | Architecture | Outputs | Max score |
|---|---|---|---|
| Baseline | MobileNetV2 classifier | `confidence` only | ~0.80 |
| Improved | MobileNetV2 + U-Net decoder | `confidence` + `mask` | ~0.88 |

The improved model's U-Net decoder unlocks the localisation component of the accuracy reward (α_loc = 0.40), giving it a significant scoring advantage while staying within the edge deployability constraints.

### Benchmark results (50-image validation set)

| Metric | Baseline | Improved |
|--------|----------|----------|
| Total reward | **0.7999** | **0.8766** |
| Accuracy (×0.50) | 0.5496 | 0.5893 |
| — Classification | 0.9160 | 0.8973 |
| — Localisation (IoU) | 0.0000 | 0.1274 |
| Speed (×0.30) | 1.0000 | 1.0000 |
| Latency | 6–8 ms | 26–32 ms |
| Robustness (×0.20) | 1.0000 | 1.0000 |

---

## Prerequisites

- Python 3.12
- Pinata account with API key + dedicated gateway (for IPFS model uploads)
- A funded wallet on the target network (testnet or mainnet)

### Environment file

Create `.env` in the `defektr/` directory:

```env
PINATA_JWT=your_jwt_token
PINATA_GATEWAY=your-subdomain.mypinata.cloud
```

Get the gateway from: Pinata dashboard → Gateways tab → create a dedicated gateway.

### Python environment

```bash
pip install -r requirements.txt
```

---

## Configuration — Files to Update When Changing Network or UIDs

When deploying to a different network or after a chain reset (netuid may change), update the following files:

### RPC endpoint

The testnet RPC (`wss://test.finney.opentensor.ai`) is hardcoded in these files — change all of them to your target endpoint:

| File | Variable/argument |
|------|----------|
| `scripts/setup_hotkeys.py` | `ENDPOINT` |
| `scripts/register_neurons.py` | `ENDPOINT` |
| `scripts/add_stake.py` | `network=` in `bt.Subtensor(...)` |
| `scripts/publish_challenge.py` | `--network` default argument |
| `scripts/run_validator.sh` | `NETWORK` default (line 3) |
| `scripts/run_miner.sh` | `NETWORK` default (line 3) |

For **localnet**, replace `wss://test.finney.opentensor.ai` with `ws://127.0.0.1:9944`.

### Netuid

After registering a new subnet (netuid may differ), update `NETUID` in:

| File | Variable |
|------|----------|
| `subnet/defektr/config.py` | `NETUID` — used by all validator/protocol code |
| `scripts/setup_hotkeys.py` | `NETUID` — used for registration and staking |
| `scripts/register_neurons.py` | `NETUID` |
| `scripts/add_stake.py` | `NETUID` |

### Wallet paths

The wallet path defaults to `defektr/wallets/` (project-local). If you store wallets elsewhere, pass `--wallet-path <path>` at runtime or update the `WALLET_PATH` constant in each script.

---

## Running the Demo on Testnet (wss://test.finney.opentensor.ai)

Current live deployment on **netuid 29**.

### Step 1 — Fund wallets

Get test TAO from the Bittensor faucet, then transfer to your miner coldkeys:

```bash
btcli wallet transfer --wallet.name owner --wallet.path wallets \
    --dest <miner_coldkey_address> --amount 1000 \
    --subtensor.network wss://test.finney.opentensor.ai
```

### Step 2 — Register your subnet

Subnet registration costs ~1000 TAO (burn). After registration a 720-block cooldown applies globally before another subnet can be registered on the same network.

```bash
python scripts/setup_subnet.py
```

Note the netuid returned and update `NETUID` in `subnet/defektr/config.py` and `scripts/setup_hotkeys.py` if it differs from 29.

### Step 3 — Create and register hotkeys

`--skip-funding` skips the Alice sudo step (which only works on localnet):

```bash
# 2 miners (baseline + improved) + 1 validator
python scripts/setup_hotkeys.py --miners 2 --skip-funding

# 3 miners (adds a copy miner for copy-detection demo)
python scripts/setup_hotkeys.py --miners 3 --skip-funding
```

This creates wallets, registers all hotkeys on the subnet, and stakes the validator hotkey.

### Step 4 — Upload models to IPFS

```bash
export WALLETS=/path/to/defektr/wallets

# Miner 1 — baseline model
python training/upload.py \
    --model training/models/baseline.onnx \
    --spec challenge_spec.json \
    --wallet-name miner --wallet-hotkey hotkey_0 \
    --wallet-path $WALLETS --architecture mobilenet_v2

# Miner 2 — improved model (MobileNetV2 + U-Net)
python training/upload.py \
    --model training/models/improved.onnx \
    --spec challenge_spec.json \
    --wallet-name miner2 --wallet-hotkey default \
    --wallet-path $WALLETS --architecture mobilenet_v2_unet

# Miner 3 — copy of baseline (triggers copy detection)
python training/upload.py \
    --model training/models/baseline.onnx \
    --spec challenge_spec.json \
    --wallet-name miner3 --wallet-hotkey default \
    --wallet-path $WALLETS --architecture mobilenet_v2
```

> **Note:** The miner hotkey needs stake on the root network (netuid 0) for `set_commitment()` to work. If you get `AccountNotAllowedCommit`, run:
> ```bash
> btcli stake add --wallet.name miner --wallet.hotkey hotkey_0 \
>     --wallet.path wallets --netuid 0 --amount 10 \
>     --subtensor.network wss://test.finney.opentensor.ai
> ```

### Step 5 — Publish challenge spec

```bash
python scripts/publish_challenge.py --spec challenge_spec.json --update-blocks
```

Uploads `challenge_spec.json` to IPFS and commits the CID on-chain from `validator/hotkey_0`. The `--update-blocks` flag auto-sets `challenge_block` and `deadline_block` relative to the current block.

### Step 6 — Run the validator

```bash
bash scripts/run_validator.sh hotkey_0 wss://test.finney.opentensor.ai /path/to/defektr/wallets
```

Logs are saved to `logs/validator_hotkey_0_<timestamp>.log`.

Expected output every epoch (~100 blocks ≈ 20 min on testnet):

```
[uid 1] score=0.7999  latency=6.2ms  outputs=1 (bin)
[uid 2] score=0.8766  latency=25.8ms  outputs=2 (seg)
[uid 3] score=0.7999  latency=6.1ms  outputs=1 (bin)
[copy-detection] uid=3 score=0.7999 matches uid=1 score=0.7999 (diff=0.0000) — uid=3 penalized to 0
Scores this epoch: uid=2 0.8766  uid=1 0.7999  uid=3 0.0000
set_weights on chain successfully!
```

### Step 7 — Run miners (for incentive logging)

```bash
# Baseline miner
bash scripts/run_miner.sh miner hotkey_0 wss://test.finney.opentensor.ai /path/to/defektr/wallets

# Improved miner
bash scripts/run_miner.sh miner2 default wss://test.finney.opentensor.ai /path/to/defektr/wallets
```

---

## Running the Demo Locally (ws://127.0.0.1:9944)

### Step 1 — Start the local Bittensor chain

```bash
docker run -d --name local_chain -p 9944-9945:9944-9945 ghcr.io/opentensor/subtensor-localnet:devnet-ready
# or start existing:
docker start local_chain
```

### Step 2 — Chain reset (run after every Docker restart)

```bash
python scripts/fund_wallets.py      # Alice sudo → 2000 TAO to owner/miner/validator
python scripts/setup_subnet.py      # creates subnet → netuid 2
python scripts/register_neurons.py  # registers validator/default on netuid 2
python scripts/add_stake.py         # stakes 100 TAO to validator/default
```

### Step 3 — Create and register hotkeys

```bash
python scripts/setup_hotkeys.py --miners 3   # funds via Alice sudo, no --skip-funding needed
```

### Steps 4–7

Same as testnet steps 4–7 above, replacing the network argument with `ws://127.0.0.1:9944`.

---

## Utility Commands

```bash
# Check all registered UIDs, incentives, and stakes
python scripts/show_metagraph.py

# Generate model output comparison image (baseline vs improved)
python scripts/visualize_models.py --out model_comparison.png

# Use a specific image
python scripts/visualize_models.py \
    --image data/datasets/bottle/test/broken_large/000.png \
    --out broken_large.png
```

---

## Protocol Details

### What miners commit on-chain

Only the **metadata CID** goes on-chain via `subtensor.set_commitment()`. The metadata JSON contains:

```json
{
  "challenge_id": "defektr-001",
  "miner_hotkey": "5F...",
  "model": {
    "filename": "model.onnx",
    "cid": "bafyXXX",
    "size_mb": 8.7,
    "sha256": "abc..."
  },
  "inference": {
    "input_name": "image",
    "input_shape": [1, 3, 256, 256],
    "outputs": [{"name": "confidence"}, {"name": "mask"}],
    "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "threshold": 0.5
  }
}
```

Validator reads metadata CID → fetches metadata.json → reads `model.cid` → downloads model.onnx.

### Model cache

Validator only re-downloads a model when the committed CID changes. This avoids repeated IPFS downloads when the model hasn't changed between epochs.

### Block-hash seeding

Benchmark images are sampled using the challenge block hash as the seed. This makes sampling deterministic and publicly verifiable, preventing validator-miner collusion.

---

## Anti-Gaming

### Copy detection
If two miners produce scores within 0.2% of each other, the later-registered miner (higher UID) is penalized to 0. Registered-first = treated as original.

### Overfitting prevention
- Large benchmark pool (MVTec AD — 15 industrial defect categories)
- Block-hash seeded sampling — miners cannot predict which images will be used
- Random augmentations at inference time feed the robustness score

### Model size limit
`max_model_size_mb = 100` in challenge spec prevents miners from uploading huge models to stall the validator.

### Edge deployability gate
Models are tested at simulated RPi4 speed (15× CPU time). Models that cannot run at ≥2 FPS on edge hardware score 0.

---

## What Is Still To Do

### For mainnet / production

- [ ] **Root subnet registration** — validator must register on netuid 0 and set weights there for TAO emissions to flow into the subnet. Without this, `metagraph.incentive` stays 0.
- [ ] **Real edge hardware benchmarking** — calibrate `HARDWARE_FACTOR` by measuring actual CPU time ratio between validator machine and RPi4. Currently set to 15× as an approximation.
- [ ] **Multi-dataset benchmark** — extend beyond MVTec AD bottle to all 7 planned datasets (NEU Surface Defect, DAGM, Kolektor SDD, DeepPCB, Severstal, Casting Product).
- [ ] **Challenge rotation** — validator publishes new challenge spec every `CHALLENGE_INTERVAL` blocks with a different defect domain.
- [ ] **Decentralized validator set** — currently running as a single centralized validator (Chutes approach). Phase 2 should open validator registration.
- [ ] **Frontend** — factory dashboard for browsing and purchasing top models.
- [ ] **Mainnet deployment** — register subnet on mainnet, configure proper `EPOCH_TEMPO` and `CHALLENGE_INTERVAL` for production timing.

### Nice to have

- [ ] Cosine-similarity based copy detection (compare raw output vectors, not just final scores)
- [ ] Malicious miner detection via model architecture fingerprinting
- [ ] `publish_challenge.py` wired into validator startup (currently manual)
- [ ] Model versioning — track multiple submissions per miner per challenge
