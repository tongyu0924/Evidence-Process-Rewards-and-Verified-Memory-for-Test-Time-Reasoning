"""Default settings. Most can be overridden from the command line (see run.py)."""

# Data
DATASET = "bio-nlp-umass/MedThinkVQA"   # Hugging Face dataset with train/test splits
OUT_DIR = "results"
LIMIT = 720            # number of test cases
TRAIN_N = 300          # training cases used to build the Playbook
CREDIT_N = 100         # labelled cases used to estimate entry credit
VAL_N = 50             # validation cases
BATCH = 16
MAX_IMAGES = 16
IMG_SIDE = 768

# Models and serving
GEN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
JUDGE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
JUDGE_QUANT = "fp8"    # "" disables quantization
GPUS = "auto"          # "auto" or a list such as "0,1"
GEN_NEED_GB = 14
JUDGE_NEED_GB = 21
GEN_PORT = 8101
JUDGE_PORT = 8102
GEN_MAX_LEN = 16384
JUDGE_MAX_LEN = 12288
GEN_CONC = 32
JUDGE_CONC = 16

# Test-time search
K = 4                  # paths per round
SLOTS = 3              # strategy paths among the K
ROUNDS = 3             # maximum rounds of recursion
TAU = 0.6              # confidence needed for contrastive reflection
TEMP = 0.7
STREAM_B = 10          # test batch size; Playbook updates apply from the next batch
VAL_SAMPLES = 4        # sampled paths per validation case
PAIR_MAX = 4           # (correct, wrong) pairs per validation case

# Evidence PRM
PRM_MAX_STEPS = 10
PRM_AGG = "geo"        # geo | min | mean
ALPHA = 0.5            # weight of visual correctness
VERIFY_IMAGES = 8      # images shown to the judge per call

# Playbook
MAX_FEAT = 8           # entries kept per disease
CHECK_ITEMS = 4        # checklist items per option
ELIM_MIN = 0.5         # lambda: lowest multiplier when no checklist item holds
EVICT = 0.4
MIN_UPD = 3
WRITE_CONF = 0.8
