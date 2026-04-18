# =============================================================================
# config.py — All hyperparameters and constants in one place
# Cognitive Radio Network TD3 Simulation
# Ramaiah Institute of Technology, Bangalore
# =============================================================================

# ─── System Model ────────────────────────────────────────────────────────────
SIGMA2 = 1e-3        # AWGN noise power at each receiver (Watts)
P_P    = 1.0         # Primary Transmitter (PT) fixed transmit power (Watts)
P_MAX  = 3.0         # Maximum Secondary Transmitter (ST) power (Watts)

# ─── Reward Function Weights ─────────────────────────────────────────────────
ALPHA         = 1.0   # Weight for SU throughput (R_s)
BETA          =1.5  # Penalty weight for PU SINR constraint violation
GAMMA_REWARD  = 0.005  # Small penalty for energy usage (encourages efficiency)
SINR_THRESHOLD = 1.0  # Minimum acceptable SINR at PR (~0 dB)

# ─── RL / TD3 Hyperparameters ─────────────────────────────────────────────────
STATE_DIM           = 7        # [h_pp^2, h_sp^2, h_ss^2, h_ps^2, SINR_p, SINR_s, P_s_prev]
ACTION_DIM          = 1        # P_s (continuous, scalar)
HIDDEN_DIM          = 512      # Width of hidden layers in Actor and Critic

REPLAY_BUFFER_SIZE  = 200_000  # Maximum transitions stored
MIN_SAMPLES         = 1000     # Buffer must have this many before training starts

POLICY_NOISE        = 0.2      # Std of target policy smoothing noise (fraction of P_MAX)
NOISE_CLIP          = 0.5      # Clipping bound for target policy noise (fraction of P_MAX)
POLICY_DELAY        = 2        # Update actor only every N critic updates

EXPLORATION_NOISE_STD = 0.1    # Initial exploration noise std (fraction of P_MAX)
EXPLORATION_NOISE_END = 0.01   # Final exploration noise std (fraction of P_MAX)

BATCH_SIZE          = 256      # Mini-batch size for gradient updates
GRAD_UPDATES_PER_STEP = 4     # Gradient updates per env step (UTD ratio) — keeps GPU busy
LR_ACTOR            = 3e-4     # Adam learning rate for Actor
LR_CRITIC           = 3e-4     # Adam learning rate for Critics
GAMMA_DISCOUNT      = 0.99     # Discount factor for future rewards
TAU                 = 0.005    # Soft target network update rate (Polyak averaging)

TRAINING_EPISODES   = 7500     # Total training episodes
STEPS_PER_EPISODE   = 200      # Time steps per episode (channel coherence blocks)

# ─── Nakagami-m Fading Channel ───────────────────────────────────────────────
NAKAGAMI_M     = 3.0   # Fading severity (m=1 → Rayleigh; m=3 → moderate Nakagami fading)
NAKAGAMI_OMEGA = 1.0   # Mean power per link (Ω)

# ─── WebSocket / Server ───────────────────────────────────────────────────────
WS_HOST            = "0.0.0.0"
WS_PORT            = 8000
BROADCAST_INTERVAL = 50    # Broadcast every N environment steps
SCATTER_WINDOW     = 200   # Number of (SINR_dB, BER) pairs kept for scatter chart
OUTAGE_WINDOW      = 500   # Rolling window size for outage probability estimate
