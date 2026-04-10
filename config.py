# =============================================================================
# config.py — All hyperparameters and constants in one place
# Cognitive Radio Network TD3 Simulation
# Ramaiah Institute of Technology, Bangalore
# =============================================================================

# ─── System Model ────────────────────────────────────────────────────────────
SIGMA2 = 1e-3        # AWGN noise power at each receiver (Watts)
P_P    = 1.0         # Primary Transmitter (PT) fixed transmit power (Watts)
P_MAX  = 1.0         # Maximum Secondary Transmitter (ST) power (Watts)

# ─── Reward Function Weights ─────────────────────────────────────────────────
ALPHA         = 1.0   # Weight for SU throughput (R_s)
BETA          = 10.0  # Penalty weight for PU SINR constraint violation
GAMMA_REWARD  = 0.01  # Small penalty for energy usage (encourages efficiency)
SINR_THRESHOLD = 2.0  # Minimum acceptable SINR at PR (~3 dB)

# ─── RL / TD3 Hyperparameters ─────────────────────────────────────────────────
STATE_DIM           = 7        # [h_pp^2, h_sp^2, h_ss^2, h_ps^2, SINR_p, SINR_s, P_s_prev]
ACTION_DIM          = 1        # P_s (continuous, scalar)
HIDDEN_DIM          = 256      # Width of hidden layers in Actor and Critic

REPLAY_BUFFER_SIZE  = 100_000  # Maximum transitions stored
MIN_SAMPLES         = 1000     # Buffer must have this many before training starts

POLICY_NOISE        = 0.2      # Std of target policy smoothing noise (fraction of P_MAX)
NOISE_CLIP          = 0.5      # Clipping bound for target policy noise (fraction of P_MAX)
POLICY_DELAY        = 2        # Update actor only every N critic updates

EXPLORATION_NOISE_STD = 0.1    # Initial exploration noise std (fraction of P_MAX)
EXPLORATION_NOISE_END = 0.01   # Final exploration noise std (fraction of P_MAX)

BATCH_SIZE          = 128      # Mini-batch size for gradient updates
LR_ACTOR            = 3e-4     # Adam learning rate for Actor
LR_CRITIC           = 3e-4     # Adam learning rate for Critics
GAMMA_DISCOUNT      = 0.99     # Discount factor for future rewards
TAU                 = 0.005    # Soft target network update rate (Polyak averaging)

TRAINING_EPISODES   = 3000000     # Total training episodes
STEPS_PER_EPISODE   = 200      # Time steps per episode (channel coherence blocks)

# ─── GUI / Pygame ─────────────────────────────────────────────────────────────
WINDOW_WIDTH      = 1620
WINDOW_HEIGHT     = 900
LEFT_PANEL_WIDTH  = 560        # Network visualization panel
RIGHT_PANEL_WIDTH = 760        # Live plots panel
INSIGHTS_WIDTH    = 300        # AI Insights sidebar (far right)
TOP_BAR_HEIGHT    = 45
BOTTOM_BAR_HEIGHT = 50
FPS_CAP           = 60

# Node layout: pixel positions within the left panel (x, y from panel origin)
NODE_POSITIONS = {
    "PT": (140, 230),   # Primary Transmitter  — top-left
    "PR": (420, 230),   # Primary Receiver     — top-right
    "ST": (140, 530),   # Secondary Transmitter — bottom-left
    "SR": (420, 530),   # Secondary Receiver   — bottom-right
}

NODE_RADIUS = 30

# Node fill colors (R, G, B)
NODE_COLORS = {
    "PT": (52,  152, 219),   # Blue
    "PR": (46,  204, 113),   # Green
    "ST": (231, 76,  60),    # Red
    "SR": (155, 89,  182),   # Purple
}

# ─── Color Palette ────────────────────────────────────────────────────────────
BG_COLOR        = (15,  18,  30)   # Main window background (dark navy)
PANEL_BG        = (20,  25,  42)   # Panel backgrounds
PANEL_BORDER    = (45,  55,  85)   # Panel border color
PLOT_BG         = (18,  22,  38)   # Plot area background

WHITE           = (230, 235, 245)
LIGHT_GRAY      = (160, 170, 190)
GRAY            = (90,  100, 120)
DARK_GRAY       = (38,  45,  62)

GREEN           = (46,  204, 113)
RED             = (231, 76,  60)
BLUE            = (52,  152, 219)
ORANGE          = (230, 126, 34)
PURPLE          = (155, 89,  182)
YELLOW          = (241, 196, 15)
CYAN            = (26,  188, 156)
PINK            = (255, 105, 180)

LINK_PRIMARY_OK  = (46,  204, 113)   # PT→PR when SINR_p >= threshold (green)
LINK_PRIMARY_BAD = (231, 76,  60)    # PT→PR when SINR_p < threshold (red)
LINK_SU          = (52,  152, 219)   # ST→SR desired link (blue)
LINK_INTERF_SP   = (231, 76,  60)    # ST→PR interference (red dashed)
LINK_INTERF_PS   = (230, 126, 34)    # PT→SR interference (orange dashed)
