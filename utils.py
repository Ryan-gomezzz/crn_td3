# =============================================================================
# utils.py — Helper utilities: rolling stats, exploration noise, logger
# =============================================================================

import numpy as np
from collections import deque


class RollingStats:
    """Maintains a fixed-size window of scalar values for computing rolling mean."""

    def __init__(self, window: int = 100):
        self._window = window
        self._data: deque[float] = deque(maxlen=window)

    def push(self, value: float) -> None:
        self._data.append(float(value))

    def mean(self) -> float:
        if len(self._data) == 0:
            return 0.0
        return float(np.mean(self._data))

    def values(self) -> list:
        """Returns a list copy of the internal deque."""
        return list(self._data)

    def __len__(self) -> int:
        return len(self._data)


class ExplorationNoise:
    """
    Gaussian exploration noise with linear decay from std_start to std_end
    over decay_steps episodes. After decay_steps the std stays at std_end.
    """

    def __init__(self, std_start: float, std_end: float, decay_steps: int):
        self._std_start  = std_start
        self._std_end    = std_end
        self._decay_steps = max(1, decay_steps)
        self._episode    = 0
        self._current_std = std_start

    def sample(self, shape=(1,)) -> np.ndarray:
        """Draw Gaussian noise with the current std."""
        return np.random.normal(0.0, self._current_std, size=shape)

    def step(self) -> None:
        """Call once at the end of each episode to decay the noise std."""
        self._episode += 1
        frac = min(1.0, self._episode / self._decay_steps)
        self._current_std = self._std_start + frac * (self._std_end - self._std_start)

    @property
    def current_std(self) -> float:
        return self._current_std


class Logger:
    """
    Prints formatted training log rows to console.
    Optionally also writes to a CSV file.
    """

    HEADER = (
        f"{'Episode':>8} | {'Steps':>6} | {'Reward':>9} | {'Avg100':>9} | "
        f"{'SU_Rate':>8} | {'PU_SINR':>8} | {'Power':>7} | {'Buffer':>8}"
    )

    def __init__(self, log_to_file: bool = False, filepath: str = "training_log.csv"):
        self._log_to_file = log_to_file
        self._filepath    = filepath
        self._header_printed = False

        if log_to_file:
            with open(filepath, "w") as f:
                f.write("episode,steps,reward,avg100,su_rate,pu_sinr,power,buffer_size\n")

    def log(
        self,
        episode: int,
        steps: int,
        reward: float,
        avg100: float,
        su_rate: float,
        pu_sinr: float,
        power: float,
        buffer_size: int,
    ) -> None:
        # Print header every 20 episodes for readability
        if not self._header_printed or episode % 20 == 0:
            print("\n" + self.HEADER)
            print("-" * len(self.HEADER))
            self._header_printed = True

        row = (
            f"{episode:>8} | {steps:>6} | {reward:>9.4f} | {avg100:>9.4f} | "
            f"{su_rate:>8.4f} | {pu_sinr:>8.4f} | {power:>7.4f} | {buffer_size:>8}"
        )
        print(row)

        if self._log_to_file:
            with open(self._filepath, "a") as f:
                f.write(
                    f"{episode},{steps},{reward:.6f},{avg100:.6f},"
                    f"{su_rate:.6f},{pu_sinr:.6f},{power:.6f},{buffer_size}\n"
                )


def training_status(episode: int, buffer_size: int, avg100: float) -> str:
    """Returns a human-readable training phase string for the GUI."""
    from config import MIN_SAMPLES, TRAINING_EPISODES
    if buffer_size < MIN_SAMPLES:
        return "Exploring"
    elif episode < TRAINING_EPISODES * 0.2:
        return "Learning"
    elif avg100 > 0.5:
        return "Converging"
    else:
        return "Training"
