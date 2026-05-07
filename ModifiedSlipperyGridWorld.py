from helpers.env import SlipperyGridWorld


class ModifiedSlipperyGridWorld(SlipperyGridWorld):
    def __init__(self, *args, obstacles=None, traps=None, trap_reward=-10, **kwargs):
        super().__init__(*args, **kwargs)
        self.obstacles = set(obstacles or [])
        self.traps = set(traps or [])
        self.trap_reward = trap_reward

    def get_transition_distribution(self, s: int, a: int) -> list[tuple[float, int]]:
        original_distribution = super().get_transition_distribution(s, a)

        new_distribution = {}

        for p, sn in original_distribution:
            next_state = sn // self.cols, sn % self.cols

            if next_state in self.obstacles:
                corrected_sn = s
            else:
                corrected_sn = sn

            new_distribution[corrected_sn] = new_distribution.get(corrected_sn, 0.0) + p

        return [(p, sn) for sn, p in new_distribution.items()]

    def reward(self, s: int, a: int, sn: int) -> float:
        next_state = sn // self.cols, sn % self.cols

        if next_state in self.traps:
            return self.trap_reward

        return super().reward(s, a, sn)

    def is_terminal_state(self, s: int) -> bool:
        state = s // self.cols, s % self.cols

        if state in self.traps:
            return True

        return super().is_terminal_state(s)