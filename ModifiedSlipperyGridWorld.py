from helpers.env import SlipperyGridWorld
from helpers.env import ACTIONS


class ModifiedSlipperyGridWorld(SlipperyGridWorld):
    def __init__(self, *args, walls=None, terminal_traps=None, trap_rewards=None, goals=None, goal_rewards=None,**kwargs):
        super().__init__(*args, **kwargs)
        self.walls =  set(
            self.row_column_to_state(r, c)
            for r, c in (walls or [])
        )
        self.terminal_traps = set(
            self.row_column_to_state(r, c)
            for r, c in (terminal_traps or [])
        )
        self.trap_rewards = {}
        if terminal_traps is not None and trap_rewards is not None:
            for (r, c), reward in zip(terminal_traps, trap_rewards):
                state = self.row_column_to_state(r, c)
                self.trap_rewards[state] = reward

        self.goals = set(
            self.row_column_to_state(r, c)
            for r, c in (goals or [])
        )
        self.goal_rewards = {}
        if goals is not None and goal_rewards is not None:
            for (r, c), reward in zip(goals, goal_rewards):
                state = self.row_column_to_state(r, c)
                self.goal_rewards[state] = reward

    def get_transition_distribution(self, s: int, a: int) -> list[tuple[float, int]]:
        """Returns env transition probability distribution for given (s,a) and respective next states (s').
            Because the environment is slippery, attempting one action may lead to several possible next states.

        Args:
            s (int): Current state (s).
            a (int): Action to attempt (a).

        Returns:
            List of (probability, next_state) pairs.
        """
        original_distribution = super().get_transition_distribution(s, a)

        new_distribution = {}

        for p, sn in original_distribution:
            if self.is_wall(sn):
                corrected_sn = s
            else:
                corrected_sn = sn

            new_distribution[corrected_sn] = new_distribution.get(corrected_sn, 0.0) + p

        return [(p, sn) for sn, p in new_distribution.items()]

    def reward(self, s: int, a: int, sn: int) -> float:
        """Return the reward R(s, a, s') for a transition.

        In this simplified GridWorld, reward depends only on the next state

        Args:
            s (int): State for which the reward should be retrieved.
            a (int): Attempted action.
            sn (int): Next state after action in state.

        Returns:
            float: reward from the environment.
        """
        if self.is_terminal_trap(sn):
            return self.trap_rewards[sn]
        if self.is_goal(sn):
            return self.goal_rewards[sn]

        return self.step_reward

    def step(self, action: int):
        """Perform one step in the environment.

        Args:
            action (int): Action to perform [0, 1, 2, 3].

        Returns:
            Next state, reward, flag done, info dictionary
        """
        assert action in ACTIONS, f"Invalid action {action}. Use 0=U,1=R,2=D,3=L."
        self._steps += 1

        intended = action
        executed = self._sample_action_with_slip(intended)

        r, c = self._agent_row_column
        nr, nc = self._apply_action(r, c, executed)

        ns = self.row_column_to_state(nr,nc)
        if self.is_wall(ns):
            nr,nc = r,c
            ns = self.row_column_to_state(r,c)

        self._agent_row_column = (nr, nc)

        done = self.is_terminal_state(ns)
        if self.max_steps is not None and self._steps >= self.max_steps:
            done = True

        reward = self.reward(self.row_column_to_state(r,c),intended,ns)

        info = {"intended_action": intended, "executed_action": executed, "steps": self._steps}
        return self.row_column_to_state(*self._agent_row_column), reward, done, info

    def is_terminal_state(self, s: int) -> bool:
        if self.is_terminal_trap(s):
            return True
        if self.is_goal(s):
            return True

        return super().is_terminal_state(s)

    def is_wall(self, state: int) -> bool:
        return state in self.walls

    def is_terminal_trap(self, state: int) -> bool:
        return state in self.terminal_traps

    def is_goal(self, state: int) -> bool:
        return state in self.goals