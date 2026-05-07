import random
from helpers.env import SlipperyGridWorld
from helpers.env import ACTIONS
from typing import Optional, Tuple



class ModifiedSlipperyGridWorld(SlipperyGridWorld):
    def __init__(self, *args, walls=None, terminal_traps=None, trap_rewards=None, additional_goals=None, additional_goal_rewards=None, **kwargs):
        """
        Initialize a modified slippery grid world with walls, traps, and multiple goals.

        Extends SlipperyGridWorld to support:
        - Walls: States the agent cannot enter (movement into wall = stay in place)
        - Terminal traps: States that end episode with negative reward
        - Multiple goals: Multiple terminal states with positive rewards

        Args:
            *args: Positional arguments passed to SlipperyGridWorld.__init__
                (rows, cols, start, goal, slip_prob, step_reward, goal_reward, max_steps, seed)
            walls (list of tuple): List of (row, col) positions that block movement.
                Agent stays in current position when attempting to move into a wall.
                Example: [(1, 1), (2, 2), (3, 3)]
            terminal_traps (list of tuple): List of (row, col) positions for trap states.
                When agent lands on a trap, episode terminates with negative reward.
                Must be same length as trap_rewards if provided.
                Example: [(2, 1), (1, 3)]
            trap_rewards (list of float): Reward values for each trap in terminal_traps.
                Index-matched to terminal_traps list (trap_rewards[i] corresponds to terminal_traps[i]).
                Must be same length as terminal_traps if provided.
                Example: [-10, -20]  # First trap gives -10, second gives -20
            additional_goals (list of tuple): List of (row, col) positions for goal states.
                When agent lands on a goal, episode terminates with positive reward.
                If not provided, uses single goal from parent class.
                Must be same length as goal_rewards if provided.
                Example: [(5, 5), (0, 5), (5, 0)]
            additional_goal_rewards (list of float): Reward values for each goal in goals.
                Index-matched to goals list (goal_rewards[i] corresponds to goals[i]).
                Must be same length as goals if provided.
                Example: [100, 30, 50]  # First goal gives 100, second gives 30, third gives 50
        """
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

        self.main_goal = self.row_column_to_state(*self.goal_row_column)

        self.goals = set(
            self.row_column_to_state(r, c)
            for r, c in (additional_goals or [])
        )
        self.goals.add(self.main_goal)
        self.goal_rewards = {}
        if additional_goals is not None and additional_goal_rewards is not None:
            for (r, c), reward in zip(additional_goals, additional_goal_rewards):
                state = self.row_column_to_state(r, c)
                self.goal_rewards[state] = reward
        self.goal_rewards[self.main_goal] = self.goal_reward

        self.original_terminal_traps = self.terminal_traps
        self.original_trap_rewards = self.trap_rewards

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

        # self.move_traps()

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

    def reset(self, start: Optional[Tuple[int, int]] = None) -> int:
        """Reset environment to start state specified (optional).
            Resets moving elements positions

        Args:
            start (Optional[Tuple[int, int]], optional): If not specified,
            takes start state from environment initialization.
            Defaults to None.

        Returns:
            int: Reset agent's state.
        """
        self.terminal_traps = self.original_terminal_traps
        self.trap_rewards = self.original_trap_rewards

        return super().reset(start)


    def move_traps(self):
        new_traps = set()
        new_trap_rewards = {}

        for trap_pos in self.original_terminal_traps:
            r, c = self.state_to_row_column(trap_pos)

            directions = [(0, 1), (0, -1), (1, 0), (-1, 0),(0,0)]
            random.shuffle(directions)

            moved = False
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                ns = self.row_column_to_state(nr, nc)

                if self._in_bounds(nr, nc) and not self.is_terminal_state(ns):
                    new_traps.add(ns)
                    new_trap_rewards[ns] = self.original_trap_rewards[trap_pos]
                    moved = True
                    break

            if not moved:
                new_traps.add(trap_pos)
                new_trap_rewards[trap_pos] = self.original_trap_rewards[trap_pos]

        self.terminal_traps = new_traps
        self.trap_rewards = new_trap_rewards
