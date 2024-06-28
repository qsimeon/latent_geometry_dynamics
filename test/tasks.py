import gym
import numpy as np
import random
from collections import deque
import pygame
from gymnasium.envs.registration import register


class ObjectReversalLearningTask(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, max_trials=float("inf"), criterion_trials=5, criterion=0.9, render_mode=None):
        """
        Initialize the ObjectReversalLearningTask environment.

        Parameters:
        - max_trials (int): The maximum number of trials before the environment is reset.
        - criterion_trials (int): The number of trials to consider when checking the performance criterion.
        - criterion (float): The performance threshold for context reversal.
        - render_mode (str): The mode for rendering the environment. Options include "human" and "rgb_array".
        """
        super().__init__()

        assert max_trials > 0
        self.max_trials = max_trials
        self.criterion_trials = criterion_trials
        self.criterion = criterion

        self.num_stimuli = 40
        self.num_contexts = 2
        self.observation_space = gym.spaces.MultiDiscrete([self.num_stimuli, self.num_contexts])
        self.action_space = gym.spaces.Discrete(2)
        self.stimulus_data = self.generate_stimulus_data()

        self.trial_counter = 0
        self.context = 0
        self.stimulus_index = 0
        # Use a deque for the performance window with a maxlen of criterion_trials
        self.performance_window = deque(maxlen=criterion_trials)

        self.last_observation = self._get_obs()
        self.last_action = self.action_space.sample()
        self.last_correct_action = self.last_action
        self.last_reward = 0

        self.reset()

        # related to rendering / PyGame
        self.window_size = 512
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        """
        Retrieve the current observation of the environment.

        Returns:
        - observation (int): The index of the current stimulus.
        """
        return self.stimulus_index

    def _get_info(self):
        """
        Retrieve a dictionary containing detailed information about the current state of the environment.

        Returns:
        - info (dict): A dictionary containing details such as trial number, context, action,
                      correct action, hit, observation, and reward.
        """
        info = {
            "trial": self.trial_counter,
            "context": self.context,
            "action": self.last_action,
            "correct_action": self.last_correct_action,
            "hit": int(self.last_action == self.last_correct_action),
            "observation": self.last_observation,
            "reward": self.last_reward,
        }
        return info


    def generate_stimulus_data(self):
        """
        Generate the stimulus data, including shapes, colors, and correct actions.

        Returns:
        - stimuli (list): A list of tuples containing correct action and shape-color combinations for two stimuli.
        """
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), (128, 0, 128)] # red, blue, green, yellow, purple
        shapes = ['circle', 'square', 'triangle']
        stimuli = []
        for i in range(self.num_stimuli):
            # using `random`
            shape1, shape2 = random.sample(shapes, 2)
            # print(f"shapes of stimuli pair %s:"%(i+1), shape1, shape2)
            color1, color2 = random.sample(colors, 2)
            # print(f"colors of stimuli pair %s:"%(i+1), color1, color2)
            correct_action = self.action_space.sample()
            stimuli.append((correct_action, (shape1, color1), (shape2, color2)))
        return stimuli


    def reset(self, seed=None):
        """
        Reset the environment to its initial state.

        Returns:
        - observation (int): The initial observation after resetting the environment.
        - info (dict): A dictionary containing detailed information about the initial state of the environment.
        """
        super().reset(seed=seed)
        self.trial_counter = 0
        self.context = 0
        # self.correct_action = 0  # Remove this line if `correct_action` is not used
        self.performance_window = deque([0] * self.criterion_trials, maxlen=self.criterion_trials)  # Updated to a deque
        self.stimulus_index = 0
        self.last_action = self.action_space.sample()
        self.last_correct_action = self.last_action
        self.last_reward = 0
        self.last_observation = self._get_obs()
        # return the agent observation and info
        return self._get_obs(), self._get_info()


    def step(self, action):
        """
        Take a step in the environment with the given action.

        Parameters:
        - action: The action to be taken.

        Returns:
        - observation: The next state.
        - reward: The reward obtained.
        - terminated: Whether the episode has ended.
        - truncated: Whether the step was truncated.
        - info: A dictionary containing additional information.
        """
        self.update_state(action)
        terminated, truncated = self.check_termination_conditions()

        if self.render_mode == "human":
            self._render()

        return self._get_obs(), self.last_reward, terminated, truncated, self._get_info()


    def update_state(self, action):
        self.last_action = action
        self.trial_counter += 1
        self.stimulus_index = (self.stimulus_index + 1) % self.num_stimuli  # Ensured looping over stimuli

        correct_action, _, _ = self.stimulus_data[self.stimulus_index]
        self.last_reward = int(action == correct_action)

        # Since it's a deque with maxlen, no need to pop the first element, it will automatically remove items
        # exceeding the maxlen
        self.performance_window.append(self.last_reward)

        self.last_observation = self._get_obs()
        self.last_correct_action = correct_action

        if self.trial_counter % self.num_stimuli == 0 and np.mean(self.performance_window) >= self.criterion:
            self.perform_context_reversal()


    def perform_context_reversal(self):
        self.context = 1 - self.context
        self.performance_window = deque([0] * self.criterion_trials, maxlen=self.criterion_trials)  # Keep it as a deque
        self.stimulus_data = [(1 - ca, s1, s2) for ca, s1, s2 in self.stimulus_data]  # Simplified reversal


    def check_termination_conditions(self):
        terminated = self.trial_counter >= self.max_trials
        truncated = False  # You can fill in the condition as per your requirements
        return terminated, truncated  # Consider removing the self.reset() call here


    def draw_shape(self, surface, shape, color, position):
        """
        Draw a shape on the given surface at the specified position.

        Parameters:
        - surface (pygame.Surface): The surface to draw the shape on.
        - shape (str): The shape to be drawn. Options include "circle", "square", and "triangle".
        - color (tuple): The RGB color value for the shape.
        - position (tuple): The coordinates where the shape will be drawn.
        """
        x, y = position
        shape_drawings = {
            "circle": lambda: pygame.draw.circle(surface, color, position, 50),
            "square": lambda: pygame.draw.rect(surface, color, (x - 50, y - 50, 100, 100)),
            "triangle": lambda: pygame.draw.polygon(surface, color, [(x, y - 50), (x - 50, y + 50), (x + 50, y + 50)])
        }
        shape_drawings[shape]()  # Call the appropriate shape drawing function based on the shape name


    def initialize_pygame(self):
        """
        Initialize pygame only if it’s not already initialized.
        """
        if self.render_mode == "human" and not pygame.display.get_init():
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()


    def _render_frame(self):
        """
        Render a frame of the environment.

        Returns:
        - image (array): An array representation of the rendered frame if the render mode is "rgb_array".
        """
        # Initialize PyGame
        self.initialize_pygame()

        # Create a surface
        surface = pygame.Surface((self.window_size, self.window_size)) if self.render_mode == 'rgb_array' else self.window
        surface.fill((255, 255, 255))

        # Retrieve the object pair (i.e. shape, and color data)
        _, (shape1, color1), (shape2, color2) = self.stimulus_data[self.last_observation]

        # Draw the two objects
        self.draw_shape(surface, shape1, color1, (self.window_size // 3, self.window_size // 2))
        self.draw_shape(surface, shape2, color2, (2 * self.window_size // 3, self.window_size // 2))

        # Code for showing correct action
        arrow_y = self.window_size // 2 + 60
        arrow_color = (0, 255, 0) if self.last_reward else (255, 0, 0) # Using reward value to decide the arrow color

        if self.last_action == 0:
            pygame.draw.polygon(surface, arrow_color, [(self.window_size // 3, arrow_y),
                                                      (self.window_size // 3 - 20, arrow_y + 20),
                                                      (self.window_size // 3 + 20, arrow_y + 20)])
        else:
            pygame.draw.polygon(surface, arrow_color, [(2 * self.window_size // 3, arrow_y),
                                                      (2 * self.window_size // 3 - 20, arrow_y + 20),
                                                      (2 * self.window_size // 3 + 20, arrow_y + 20)])

        if self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == 'rgb_array':
            return pygame.surfarray.array3d(surface)


    def render(self):
        """
        Render the environment according to the specified render mode.

        Returns:
        - image (array): An array representation of the rendered environment if the render mode is "rgb_array".
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def close(self):
        """
        Close the rendering window and clean up resources.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


class SerialReversalLearningTask(gym.Env):
    """A serial reversal learning task environment following OpenAI Gym's style."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    HOLD = 0
    RELEASE = 1
    NEUTRAL = 0
    POSITIVE = 1

    def __init__(self, max_trials=float("inf"), switch_every=(50, 71), render_mode=None):
        """Initialize the Serial Reversal Learning Task environment.

        Args:
        - max_trials: Maximum number of trials before the environment is reset.
        - switch_every: Range within which the context can switch.
        - render_mode: Mode to render the environment in ("human" or "rgb_array").
        """
        super().__init__()
        self.initialize_environment(max_trials, switch_every)
        self.initialize_rendering(render_mode)

    def initialize_environment(self, max_trials, switch_every):
        """Set up the environment's variables and spaces.

        Args:
            max_trials: The maximum number of trials before resetting.
            switch_every: The range of trials within which the context can switch.
        """
        assert max_trials > 0, "Max trials should be greater than 0."
        self.max_trials = max_trials
        self.switch_every = switch_every

        self.context_switch_trials = []
        self.next_switch = None
        self.observation_space = gym.spaces.Discrete(4)
        self.action_space = gym.spaces.Discrete(2)
        self._trial = 0
        self._context = 0
        self._contexts = [
            [(self.RELEASE, self.POSITIVE), (self.HOLD, self.POSITIVE),
             (self.RELEASE, self.NEUTRAL), (self.HOLD, self.NEUTRAL)],
            [(self.HOLD, self.NEUTRAL), (self.RELEASE, self.POSITIVE),
             (self.HOLD, self.POSITIVE), (self.RELEASE, self.NEUTRAL)]
        ]
        self.stimulus_colors = [
            (0, 195, 255),
            (255, 0, 255),
            (34, 139, 34),
            (255, 165, 0)
        ]

        self._observation = self.observation_space.sample()
        self._correct_action, self._reinforcer = self._contexts[self._context][self._observation]
        self._last_observation = self._observation
        self._last_action = self.action_space.sample()
        self._last_reward = 0

    def initialize_rendering(self, render_mode):
        """Set up the rendering settings.

        Args:
            render_mode: The mode to render the environment in.
        """
        self.window_size = 512
        assert render_mode is None or render_mode in self.metadata["render_modes"], f"Invalid render_mode {render_mode}"
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self, seed=None):
        """Reset the environment to its initial state.

        Args:
            seed: The seed for the random number generator.

        Returns:
            A tuple containing the initial observation and information dictionary.
        """
        super().reset(seed=seed)
        self._trial = 0
        self._context = 0
        self._observation = self.observation_space.sample()
        self.update_task_condition()
        self.context_switch_trials = []
        self.next_switch = np.random.randint(*self.switch_every)
        self._last_observation = self._observation
        self._last_action = self.action_space.sample()
        self._last_reward = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        """Execute one time step within the environment.

        Args:
        - action: The action taken by the agent.

        Returns:
        - A tuple of the new observation, reward, termination flag, truncation flag, and an info dictionary.
        """
        self._trial += 1
        self.check_context_switch()

        terminated, truncated = self._trial >= self.max_trials, False
        self.update_task_condition()

        reward = (action == self._correct_action) * self._reinforcer
        self._last_observation, self._last_action, self._last_reward = self._observation, action, reward

        if self.render_mode == "human":
            self._render_frame()

        if action == self._correct_action:
            self._observation = self.observation_space.sample()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def check_context_switch(self):
        """Check if it's time to switch the context and do so if necessary."""
        if self._trial == self.next_switch:
            self._context = 1 - self._context
            self.context_switch_trials.append(self._trial)
            self.next_switch += np.random.randint(*self.switch_every)

    def check_termination_conditions(self):
        """Check if the termination conditions are met.

        Returns:
            A boolean indicating whether the termination conditions are met.
        """
        truncated = False  # You can fill in the condition as per your requirements
        terminated = self._trial >= self.max_trials
        return terminated, truncated

    def update_task_condition(self):
        """Update the task condition based on the current context and observation."""
        self._correct_action, self._reinforcer = self._contexts[self._context][self._observation]

    def _get_obs(self):
        """Get the current observation.

        Returns:
            The current observation.
        """
        return self._observation

    def _get_info(self):
        """Get the current information dictionary.

        Returns:
            A dictionary containing current information about the environment state.
        """
        info = {
            "observation": self._last_observation,
            "reinforcer": self._reinforcer,
            "correct_action": self._correct_action,
            "reward": self._last_reward,
            "action": self._last_action,
            "trial": self._trial,
            "context": self._context,
            "hit": int(self._last_action == self._correct_action),
            "next_switch": self.next_switch
        }
        return info

    def _render_frame(self):
        """Render the current state of the environment on the screen or as an array."""
        pygame.font.init()  # initialize the font system unconditionally

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Get stimulus color
        stimulus_color = self.stimulus_colors[self._last_observation]

        # Drawing the stimulus at the center with 2x size
        stimulus_rect = pygame.Rect((self.window_size - 112)//2, (self.window_size - 112)//2, 112, 112)
        pygame.draw.rect(canvas, stimulus_color, stimulus_rect)

        # Drawing the R and H for Release and Hold
        font = pygame.font.Font(None, 72)
        font.set_bold(True)
        r_text = font.render('R', True, (0, 0, 0))
        h_text = font.render('H', True, (0, 0, 0))
        r_position = (self.window_size // 4 - r_text.get_width() // 2, 3 * self.window_size // 4 - r_text.get_height() // 2)
        h_position = (3 * self.window_size // 4 - h_text.get_width() // 2, 3 * self.window_size // 4 - h_text.get_height() // 2)
        canvas.blit(r_text, r_position)
        canvas.blit(h_text, h_position)

        # Circle the action the agent took, with color based on correctness
        if self._last_action is not None:
            color = (0, 255, 0) if self._last_action == self._correct_action else (255, 0, 0)
            if self._last_action == 0:  # hold
                pygame.draw.circle(canvas, color, (h_position[0] + h_text.get_width() // 2, h_position[1] + h_text.get_height() // 2), 40, 5)
            else:  # release
                pygame.draw.circle(canvas, color, (r_position[0] + r_text.get_width() // 2, r_position[1] + r_text.get_height() // 2), 40, 5)

        # Drawing smiley or straight face based on reward with 2x size
        emoji_position = (self.window_size // 2, 7 * self.window_size // 8)
        pygame.draw.circle(canvas, (255, 255, 0), emoji_position, 40)
        pygame.draw.circle(canvas, (0, 0, 0), (emoji_position[0] - 10, emoji_position[1] - 10), 10) # Eyes
        pygame.draw.circle(canvas, (0, 0, 0), (emoji_position[0] + 10, emoji_position[1] - 10), 10) # Eyes
        if self._last_reward > 0:
            pygame.draw.arc(canvas, (0, 0, 0), (emoji_position[0] - 20, emoji_position[1], 40, 20), math.pi, 2*math.pi, 6)
        else:
            pygame.draw.line(canvas, (0, 0, 0), (emoji_position[0] - 20, emoji_position[1] + 5), (emoji_position[0] + 20, emoji_position[1] + 5), 6)

        # Do the rendering
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def render(self):
        """Render the environment based on the selected render mode.

        Returns:
            An array representing the rendered environment if render_mode is "rgb_array".
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        """Close and clean up the rendering window."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
