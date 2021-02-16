# coding=utf-8
# Copyright 2020 Anonymous.
#
"""Parallel implementation of Catch prey game """

# Import all required packages
import cv2
import numpy as np


class CatchPrey:
  """MARL environment where the predators have to collaborate to catch the pray."""

  def __init__(self, agent_nmbr: int, seed=None):
    """
    Args:
      seed: Optional integer. Seed for numpy's random number generator (RNG).
    """
    self._video = None
    self._make_video = False
    self._video_framerate = 20
    self._video_max_t = 20  #Time in seconds of the longest possible episode.
    self._video_size = (1280, 1100)
    self._video_codec = cv2.VideoWriter_fourcc(*'mp4v')
    self._video_ext = '.mp4'
    self._source_images_path = '../'
    self._rng = np.random.RandomState(seed)

    self._parallel_games = 1
    self._agent_nmbr = agent_nmbr

    #Game constants
    self._slow_movement_prob = 0.1
    self._nervous_movement_prob = 0.3
    self._run_distance = 10
    self._max_distance = 2 * self._run_distance
    self._catch_distance = 3
    self._init_distance = 2 * self._run_distance   #2*self._catch_distance  #
    self._dist_margin = 5
    self._pred_walk_speed = 1
    self._prey_walk_speed = 1
    self._prey_run_speed = 1.2 * self._prey_walk_speed
    self._angle_margin = 1.2 * 2 * np.pi / 4
    self._max_time = 75
    self._catch_reward = 1
    self._move_reward = - 1 / (agent_nmbr * self._max_time)
    self._num_actions = 5
    self._stay = 0
    self._forward = 1
    self._counter_clockwise = 2
    self._backward = 3
    self._clockwise = 4
    self._n_actions_per_agent = 5
    self._n_actions = self._n_actions_per_agent**agent_nmbr
    self._no_move = np.zeros((self._parallel_games, self._agent_nmbr))

    self._discount = np.ones(self._parallel_games, dtype=float)
    self._prey_pos = np.zeros((self._parallel_games, 1, 2), dtype=float)
    self._pred_pos = self._rng.rand(self._parallel_games, self._agent_nmbr, 2)
    self._old_pred_pos = np.copy(self._pred_pos)
    self._old_prey_pos = np.copy(self._prey_pos)

    self._time_elapsed = np.zeros(self._parallel_games, dtype=float)
    self._ep_return = np.zeros(self._parallel_games, dtype=float)

    self._offset = np.concatenate([[2 * a * self._num_actions, 2 * a * self._num_actions + 1] for a in range(agent_nmbr)], axis=0)

    # for debugging purposes.
    self._nmbr_configs = 3
    self._deb_times = np.array([0, 6, 30], dtype=float)
    self._deb_prey_pos = np.zeros((1, 2), dtype=float)
    self._deb_pred_pos = np.zeros((self._nmbr_configs, self._agent_nmbr, 2), dtype=float)
    angles = self._rng.rand(self._agent_nmbr) * 2 * np.pi/100
    self._deb_pred_pos[0, :, 0] = 17 * np.sin(angles)
    self._deb_pred_pos[0, :, 1] = 17 * np.cos(angles)

    self._deb_pred_pos[1, :, 0] = 11 * np.sin(angles)
    self._deb_pred_pos[1, :, 1] = 11 * np.cos(angles)

    angle_step = 2 * np.pi / self._agent_nmbr
    self._deb_pred_pos[2, :, 0] = 11 * np.sin(np.arange(self._agent_nmbr) * angle_step)
    self._deb_pred_pos[2, :, 1] = 11 * np.cos(np.arange(self._agent_nmbr) * angle_step)

  def reset(self, parallel_games: int):
    """Initialize parallel_games new games"""
    self._parallel_games = parallel_games
    self._no_move = np.zeros((self._parallel_games, self._agent_nmbr))
    self._ep_return = np.zeros(parallel_games, dtype=float)
    self._time_elapsed = np.zeros(parallel_games, dtype=float)
    self._prey_pos = np.zeros(shape=(parallel_games, 1, 2), dtype=float)

    angles = self._rng.rand(self._parallel_games, self._agent_nmbr) * 2 * np.pi
    dist = self._run_distance + (0.1 + 0.8*self._rng.rand(self._parallel_games, self._agent_nmbr, 1)) * (self._init_distance - self._run_distance)
    self._pred_pos = dist * np.stack([np.sin(angles), np.cos(angles)], axis=2)
    observation = self._get_observation()

    self._discount = np.ones(self._parallel_games, dtype=float)
    return observation, self._discount

  def step(self, actions):
    self._old_pred_pos = np.copy(self._pred_pos)
    self._old_prey_pos = np.copy(self._prey_pos)

    self._time_elapsed += self._discount
    deltas = self._pred_pos - self._prey_pos
    distance_to_prey = np.linalg.norm(deltas, axis=-1)
    angles = np.arctan2(deltas[:, :, 0], deltas[:, :, 1])
    angles = (2 * np.pi + angles) * (angles < 0) + angles * (angles > 0)  #Wrap to 2pi
    sorted_angles = np.sort(angles, axis=-1)

    # Move prey
    too_close = np.expand_dims(np.any(distance_to_prey < self._run_distance, axis=-1), axis=-1)
    rand_vec = np.expand_dims(self._rng.rand(self._parallel_games), axis=-1)
    rand_angles = np.expand_dims(self._rng.rand(self._parallel_games) * 2 * np.pi, axis=-1)
    escape_angles = np.diff(sorted_angles, append=np.expand_dims(2 * np.pi + sorted_angles[:, 0], axis=-1), axis=-1)
    angle_pos = np.expand_dims(np.argmax(escape_angles, axis=-1), axis=-1)
    max_angle = np.take_along_axis(escape_angles, angle_pos, axis=-1)
    dis_pos = np.expand_dims(np.argmin(distance_to_prey, axis=-1), axis=-1)
    min_distance = np.take_along_axis(distance_to_prey, dis_pos, axis=-1)

    hole_found = np.logical_and(too_close, max_angle >= self._angle_margin)  #There's a hole
    move_angle_escape = np.take_along_axis(sorted_angles, angle_pos, axis=-1) + max_angle / 2  # Found a hole to escape.

    too_close_no_hole = np.logical_and(np.expand_dims(np.amax(distance_to_prey, axis=-1), axis=-1) - min_distance > self._dist_margin, (max_angle < self._angle_margin))
    too_close_no_hole = np.logical_and(too_close, too_close_no_hole)
    move_angle_too_close = np.take_along_axis(angles, dis_pos, axis=-1) + np.pi  # No hole but one predator is much closer than the others

    scared_rand_move = np.logical_or(too_close_no_hole, hole_found)
    scared_rand_move = np.logical_and(np.logical_not(scared_rand_move), rand_vec < self._nervous_movement_prob)
    scared_rand_move = np.logical_and(too_close, scared_rand_move)

    move_fast = np.logical_or(scared_rand_move, np.logical_or(hole_found, too_close_no_hole))

    slow_rand_move = np.logical_and(np.logical_not(too_close), rand_vec < self._slow_movement_prob)

    move_angle = hole_found * move_angle_escape + too_close_no_hole * move_angle_too_close + np.logical_or(scared_rand_move, slow_rand_move) * rand_angles
    move_speed = np.expand_dims(self._discount, axis=-1) * (move_fast * self._prey_run_speed + slow_rand_move * self._prey_walk_speed)
    self._prey_pos += np.expand_dims(move_speed * np.concatenate([np.sin(move_angle), np.cos(move_angle)], axis=-1), axis=1)

    # Move predators
    deltas = - self._pred_walk_speed * np.divide(deltas, np.expand_dims(distance_to_prey, axis=-1))  #Normalize so that the steps are always of the same magnitude.

    reward = self._move_reward * np.count_nonzero(actions, axis=-1)  # Accumulate movement rewards

    #I multiply by the discount so that if the game is over no one moves (this is due to the parallel implementation).
    moves_dim1 = np.stack([self._no_move,
                           deltas[:, :, 0],
                           deltas[:, :, 1],
                           -deltas[:, :, 0],
                           -deltas[:, :, 1]], axis=-1)
    moves_dim2 = np.stack([self._no_move,
                           deltas[:, :, 1],
                           -deltas[:, :, 0],
                           -deltas[:, :, 1],
                           deltas[:, :, 0]], axis=-1)
    moves = np.concatenate([np.take_along_axis(moves_dim1, np.expand_dims(actions, axis=-1), axis=-1),
                            np.take_along_axis(moves_dim2, np.expand_dims(actions, axis=-1), axis=-1)], axis=-1)
    self._pred_pos += np.expand_dims(np.expand_dims(self._discount, axis=-1), axis=-1) * moves

    observations = self._get_observation()

    caught_prey = np.any(observations[:, :, 0]*self._run_distance < self._catch_distance, axis=-1)
    reward += np.logical_and(caught_prey, self._discount == 1) * self._catch_reward
    self._discount = self._discount * np.logical_not(caught_prey) * np.logical_not(self._time_elapsed >= self._max_time)

    self._ep_return += reward

    return reward, observations, self._discount

  def _get_state(self):
    return np.concatenate([self._prey_pos, self._pred_pos], axis=0)

  def action_space_size(self):
    return self._n_actions_per_agent

  def observation_shape(self):
    return self._get_observation(pred_pos=self._deb_pred_pos[0], prey_pos=self._deb_prey_pos[0])[0, 0, :].shape

  def get_debug_observations(self):
    observation_list = [self._get_observation(pred_pos=self._deb_pred_pos[k],
                                              prey_pos=self._deb_prey_pos[0],
                                              time_elapsed=self._deb_times[k]) for k in range(self._nmbr_configs)]

    return np.squeeze(np.concatenate(observation_list, axis=1))

  def _get_observation(self, pred_pos=None, prey_pos=None, time_elapsed=None):
    if pred_pos is not None:
      if len(pred_pos.shape) == 2:
        pred_pos = np.expand_dims(pred_pos, axis=0)
    else:
      pred_pos = self._pred_pos
    if prey_pos is not None:
      if len(prey_pos.shape) == 1:
        prey_pos = np.expand_dims(prey_pos, axis=0)
    else:
      prey_pos = self._prey_pos
    if time_elapsed is not None:
      if len(time_elapsed.shape) == 0:
        time_elapsed = np.expand_dims(time_elapsed, axis=0)
    else:
      time_elapsed = self._time_elapsed

    time_elapsed = np.expand_dims(time_elapsed, axis=1)
    parallel_games = pred_pos.shape[0]

    deltas = pred_pos - prey_pos
    distance_to_prey = np.linalg.norm(deltas, axis=-1)
    distance_to_front_agent = distance_to_prey - np.expand_dims(np.amin(distance_to_prey, axis=-1), axis=-1)
    distance_to_back_agent = np.expand_dims(np.amax(distance_to_prey, axis=-1), axis=-1) - distance_to_prey

    angles = np.arctan2(deltas[:, :, 0], deltas[:, :, 1])
    angles = (2 * np.pi + angles) * (angles < 0) + angles * (angles > 0)  # Wrap to 2pi
    arg_sorted_angles = np.argsort(angles, axis=-1)
    sorted_angles = np.take_along_axis(angles, arg_sorted_angles, axis=-1)
    angle_diffs = np.diff(sorted_angles, append=np.expand_dims(2 * np.pi + sorted_angles[:, 0], axis=-1), axis=-1)

    left_angle = np.take_along_axis(angle_diffs, np.argsort(arg_sorted_angles), axis=-1)
    right_angle = np.take_along_axis(np.roll(angle_diffs, 1, axis=-1), np.argsort(arg_sorted_angles, axis=-1), axis=-1)
    max_angle_diff = np.expand_dims(np.amax(angle_diffs, axis=-1), axis=1)

    return np.stack([distance_to_prey/self._run_distance,
                     distance_to_front_agent/self._dist_margin,
                     distance_to_back_agent/self._dist_margin,
                     right_angle/(2*np.pi),
                     left_angle/(2*np.pi),
                     np.ones((parallel_games, self._agent_nmbr)) * max_angle_diff / (2*np.pi),
                     np.ones((parallel_games, self._agent_nmbr)) * time_elapsed / self._max_time], axis=2)

  def max_return(self, gamma):
    return self._catch_reward + self._move_reward * self._agent_nmbr

  def min_return(self, gamma):
    if gamma < 1.0:
      return self._move_reward * self._agent_nmbr * (1 - gamma**self._max_time) / (1 - gamma)
    else:
      return self._move_reward * self._agent_nmbr * self._max_time

  def init_videos(self, path):
    self._video = []
    for n in range(self._parallel_games):
      if path[-1] == '/':
        file_name = path + '%d' % n + self._video_ext
      else:
        file_name = path + '/%d' % n + self._video_ext
      self._video.append(cv2.VideoWriter(file_name, self._video_codec, self._video_framerate, self._video_size))

  def add_frames(self, reward: float):

    path = self._source_images_path
    grass = cv2.imread(path+'grass.png')
    background = cv2.resize(grass[:500, :500, :], (1000, 1000))
    dim_back_y, dim_back_x, _ = background.shape
    center = np.round(np.array([dim_back_y, dim_back_x]) / 2)

    predator = cv2.imread(path+'cowboy.jpg')
    predator = predator[::-12, ::-12, :]
    predator_mask = np.uint8(np.repeat(np.expand_dims(np.average(predator, axis=2) == 255, axis=2), repeats=3, axis=2))
    y_pred, x_pred, _ = predator_mask.shape

    prey = cv2.imread(path+'bull2.jpeg')
    prey = prey[::-4, ::4, :]
    prey_mask = np.uint8(np.repeat(np.expand_dims(np.average(prey, axis=2) == 255, axis=2), repeats=3, axis=2))
    y_prey, x_prey, _ = prey_mask.shape

    step_size = np.average([y_prey, x_prey]) / self._catch_distance

    seconds_per_step = self._video_max_t / self._max_time
    total_frames = np.int8(np.ceil(seconds_per_step * self._video_framerate))
    guard = 1

    for p in range(self._parallel_games):
      new_positions = np.concatenate((self._prey_pos[p], self._pred_pos[p]), axis=0).T
      positions = np.concatenate((self._old_prey_pos[p], self._old_pred_pos[p]), axis=0).T
      delta_movement = step_size * (new_positions - positions) / float(total_frames)  #np.round(self._video_framerate / 2)
      positions = positions * step_size + np.repeat(np.expand_dims(center, axis=1), repeats=1+self._agent_nmbr, axis=1)
      game = np.zeros([dim_back_y * (2 * guard + 1), dim_back_x * (2 * guard + 1), 3], dtype=np.uint8)
      for f in range(total_frames):
        prey_coor = np.round(positions[:, 0]-np.array([x_prey, y_prey]) / 2)
        pred_y = np.round(positions[1, 1:] - y_prey / 2)
        pred_x = np.round(positions[0, 1:] - x_prey / 2)
        for n in range(3):
          game[:,:, n] = np.kron(np.ones([1 + 2 * guard, 1 + 2 * guard], dtype=np.uint8), background[:, :, n])

        y_range = np.array([dim_back_y * guard + prey_coor[1] - 1, dim_back_y * guard + prey_coor[1] + y_prey - 1], dtype=np.int16)
        x_range = np.array([dim_back_x * guard + prey_coor[0] - 1, dim_back_x * guard + prey_coor[0] + x_prey - 1], dtype=np.int16)
        game[y_range[0]:y_range[1], x_range[0]:x_range[1], :] = game[y_range[0]:y_range[1], x_range[0]:x_range[1], :] * prey_mask + (1 - prey_mask) * prey

        # Draw the limit of the run zone
        points = 2000
        for k in range(points):
          coord_x = np.int16(np.round(dim_back_x * guard + positions[0, 0] + step_size * self._run_distance * np.sin((k - 1) * 2 * np.pi / points)))
          coord_y = np.int16(np.round(dim_back_y * guard + positions[1, 0] + step_size * self._run_distance * np.cos((k - 1) * 2 * np.pi / points)))
          game[coord_y, coord_x,:] = np.array([255, 0,0], dtype=np.uint8)

        for k in range(self._agent_nmbr):
          if pred_y[k] < 1500 and pred_x[k] < 1500:  # Only draw if it's in the frame.
            y_range = np.array([dim_back_y * guard + pred_y[k] - 1, dim_back_y * guard + pred_y[k] + y_pred - 1], dtype=np.int16)
            x_range = np.array([dim_back_x * guard + pred_x[k] - 1, dim_back_x * guard + pred_x[k] + x_pred - 1], dtype=np.int16)
            game[y_range[0]:y_range[1], x_range[0]:x_range[1], :] = game[y_range[0]:y_range[1], x_range[0]:x_range[1], :] * predator_mask + (1 - predator_mask) * predator

        score = 255 * np.ones([100, 1280, 3], dtype=np.uint8)
        frame = np.concatenate((score, np.flipud(game[dim_back_y * guard: dim_back_y * (1 + guard), dim_back_x * guard - 140 : dim_back_x * (1 + guard) + 140,:])), axis=0)
        if f >= total_frames-1:
          text_str1 = 'Return:%0.3f' % self._ep_return[p]
          text_str2 = 'Time:%d' % self._time_elapsed[p]
        else:
          text_str1 = 'Return:%0.3f' % (self._ep_return[p] - reward[p])
          text_str2 = 'Time:%d' % (self._time_elapsed[p] - 1)

        cv2.putText(frame, text_str1, (10, 75), cv2.FONT_ITALIC, 2, (0, 0, 0), 1)
        cv2.putText(frame, text_str2, (900, 75), cv2.FONT_ITALIC, 2, (0, 0, 0), 1)
        self._video[p].write(frame)
        positions = positions + delta_movement

      if self._discount[p] == 0:
        self._video[p].release()  # Video finished.
