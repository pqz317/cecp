from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from ..spaces import Observation


class CoopForagingVisualizer:
    def __init__(self, env: Any) -> None:
        self.env = env

    def export_gif(
        self,
        states: Sequence[Any],
        agent_id: int = 0,
        filename: str = "goal_cycle.gif",
        frame_duration_ms: int = 500,
    ) -> None:
        frames = []
        agent_ret = 0.0
        for env_step, state in enumerate(states):
            agent_ret += float(state.agent_state.last_step_reward[agent_id])
            frame = self._render_state_frame(
                state=state,
                env_step=env_step,
                total_steps=len(states),
                cumulative_return=agent_ret,
                agent_id=agent_id,
                title="Environment",
            )
            frames.append(frame)

        self._save_gif_frames(frames, filename=filename, frame_duration_ms=frame_duration_ms)
        tqdm.write(f"\nVisualizing: GIF successfully exported to {filename}\n")

    def export_predictive_comparison_gif(
        self,
        states: Sequence[Any],
        predicted_next_obs_seq: Sequence[Observation],
        agent_id: int = 0,
        filename: str = "goal_cycle_predictive_comparison.gif",
        frame_duration_ms: int = 500,
    ) -> None:
        num_frames = min(len(states), len(predicted_next_obs_seq))
        if num_frames == 0:
            raise ValueError("states and predicted_next_obs_seq must both be non-empty")

        frames = []
        agent_ret = 0.0
        for env_step in range(num_frames):
            state = states[env_step]
            pred_obs = predicted_next_obs_seq[env_step]
            agent_ret += float(state.agent_state.last_step_reward[agent_id])

            frame_true = self._render_state_frame(
                state=state,
                env_step=env_step,
                total_steps=num_frames,
                cumulative_return=agent_ret,
                agent_id=agent_id,
                title="Environment",
            )
            frame_pred = self._render_predicted_obs_frame(
                pred_obs=pred_obs,
                env_step=env_step,
                total_steps=num_frames,
                agent_id=agent_id,
                title=f"Predicted next obs (agent {agent_id})",
            )

            pad = 16
            canvas = Image.new("RGB", (frame_true.width + frame_pred.width + pad, frame_true.height), (255, 255, 255))
            canvas.paste(frame_true, (0, 0))
            canvas.paste(frame_pred, (frame_true.width + pad, 0))
            frames.append(canvas)

        self._save_gif_frames(frames, filename=filename, frame_duration_ms=frame_duration_ms)
        tqdm.write(f"\nVisualizing: predictive comparison GIF successfully exported to {filename}\n")

    def _render_state_frame(
        self,
        state: Any,
        env_step: int,
        total_steps: int,
        cumulative_return: float,
        agent_id: int,
        title: str,
    ) -> Image.Image:
        tile_size = 80
        width = self.env.grid_size * tile_size
        cell_font = ImageFont.load_default(50)
        hud_font = ImageFont.load_default(50)

        color_gray, color_black, color_green, color_red = (60, 60, 60), (20, 20, 20), (0, 255, 0), (255, 0, 0)
        wall_mask = np.array(state.wall_map)
        base_layer = self._build_base_layer(wall_mask=wall_mask, tile_size=tile_size, wall_color=color_gray)

        bg_map = np.full((self.env.grid_size, self.env.grid_size), 255.0)
        pixel_bg = np.repeat(np.repeat(bg_map, tile_size, axis=0), tile_size, axis=1).astype(np.uint8)
        img = Image.fromarray(np.stack([pixel_bg] * 3, axis=-1), mode="RGB")
        img.paste(base_layer, (0, 0), base_layer)
        draw = ImageDraw.Draw(img)

        for goal_idx in range(self.env.n_goal):
            goal_i, goal_j = int(state.goal_pos[goal_idx, 0]), int(state.goal_pos[goal_idx, 1])
            draw.rectangle([
                goal_j * tile_size + 10,
                goal_i * tile_size + 10,
                (goal_j + 1) * tile_size - 10,
                (goal_i + 1) * tile_size - 10,
            ])

            text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdedfhijklmnopqrstuvwxyz"[goal_idx]
            bbox = draw.textbbox((0, 0), text, font=cell_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (goal_j * tile_size + (tile_size - tw) // 2, goal_i * tile_size + (tile_size - th) // 2 - 10),
                text,
                fill=(0, 0, 0),
                font=cell_font,
            )

        agent_colors = [color_green, color_red, color_black]
        for idx in range(self.env.n_agent):
            pos_i, pos_j = int(state.agent_state.pos[idx][0]), int(state.agent_state.pos[idx][1])
            rect = [pos_j * tile_size, pos_i * tile_size, (pos_j + 1) * tile_size, (pos_i + 1) * tile_size]
            draw.rectangle(rect, fill=agent_colors[idx % len(agent_colors)], outline=(0, 0, 0), width=2)

            text = str(idx)
            bbox = draw.textbbox((0, 0), text, font=cell_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((rect[0] + (tile_size - tw) // 2, rect[1] + (tile_size - th) // 2 - 10), text, fill=(255, 255, 255), font=cell_font)

        draw.text((16, 10), title, fill=(0, 0, 0), font=hud_font)
        step_text = f"Step: {env_step + 1} / {total_steps}"
        reward_text = f"Agent {agent_id} return: {cumulative_return:.2f}"
        t_bbox = draw.textbbox((0, 0), step_text, font=hud_font)
        t_w = t_bbox[2] - t_bbox[0]
        draw.text((width - t_w - 20, width - tile_size + 10), step_text, fill=(255, 255, 255), font=hud_font)

        r_bbox = draw.textbbox((0, 0), reward_text, font=hud_font)
        r_w = r_bbox[2] - r_bbox[0]
        draw.text((width - r_w - t_w - 30, width - tile_size + 10), reward_text, fill=(255, 255, 255), font=hud_font)
        return img

    def _render_predicted_obs_frame(
        self,
        pred_obs: Observation,
        env_step: int,
        total_steps: int,
        agent_id: int,
        title: str,
    ) -> Image.Image:
        tile_size = 80
        width = self.env.grid_size * tile_size
        cell_font = ImageFont.load_default(50)
        hud_font = ImageFont.load_default(50)

        color_gray, color_black, color_green, color_red = (60, 60, 60), (20, 20, 20), (0, 255, 0), (255, 0, 0)
        grid = np.array(pred_obs['grid'])
        if grid.ndim == 4:
            grid = grid[0]
        if grid.ndim != 3:
            raise ValueError(f"pred_obs['grid'] is expected to have shape [H, W, C] or [B, H, W, C], got {grid.shape}")

        wall = grid[..., 0]
        my_pos = grid[..., 1]
        goal = grid[..., 2:2 + self.env.n_goal]
        all_agent_pos = grid[..., 2 + self.env.n_goal:2 + self.env.n_goal + self.env.n_agent]

        wall_mask = wall > 0.5
        base_layer = self._build_base_layer(wall_mask=wall_mask, tile_size=tile_size, wall_color=color_gray)

        bg_map = np.full((self.env.grid_size, self.env.grid_size), 255.0)
        pixel_bg = np.repeat(np.repeat(bg_map, tile_size, axis=0), tile_size, axis=1).astype(np.uint8)
        img = Image.fromarray(np.stack([pixel_bg] * 3, axis=-1), mode="RGB")
        img.paste(base_layer, (0, 0), base_layer)
        draw = ImageDraw.Draw(img)

        for goal_idx in range(self.env.n_goal):
            goal_pos = np.unravel_index(np.argmax(goal[..., goal_idx]), goal[..., goal_idx].shape)
            goal_i, goal_j = int(goal_pos[0]), int(goal_pos[1])
            draw.rectangle([
                goal_j * tile_size + 10,
                goal_i * tile_size + 10,
                (goal_j + 1) * tile_size - 10,
                (goal_i + 1) * tile_size - 10,
            ])
            text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdedfhijklmnopqrstuvwxyz"[goal_idx]
            bbox = draw.textbbox((0, 0), text, font=cell_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (goal_j * tile_size + (tile_size - tw) // 2, goal_i * tile_size + (tile_size - th) // 2 - 10),
                text,
                fill=(0, 0, 0),
                font=cell_font,
            )

        pred_positions = []
        for idx in range(self.env.n_agent):
            channel = all_agent_pos[..., idx]
            pos = np.unravel_index(np.argmax(channel), channel.shape)
            pred_positions.append((int(pos[0]), int(pos[1])))
        my_pos_idx = np.unravel_index(np.argmax(my_pos), my_pos.shape)
        pred_positions[agent_id] = (int(my_pos_idx[0]), int(my_pos_idx[1]))

        agent_colors = [color_green, color_red, color_black]
        for idx, (pos_i, pos_j) in enumerate(pred_positions):
            rect = [pos_j * tile_size, pos_i * tile_size, (pos_j + 1) * tile_size, (pos_i + 1) * tile_size]
            draw.rectangle(rect, fill=agent_colors[idx % len(agent_colors)], outline=(0, 0, 0), width=2)
            text = str(idx)
            bbox = draw.textbbox((0, 0), text, font=cell_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((rect[0] + (tile_size - tw) // 2, rect[1] + (tile_size - th) // 2 - 10), text, fill=(255, 255, 255), font=cell_font)

        draw.text((16, 10), title, fill=(0, 0, 0), font=hud_font)
        step_text = f"Step: {env_step + 1} / {total_steps}"
        t_bbox = draw.textbbox((0, 0), step_text, font=hud_font)
        t_w = t_bbox[2] - t_bbox[0]
        draw.text((width - t_w - 20, width - tile_size + 10), step_text, fill=(255, 255, 255), font=hud_font)
        return img

    def _build_base_layer(self, wall_mask: np.ndarray, tile_size: int, wall_color: tuple[int, int, int]) -> Image.Image:
        width = self.env.grid_size * tile_size
        base_layer = Image.new("RGBA", (width, width), (0, 0, 0, 0))
        draw = ImageDraw.Draw(base_layer)
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                rect = [j * tile_size, i * tile_size, (j + 1) * tile_size, (i + 1) * tile_size]
                if wall_mask[i, j]:
                    draw.rectangle(rect, fill=(*wall_color, 255))
                draw.rectangle(rect, outline=(0, 0, 0, 255), width=1)
        return base_layer

    def _save_gif_frames(self, frames: Sequence[Image.Image], filename: str, frame_duration_ms: int = 500) -> None:
        sample_img = Image.fromarray(np.uint8(np.concatenate([np.array(frame) for frame in frames], axis=0)))
        global_palette = sample_img.convert("P", palette=Image.ADAPTIVE, colors=256)
        quantized = [frame.quantize(palette=global_palette) for frame in frames]
        quantized[0].save(filename, save_all=True, append_images=quantized[1:], duration=frame_duration_ms, loop=0, optimize=True)
