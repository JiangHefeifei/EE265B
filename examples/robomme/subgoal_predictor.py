from typing import Optional, Tuple, Any
from pathlib import Path

import os
import re
import shutil
import numpy as np
from env_runner import EnvRunner
from utils import EpisodeState, SUBGOAL_TYPES, TASK_WITH_VIDEO_DEMO

LONG_FIRST_ACTION_TASKS = [
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    
    "PickHighlight",
    "VideoRepick",
    
    "VideoPlaceButton",
    "VideoPlaceOrder",
    
    "MoveCube",
    "InsertPeg"
] # For Gemini only. Due to we found Gemini is very inconsistent for incremental video understanding, hard code to make it work better



class SubgoalPredictorBase:
    def __init__(
        self,
        args,
        save_dir: Path,
    ):
        self.args = args
        self.save_dir = save_dir
        self.video_buffer = []
        self.episode_dir: Optional[str] = None
        
        self.setup_api()

    def setup_api(self) -> None:
        pass

    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        self.env_name = env_runner.env_id
        self.episode_id = env_runner.episode_id
        self.task_goal = env_runner.task_goal
        self.env_runner = env_runner

    def step(self, epstate: EpisodeState) -> None:
        pass

    def maybe_extend_video(self, images: list) -> None:
        pass

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        # return (subgoal_str, has_api_error)
        raise NotImplementedError

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        pass


class NullSubgoalPredictor(SubgoalPredictorBase):
    def get_subgoal(self, *args, **kwargs) -> Tuple[Optional[str], bool]:
        return None, False
    

class GeminiSubgoalPredictor(SubgoalPredictorBase):
    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        from subgoal_prediction.gemini.api import GeminiModel
        from subgoal_prediction.gemini.prompts import (
            DEMO_TEXT_QUERY,
            IMAGE_TEXT_QUERY,
            VIDEO_TEXT_QUERY,
        )

        super().start_episode(epstate, env_runner)
        self.demo_text_query = DEMO_TEXT_QUERY
        self.image_text_query = IMAGE_TEXT_QUERY
        self.video_text_query = VIDEO_TEXT_QUERY
        self.api = GeminiModel(
            save_dir=os.path.join(self.save_dir, self.env_name, f"ep{self.episode_id}"),
            task_id=self.env_name,
            model_name=self.args.gemini_model_name,
            task_goal=self.task_goal,
            subgoal_type=self.args.subgoal_type,
        )
        self.video_buffer.extend(epstate.image_buffer[:-1])
        print(f"[robomme] Gemini agent for {self.args.subgoal_type}, task {self.env_name}, episode {self.episode_id}, setup finished")

    def step(self, epstate: EpisodeState) -> None:
        self.video_buffer.append(epstate.image_buffer[-1])
    
    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        if not self._should_call(count):
            return current_subgoal, False

        text_query = self._get_text_query(count)
        input_data = self.api.prepare_input_data(self.video_buffer, text_query, count)
        response, _ = self.api.call(input_data)
        self.video_buffer.clear()

        if response is None:
            return None, True

        subgoal = response['subgoal']
        if "is complete" in subgoal or "is finished" in subgoal: # avoid using these subgoals as the final subgoal
            subgoal = last_subgoal
        return subgoal, False

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        if not self.api:
            return
        self.api.save_conversation()
        self.api.prepare_input_data(
            epstate.image_buffer,
            self._get_text_query(epstate.count),
            epstate.count,
        )
        self.api.save_final_video(f"{success_flag}_ep{self.episode_id}_{self.task_goal}.mp4")
        self.api.clear_uploaded_files()
        del self.api

    def _get_text_query(self, count: int) -> str:
        if count == 0:
            if self.env_name in TASK_WITH_VIDEO_DEMO:
                template = self.demo_text_query
            else:
                template = self.image_text_query
        else:
            template = self.video_text_query
        return template.format(task_goal=self.task_goal)

    def _should_call(self, count: int) -> bool:
        if count == 0:
            return True
        if self.env_name in LONG_FIRST_ACTION_TASKS and count < 75:
            return False # avoid changing the first action too early
        return count % 48 == 0


class QwenVLSubgoalPredictor(SubgoalPredictorBase):
    
    def setup_api(self) -> None:
        from subgoal_prediction.qwenvl.api import Qwen3VLModel

        self.api = Qwen3VLModel(
            adapter_path=self.args.qwenvl_simpleSG_adapter_path if self.args.subgoal_type == "simple_subgoal" else self.args.qwenvl_groundSG_adapter_path,
            subgoal_type=self.args.subgoal_type,
        )
        print(f"[robomme] QwenVL {self.args.subgoal_type} agent setup finished")
        
    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        self.episode_dir = os.path.join(self.save_dir, self.env_name, f"ep{self.episode_id}")
        self.api.start_new_episode(self.episode_dir, epstate.image_buffer[:-1], self.task_goal)

    def step(self, epstate: EpisodeState) -> None:
        self.video_buffer.append(epstate.image_buffer[-1])

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        # Some tricks. QwenVL sometimes thinks the button has been pressed. hot fix for now.
        # Such special tricks are not encouraged if you consider participate RoboMME challenge @ CVPR 2026
        if self.env_name in ["ButtonUnmask", "PickHighlight"]:
            keep_period = 90
        elif self.env_name == "ButtonUnmaskSwap":
            if last_subgoal and "press the first button" in last_subgoal:
                keep_period = 100
            elif last_subgoal and "press the second button" in last_subgoal:
                keep_period = 250
            else:
                keep_period = 0
        else:
            keep_period = 0

        response = self.api.call(self.video_buffer[-1], count, keep_period)
        self.video_buffer.clear()
        return response, False
    
    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        if self.episode_dir:
            shutil.rmtree(self.episode_dir) # save some space, you can comment this function out to keep all video frames


class PickXtimesProgressQwenVLSubgoalPredictor(QwenVLSubgoalPredictor):
    """QwenVL predictor with a lightweight PickXtimes progress controller.

    The controller uses QwenVL for object/target grounding, but owns the
    pick/place/stop phase transitions. This prevents repeated VLM outputs from
    trapping the policy on a completed subgoal.
    """

    ORDINALS = [
        "first", "second", "third", "fourth", "fifth",
        "sixth", "seventh", "eighth", "ninth", "tenth",
    ]
    WORD_TO_COUNT = {
        "one": 1,
        "once": 1,
        "two": 2,
        "twice": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        self.target_count = self._parse_target_count(self.task_goal)
        self.object_color = self._parse_object_color(self.task_goal)
        self.stage_index = 0
        self.stage_start_count = 0
        self.cached_pick_bbox = None
        self.cached_place_bbox = None
        self.cached_button_bbox = None
        self.last_controller_subgoal = None
        self.last_raw_subgoal = None
        self.last_vlm_call_count = None
        self.latest_image = epstate.image_buffer[-1] if epstate.image_buffer else None
        print(
            "[robomme] PickXtimes progress controller enabled: "
            f"target_count={self.target_count}, color={self.object_color}"
        )

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        if self.env_name == "PickXtimes" and not self._should_call_vlm(count):
            controlled = self.last_controller_subgoal or self._make_stage_subgoal()
            self.last_controller_subgoal = controlled
            return controlled, False

        if self.env_name != "PickXtimes":
            return super().get_subgoal(count, current_subgoal, last_subgoal)

        try:
            candidate = self.api.call(
                self.video_buffer[-1],
                count,
                keep_period=0,
                update_history=False,
            )
        except Exception:
            raise

        self.last_vlm_call_count = count
        self.last_raw_subgoal = candidate
        self._cache_grounding(candidate)

        current_expected = self._make_stage_subgoal(self.stage_index)
        if self.stage_index < self._num_stages() - 1:
            next_expected = self._make_stage_subgoal(self.stage_index + 1)
        else:
            next_expected = current_expected
        progress_state = (
            f"PickXtimes counting plan. target_count={self.target_count}; "
            f"current_stage_index={self.stage_index}; current_stage={self._stage_name(self.stage_index)}; "
            f"current_expected='{current_expected}'; next_if_current_done='{next_expected}'."
        )

        recent_images = self._sample_recent_images()
        controlled = self.api.call_revision(
            recent_images,
            count,
            candidate_subgoal=candidate,
            expected_subgoal=f"current: {current_expected}; next if visually completed: {next_expected}",
            progress_state=progress_state,
        )
        self.video_buffer.clear()
        self._cache_grounding(controlled)
        self._sync_stage_from_corrected_subgoal(controlled, count)

        if controlled != candidate:
            print(
                "[robomme] revision corrected QwenVL subgoal: "
                f"candidate='{candidate}' -> controlled='{controlled}'"
            )
        self.last_controller_subgoal = controlled
        return controlled, False

    def _sample_recent_images(self) -> list:
        if len(self.video_buffer) <= 4:
            return list(self.video_buffer)
        indices = np.linspace(0, len(self.video_buffer) - 1, 4, dtype=int)
        return [self.video_buffer[i] for i in indices]

    def _should_call_vlm(self, count: int) -> bool:
        if self.last_vlm_call_count is None:
            return True
        return count - self.last_vlm_call_count >= self.args.progress_vlm_call_period_steps

    def _sync_stage_from_corrected_subgoal(self, corrected_subgoal: str, count: int) -> None:
        corrected_stage = self._stage_from_subgoal(corrected_subgoal)
        corrected_index = self._stage_index_from_raw_stage(corrected_stage)
        if corrected_index is None:
            return
        if corrected_index < self.stage_index:
            print(
                "[robomme] revision returned stale stage, keeping controller stage: "
                f"corrected_stage={corrected_stage}, controller_stage={self.stage_index}"
            )
            return
        if corrected_index != self.stage_index:
            self._set_stage(corrected_index, count, reason="qwen_revision")

    def _set_stage(self, next_stage: int, count: int, reason: str) -> None:
        next_stage = min(max(next_stage, 0), self._num_stages() - 1)
        if next_stage == self.stage_index:
            return
        print(
            "[robomme] PickXtimes progress stage advance: "
            f"{self.stage_index}->{next_stage} at step {count} ({reason})"
        )
        self.stage_index = next_stage
        self.stage_start_count = count

    def _num_stages(self) -> int:
        return self.target_count * 2 + 1

    def _current_phase(self) -> str:
        if self.stage_index >= self.target_count * 2:
            return "stop"
        return "pick" if self.stage_index % 2 == 0 else "place"

    def _current_repetition(self) -> int:
        return min(self.stage_index // 2 + 1, self.target_count)

    def _stage_index_from_raw_stage(self, raw_stage: Optional[tuple[str, Optional[int]]]) -> Optional[int]:
        if raw_stage is None:
            return None

        phase, repetition = raw_stage
        if phase == "stop":
            if self.stage_index >= self.target_count * 2:
                return self.stage_index
            if self.stage_index == self.target_count * 2 - 1:
                return self.target_count * 2
            return None

        if repetition is None:
            repetition = self._current_repetition()
        repetition = min(max(repetition, 1), self.target_count)

        if phase == "pick":
            return (repetition - 1) * 2
        if phase == "place":
            return (repetition - 1) * 2 + 1
        return None

    def _stage_from_subgoal(self, subgoal: str) -> Optional[tuple[str, Optional[int]]]:
        text = subgoal.lower()
        repetition = self._parse_ordinal_index(text)
        if "press" in text and "button" in text:
            return ("stop", None)
        if "place" in text or "target" in text:
            return ("place", repetition)
        if "pick" in text or "cube" in text:
            return ("pick", repetition)
        return None

    def _make_stage_subgoal(self, stage_index: Optional[int] = None) -> str:
        if stage_index is None:
            stage_index = self.stage_index
        phase = self._phase_for_stage(stage_index)
        color = self.object_color or "cube"
        if phase == "stop":
            return self._with_bbox("press the button to stop", self.cached_button_bbox)

        if phase == "place":
            return self._with_bbox(f"place the {color} cube onto the target", self.cached_place_bbox)

        repetition = self._repetition_for_stage(stage_index)
        ordinal = self.ORDINALS[repetition - 1] if repetition <= len(self.ORDINALS) else f"{repetition}th"
        if self.cached_pick_bbox is None:
            return f"pick up the {color} cube for the {ordinal} time"
        return f"pick up the {color} cube at {self.cached_pick_bbox} for the {ordinal} time"

    def _stage_name(self, stage_index: int) -> str:
        phase = self._phase_for_stage(stage_index)
        if phase == "stop":
            return "stop"
        return f"{phase}{self._repetition_for_stage(stage_index)}"

    def _phase_for_stage(self, stage_index: int) -> str:
        if stage_index >= self.target_count * 2:
            return "stop"
        return "pick" if stage_index % 2 == 0 else "place"

    def _repetition_for_stage(self, stage_index: int) -> int:
        return min(stage_index // 2 + 1, self.target_count)

    def _with_bbox(self, text: str, bbox: Optional[str]) -> str:
        if bbox is None:
            return text
        if "press the button" in text:
            return f"{text} at {bbox}"
        if "place" in text:
            return f"{text} at {bbox}"
        return f"{text} at {bbox}"

    def _cache_grounding(self, subgoal: str) -> None:
        bbox = self._extract_bbox(subgoal)
        if bbox is None:
            return
        phase = self._stage_from_subgoal(subgoal)
        if phase is None:
            return
        if phase[0] == "pick":
            self.cached_pick_bbox = bbox
        elif phase[0] == "place":
            self.cached_place_bbox = bbox
        elif phase[0] == "stop":
            self.cached_button_bbox = bbox

    def _extract_bbox(self, subgoal: str) -> Optional[str]:
        box_match = re.search(r"<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>", subgoal)
        if box_match:
            return f"<|box_start|>({box_match.group(1)},{box_match.group(2)})<|box_end|>"

        point_match = re.search(r"<\s*(\d+)\s*,\s*(\d+)\s*>", subgoal)
        if point_match:
            return f"<{point_match.group(1)}, {point_match.group(2)}>"
        return None

    def _parse_ordinal_index(self, text: str) -> Optional[int]:
        for idx, ordinal in enumerate(self.ORDINALS, start=1):
            if re.search(rf"\bfor\s+the\s+{ordinal}\s+time\b", text):
                return idx
        number_match = re.search(r"\bfor\s+the\s+(\d+)(?:st|nd|rd|th)?\s+time\b", text)
        if number_match:
            return int(number_match.group(1))
        return None

    def _parse_target_count(self, task_goal: str) -> int:
        text = task_goal.lower()
        digit_match = re.search(r"\b(\d+)\s+times?\b", text)
        if digit_match:
            return int(digit_match.group(1))
        for word, value in self.WORD_TO_COUNT.items():
            if re.search(rf"\b{word}\b", text):
                return value
        return 1

    def _parse_object_color(self, task_goal: str) -> str:
        text = task_goal.lower()
        match = re.search(r"\b(red|green|blue|yellow|orange|purple|black|white)\s+cube\b", text)
        if match:
            return match.group(1)
        return ""


class PickXtimesStuckFallbackQwenVLSubgoalPredictor(PickXtimesProgressQwenVLSubgoalPredictor):
    """Single-Qwen-call PickXtimes controller with timeout-based stage fallback.

    This keeps QwenVL as the normal symbolic-memory source. It only overrides
    the returned subgoal when the predicted counting stage is stale for too long.
    """

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        if self.env_name != "PickXtimes":
            return QwenVLSubgoalPredictor.get_subgoal(self, count, current_subgoal, last_subgoal)

        candidate = self.api.call(
            self.video_buffer[-1],
            count,
            keep_period=0,
            update_history=False,
        )
        self.last_vlm_call_count = count
        self.last_raw_subgoal = candidate
        self._cache_grounding(candidate)

        candidate_index = self._stage_index_from_raw_stage(self._stage_from_subgoal(candidate))
        controlled = candidate
        reason = "qwen"
        needs_verifier = False

        if candidate_index is None:
            needs_verifier = True
            reason = "unparsed_candidate"
        elif candidate_index > self.stage_index:
            self._set_stage(candidate_index, count, reason="qwen_advance")
        elif candidate_index < self.stage_index:
            needs_verifier = True
            reason = "qwen_stale"
        else:
            timeout = self._timeout_for_current_stage()
            if count - self.stage_start_count >= timeout and self.stage_index < self._num_stages() - 1:
                needs_verifier = True
                reason = "stuck_timeout"

        if needs_verifier and getattr(self.args, "use_pickxtimes_perceptual_verifier", True):
            controlled, verified_reason = self._verify_current_or_next(candidate, count, reason)
            reason = verified_reason
        elif needs_verifier:
            controlled = self._make_stage_subgoal()

        self.video_buffer.clear()
        self._cache_grounding(controlled)
        self.api.update_history_subgoals(controlled)
        self.last_controller_subgoal = controlled
        if controlled != candidate:
            print(
                "[robomme] PickXtimes stuck fallback controlled QwenVL subgoal: "
                f"reason={reason}, candidate='{candidate}' -> controlled='{controlled}'"
            )
        return controlled, False

    def _timeout_for_current_stage(self) -> int:
        phase = self._current_phase()
        if phase == "pick":
            return self.args.progress_pick_timeout_steps
        if phase == "place":
            return self.args.progress_place_timeout_steps
        return self.args.progress_stop_timeout_steps

    def _verify_current_or_next(
        self,
        candidate: str,
        count: int,
        trigger_reason: str,
    ) -> Tuple[str, str]:
        current_expected = self._make_stage_subgoal(self.stage_index)
        next_stage = min(self.stage_index + 1, self._num_stages() - 1)
        next_expected = self._make_stage_subgoal(next_stage)
        progress_state = (
            f"PickXtimes counting plan. target_count={self.target_count}; "
            f"current_stage_index={self.stage_index}; current_stage={self._stage_name(self.stage_index)}; "
            f"trigger={trigger_reason}."
        )
        allowed = f"current: {current_expected}; next if visually completed: {next_expected}"
        try:
            verified = self.api.call_revision(
                self._sample_recent_images(),
                count,
                candidate_subgoal=candidate,
                expected_subgoal=allowed,
                progress_state=progress_state,
            )
        except Exception as exc:
            print(f"[robomme] PickXtimes verifier failed ({exc}); keeping current stage")
            self.stage_start_count = count
            return current_expected, f"{trigger_reason}_verifier_error"

        verified_index = self._stage_index_from_raw_stage(self._stage_from_subgoal(verified))
        if verified_index is None:
            self.stage_start_count = count
            return current_expected, f"{trigger_reason}_verifier_unparsed"
        if verified_index > self.stage_index:
            self._set_stage(verified_index, count, reason=f"{trigger_reason}_verifier_advance")
            return self._make_stage_subgoal(), f"{trigger_reason}_verifier_advance"
        if verified_index < self.stage_index:
            self.stage_start_count = count
            return current_expected, f"{trigger_reason}_verifier_stale"
        self.stage_start_count = count
        return self._make_stage_subgoal(), f"{trigger_reason}_verifier_hold"


class MemERSubgoalPredictor(SubgoalPredictorBase):
    def setup_api(self) -> None:
        from subgoal_prediction.qwenvl.api_memer import Qwen3VLModelMemER

        self.api = Qwen3VLModelMemER(adapter_path=self.args.memer_adapter_path)
        print("[robomme] MemER agent setup finished")
    
    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        self.episode_dir = os.path.join(self.save_dir, self.env_name, f"ep{self.episode_id}")
        self.api.start_new_episode(self.episode_dir, epstate.image_buffer[:-1], self.task_goal)

    def step(self, epstate: EpisodeState) -> None:
        self.api.add_execution_frame(epstate.image_buffer[-1])

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        response = self.api.call()
        return response, False

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        if self.episode_dir:
            shutil.rmtree(self.episode_dir) # save some space, you can comment this function out to keep all video frames


class OracleSubgoalPredictor(SubgoalPredictorBase):
    
    def setup_api(self) -> None:
        print("[robomme] Oracle agent setup finished")
    
    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        if self.args.subgoal_type == "simple_subgoal":
            return self.env_runner.simple_subgoal_oracle, False
        else:
            return self.env_runner.grounded_subgoal_oracle, False


def build_subgoal_predictor(
    args,
    save_dir: Path,
) -> SubgoalPredictorBase:
    if args.use_gemini:
        return GeminiSubgoalPredictor(args, save_dir)
    if args.use_qwenvl:
        if args.use_pickxtimes_progress:
            return PickXtimesProgressQwenVLSubgoalPredictor(args, save_dir)
        if args.use_pickxtimes_stuck_fallback:
            return PickXtimesStuckFallbackQwenVLSubgoalPredictor(args, save_dir)
        return QwenVLSubgoalPredictor(args, save_dir)
    if args.use_memer:
        return MemERSubgoalPredictor(args, save_dir)
    if args.use_oracle:
        return OracleSubgoalPredictor(args, save_dir)
    
    return NullSubgoalPredictor(args, save_dir)
