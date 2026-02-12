"""Scratchpads addon."""

import asyncio
from concurrent.futures.thread import BrokenThreadPool
import contextlib
import json
from functools import partial
from pathlib import Path
from typing import Awaitable, cast

from pyprland.ipc_paths import IPC_FOLDER

from ...adapters.units import convert_coords
from ...aioops import TaskManager
from ...common import MINIMUM_FULL_ADDR_LEN, is_rotated
from ...models import ClientInfo, Environment, ReloadReason, VersionInfo
from ..interface import Plugin
from .common import FocusTracker, HideFlavors
from .events import EventsMixin
from .helpers import (
    compute_offset,
    get_active_space_identifier,
    get_all_space_identifiers,
    get_match_fn,
    mk_scratch_name,
)
from .lifecycle import LifecycleMixin
from .lookup import ScratchDB
from .objects import Scratch
from .schema import validate_scratchpad_config
from .transitions import TransitionsMixin
from .windowruleset import WindowRuleSet

DYNAMIC_ASSIGNMENT_FILE = Path(IPC_FOLDER) / ".pyprland_scratch_assignment_json"

class Extension(LifecycleMixin, EventsMixin, TransitionsMixin, Plugin, environments=[Environment.HYPRLAND]):
    """Makes your applications into dropdowns & togglable popups."""

    procs: dict[str, asyncio.subprocess.Process]
    scratches: ScratchDB

    workspace = ""  # Currently active workspace
    monitor = ""  # Currently active monitor

    _tasks: TaskManager  # Task manager for hysteresis and other background tasks
    focused_window_tracking: dict[str, FocusTracker]
    previously_focused_window: str = ""
    last_focused: Scratch | None = None
    _dynamic_recovery_done: bool

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.procs = {}
        self.scratches = ScratchDB()
        self.focused_window_tracking = {}
        self._tasks = TaskManager()
        self._tasks.start()
        self._dynamic_recovery_done = False

    async def _restore_dynamic_windows(self) -> None:
        """Move dynamic scratchpad windows back to a real workspace.

        Called during exit to prevent windows from being stranded in special workspaces.
        """
        target_workspace = self.state.active_workspace or "1"
        for scratch in self.scratches.values():
            if not scratch.is_dynamic or not scratch.dynamic_window_addr:
                continue

            try:
                await self.backend.move_window_to_workspace(scratch.dynamic_window_addr, target_workspace)
                if not scratch.was_floating_before_send:
                    await self.backend.execute(f"settiled address:{scratch.dynamic_window_addr}")
                scratch.clear_dynamic_window()
            except (OSError, ConnectionError):
                self.log.warning("Failed to restore dynamic window for '%s'", scratch.uid, exc_info=True)

    async def exit(self) -> None:
        """Exit hook."""
        # Stop all managed tasks (hysteresis, etc.)
        await self._tasks.stop()

        # Restore dynamic windows to a real workspace so they're not stranded
        await self._restore_dynamic_windows()

        # Save assignments (cleared for successfully restored windows, preserved for failed ones)
        self._save_dynamic_assignments()
        
        async def die_in_piece(scratch: Scratch) -> None:
            if scratch.uid in self.procs:
                proc = self.procs[scratch.uid]

                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                else:
                    for _ in range(10):
                        if not await scratch.is_alive():
                            break
                        await asyncio.sleep(0.1)
                    if await scratch.is_alive():
                        with contextlib.suppress(ProcessLookupError):
                            proc.kill()
                await proc.wait()

        await asyncio.gather(*(die_in_piece(scratch) for scratch in self.scratches.values()))

    def validate_config(self) -> list[str]:
        """Validate scratchpads configuration."""
        errors: list[str] = []

        for name, scratch_config in self.config.iter_subsections():
            # Skip per-monitor subsections (e.g. "term.monitor.DP-1") - validated within schema
            if "." in name:
                continue

            errors.extend(validate_scratchpad_config(name, scratch_config))

        return errors

    @classmethod
    def validate_config_static(cls, _plugin_name: str, config: dict) -> list[str]:
        """Validate scratchpads configuration without instantiation."""
        errors: list[str] = []
        for name, scratch_config in config.items():
            if not isinstance(scratch_config, dict) or "." in name:
                continue
            errors.extend(validate_scratchpad_config(name, scratch_config))
        return errors

    async def on_reload(self, reason: ReloadReason = ReloadReason.RELOAD) -> None:
        """Config loader."""
        _ = reason  # unused
        # Sanity checks
        _scratch_classes: dict[str, str] = {}
        for uid, scratch in self.config.items():
            _klass = scratch.get("class")
            if _klass:
                if _klass in _scratch_classes:
                    text = "Scratch class %s is duplicated (in %s and %s)"
                    args = (
                        scratch["class"],
                        uid,
                        _scratch_classes[_klass],
                    )
                    self.log.error(text, *args)
                    await self.backend.notify_error(text % args)
                _scratch_classes[_klass] = uid

        # Create new scratches with fresh config items
        scratches = {name: Scratch(name, self.config, self) for name, options in self.config.iter_subsections()}

        scratches_to_spawn = set()
        for name, new_scratch in scratches.items():
            scratch = self.scratches.get(name)
            if scratch:  # if existing scratch exists, overrides the conf object
                scratch.set_config(self.config)
            else:
                # else register it
                self.scratches.register(new_scratch, name)
                is_lazy = new_scratch.conf.get_bool("lazy")
                if not is_lazy:
                    scratches_to_spawn.add(name)

        for name in scratches_to_spawn:
            if await self.ensure_alive(name):
                scratch = self.scratches.get(name)
                assert scratch
                scratch.meta.should_hide = True
            else:
                self.log.error("Failure starting %s", name)

        for scratch in list(self.scratches.get_by_state("configured")):
            assert scratch
            self.scratches.clear_state(scratch, "configured")

        # Recovery is startup-only: don't force-release dynamic windows on normal config reloads.
        if not self._dynamic_recovery_done:
            await self._load_dynamic_assignments()
            self._dynamic_recovery_done = True



    async def _unset_windowrules(self, scratch: Scratch) -> None:
        """Unset the windowrules.

        Args:
            scratch: The scratchpad object
        """
        defined_class = scratch.conf.get("class")
        if defined_class:
            if self.state.hyprland_version < VersionInfo(0, 53, 0):
                if self.state.hyprland_version > VersionInfo(0, 47, 2):
                    await self.backend.set_keyword(f"windowrule unset, class: {defined_class}")
                else:
                    await self.backend.set_keyword(f"windowrule unset, ^({defined_class})$")
            else:
                await self.backend.set_keyword(f"windowrule[{scratch.uid}]:enable false")

    async def _configure_windowrules(self, scratch: Scratch) -> None:
        """Set initial client window state (sets windowrules).

        Args:
            scratch: The scratchpad object
        """
        self.scratches.set_state(scratch, "configured")
        animation_type: str = scratch.conf.get_str("animation").lower()
        defined_class: str = scratch.conf.get_str("class")
        skipped_windowrules: list[str] = cast("list", scratch.conf.get("skip_windowrules"))
        wr = WindowRuleSet(self.state)
        wr.set_class(defined_class)
        wr.set_name(scratch.uid)
        if defined_class:
            forced_monitor = scratch.conf.get("force_monitor")
            if forced_monitor and forced_monitor not in self.state.active_monitors:
                self.log.error("forced monitor %s doesn't exist", forced_monitor)
                await self.backend.notify_error(f"Monitor '{forced_monitor}' doesn't exist, check {scratch.uid}'s scratch configuration")
                forced_monitor = None
            monitor = await self.backend.get_monitor_props(name=cast("str | None", forced_monitor))
            width, height = convert_coords(scratch.conf.get_str("size"), monitor)

            if "float" not in skipped_windowrules:
                if self.state.hyprland_version < VersionInfo(0, 53, 0):
                    wr.set("float", "")
                else:
                    wr.set("float", "on")
            if "workspace" not in skipped_windowrules:
                wr.set("workspace", f"{mk_scratch_name(scratch.uid)} silent")
            set_aspect = "aspect" not in skipped_windowrules

            if animation_type:
                margin_x = (monitor["width"] - width) // 2
                margin_y = (monitor["height"] - height) // 2

                if is_rotated(monitor):
                    margin_x, margin_y = margin_y, margin_x

                t_pos = {
                    "fromtop": f"{margin_x} -200%",
                    "frombottom": f"{margin_x} 200%",
                    "fromright": f"200% {margin_y}",
                    "fromleft": f"-200% {margin_y}",
                }[animation_type]
                if set_aspect:
                    wr.set("move", t_pos)

            if set_aspect:
                wr.set("size", f"{width} {height}")

            await self.backend.execute(wr.get_content(), base_command="keyword")

    def cancel_task(self, uid: str) -> bool:
        """Cancel a hysteresis task.

        Args:
            uid: The scratchpad name

        Returns:
            True if task was cancelled, False if no task existed
        """
        cancelled = self._tasks.cancel_keyed(uid)
        if cancelled:
            self.log.debug("Canceled previous task for %s", uid)
        return cancelled

    async def _detach_window(self, scratch: Scratch, address: str) -> None:
        """Detach a window from a scratchpad.

        Args:
            scratch: The scratchpad to detach from
            address: The window address to detach
        """
        scratch.extra_addr.remove(address)
        if scratch.conf.get("pinned"):
            await self.backend.pin_window(address)  # toggles pin off

    async def _attach_window(self, scratch: Scratch, address: str) -> None:
        """Attach a window to a scratchpad.

        Args:
            scratch: The scratchpad to attach to
            address: The window address to attach
        """
        # Remove from any other scratchpad first
        for s in self.scratches.values():
            if address in s.extra_addr:
                s.extra_addr.remove(address)
        scratch.extra_addr.add(address)
        if scratch.conf.get("pinned"):
            await self.backend.pin_window(address)
        # If scratchpad is visible, move the attached window to same workspace
        if scratch.visible and scratch.client_info is not None:
            workspace = scratch.client_info.get("workspace", {}).get("name", "")
            if workspace:
                await self.backend.move_window_to_workspace(address, workspace)

    async def run_attach(self) -> None:
        """Attach the focused window to the last focused scratchpad."""
        if not self.last_focused:
            await self.backend.notify_error("No scratchpad was focused")
            return
        focused = self.state.active_window
        if not focused or len(focused) < MINIMUM_FULL_ADDR_LEN:
            await self.backend.notify_error("No valid window focused")
            return
        scratch = self.last_focused
        if focused == scratch.full_address:
            await self.backend.notify_info(f"Scratch {scratch.uid} can't attach or detach to itself")
            return
        if not scratch.visible:
            await self.run_show(scratch.uid)

        if focused in scratch.extra_addr:
            await self._detach_window(scratch, focused)
        else:
            await self._attach_window(scratch, focused)

    def _detach_from_other_scratches(self, focused: str, uid: str) -> None:
        """Detach a window from any other dynamic scratchpad it may belong to.

        Args:
            focused: The focused window address
            uid: The target scratchpad uid (to exclude)
        """
        for s in self.scratches.values():
            if s.is_dynamic and s.dynamic_window_addr == focused and s.uid != uid:
                old_addr = s.address
                s.clear_dynamic_window()
                if old_addr:
                    self.scratches.clear(addr=old_addr)
                break

    async def _swap_dynamic_window(self, scratch: Scratch) -> None:
        """Release the current dynamic window to make room for a new one.

        Args:
            scratch: The scratchpad whose current window should be released
        """
        await self._release_dynamic_window(scratch)

    async def run_send(self, uid: str) -> None:
        """<name> sends the focused window to the dynamic scratchpad "name".

        Args:
            uid: The scratchpad name

        Example:
            pypr send float1
        """
        scratch = self.scratches.get(uid)
        if not scratch or not scratch.is_dynamic:
            msg = f"Scratchpad '{uid}' not found" if not scratch else f"Scratchpad '{uid}' is command-based, use empty scratchpads for send"
            await self.backend.notify_error(msg)
            return

        focused = self.state.active_window
        if not focused or len(focused) < MINIMUM_FULL_ADDR_LEN:
            await self.backend.notify_error("No valid window focused")
            return

        # If the focused window IS this scratchpad's window, release it
        if scratch.dynamic_window_addr == focused:
            await self._release_dynamic_window(scratch)
            return

        # Check if this window belongs to a command-based scratchpad
        for s in self.scratches.values():
            if s.have_command and s.full_address == focused:
                await self.backend.notify_error(f"Window belongs to command-based scratchpad '{s.uid}'")
                return

        self._detach_from_other_scratches(focused, uid)

        # If this scratch already has a different window, swap it out
        is_swap = bool(scratch.dynamic_window_addr and scratch.dynamic_window_addr != focused)
        if is_swap:
            await self._swap_dynamic_window(scratch)

        await self._assign_focused_window(scratch, uid, focused, is_swap)

    async def _assign_focused_window(self, scratch: Scratch, uid: str, focused: str, auto_show: bool) -> None:
        """Assign the focused window to a dynamic scratchpad.

        Args:
            scratch: The target scratchpad
            uid: The scratchpad name
            focused: The focused window address
            auto_show: Whether to auto-show the scratchpad after assignment
        """
        client_info = await self.backend.get_client_props(addr=focused)
        if not client_info:
            await self.backend.notify_error("Could not get window properties")
            return

        scratch.assign_window(client_info)
        self.scratches.register(scratch, addr=scratch.address)

        if not client_info.get("floating"):
            await self.backend.execute(f"setfloating address:{focused}")

        await self.backend.execute(f"movetoworkspacesilent {mk_scratch_name(uid)},address:{focused}")
        scratch.visible = False

        self._save_dynamic_assignments()

        # When swapping, auto-show the new window so the user doesn't need to toggle
        if auto_show:
            await self.run_show(uid)
    
            
    async def run_toggle(self, uid_or_uids: str) -> None:
        """<name> toggles visibility of scratchpad "name" (supports multiple names).

        Args:
            uid_or_uids: Space-separated scratchpad name(s)

        Example:
            pypr toggle term
            pypr toggle term music
        """
        uids = list(filter(bool, map(str.strip, uid_or_uids.split()))) if " " in uid_or_uids else [uid_or_uids.strip()]

        for uid in uids:
            self.cancel_task(uid)

        assert uids
        first_scratch = self.scratches.get(uids[0])
        if not first_scratch:
            self.log.warning("%s doesn't exist, can't toggle.", uids[0])
            await self.backend.notify_error(f"Scratchpad '{uids[0]}' not found, check your configuration & the toggle parameter")
            return

        self.log.debug(
            "visibility_check: %s == %s",
            first_scratch.meta.space_identifier,
            get_active_space_identifier(self.state),
        )
        if first_scratch.conf.get_bool("alt_toggle"):
            # Needs to be on any monitor (if workspace matches)
            extra_visibility_check = first_scratch.meta.space_identifier in await get_all_space_identifiers(
                await self.backend.get_monitors()
            )
        else:
            # Needs to be on the active monitor+workspace
            extra_visibility_check = first_scratch.meta.space_identifier == get_active_space_identifier(
                self.state
            )  # Visible on the currently focused monitor

        is_visible = first_scratch.visible and (
            first_scratch.forced_monitor or extra_visibility_check
        )  # Always showing on the same monitor
        tasks = []

        for uid in uids:
            item = self.scratches.get(uid)
            if not item:
                self.log.warning("%s is not configured", uid)
            else:
                self.log.debug("%s visibility: %s and %s", uid, is_visible, item.visible)
                if is_visible and await item.is_alive():
                    tasks.append(partial(self.run_hide, uid))
                else:
                    tasks.append(partial(self.run_show, uid))
        await asyncio.gather(*(asyncio.create_task(t()) for t in tasks))

    async def run_show(self, uid: str) -> None:
        """<name> shows scratchpad "name" (accepts "*").

        Args:
            uid: The scratchpad name, or "*" to show all hidden scratchpads
        """
        if uid == "*":
            await asyncio.gather(*(self.run_show(s.uid) for s in self.scratches.values() if not s.visible))
            return
        scratch = self.scratches.get(uid)

        if not scratch:
            self.log.warning("%s doesn't exist, can't hide.", uid)
            await self.backend.notify_error(f"Scratchpad '{uid}' not found, check your configuration or the show parameter")
            return

        self.cancel_task(uid)

        self.log.info("Showing %s", uid)
        was_alive = await scratch.is_alive()
        if not await self.ensure_alive(uid):
            self.log.error("Failed to show %s, aborting.", uid)
            return

        if not was_alive:
            was_alive = await scratch.is_alive()

        excluded_ids = scratch.conf.get("excludes")
        restore_excluded = scratch.conf.get_bool("restore_excluded")
        if excluded_ids == "*":
            excluded_ids = [excluded.uid for excluded in self.scratches.values() if excluded.uid != scratch.uid]
        elif excluded_ids is None:
            excluded_ids = []

        for e_uid in cast("list[str]", excluded_ids):
            excluded = self.scratches.get(e_uid)
            assert excluded
            if excluded.visible:
                await self.run_hide(e_uid, flavor=HideFlavors.TRIGGERED_BY_AUTOHIDE | HideFlavors.IGNORE_TILED)
                if restore_excluded:
                    scratch.excluded_scratches.append(e_uid)

        await scratch.initialize(self)

        scratch.visible = True
        scratch.meta.space_identifier = get_active_space_identifier(self.state)
        monitor = await self.backend.get_monitor_props(name=scratch.forced_monitor)

        assert monitor
        assert scratch.full_address, "No address !"

        await self._show_transition(scratch, monitor, was_alive)
        scratch.monitor = monitor["name"]

    async def _handle_multiwindow(self, scratch: Scratch, clients: list[ClientInfo]) -> bool:
        """Collect every matching client for the scratchpad and add them to extra_addr if needed.

        Args:
            scratch: The scratchpad object
            clients: The list of clients
        """
        if not scratch.conf.get_bool("multi"):
            return False
        try:
            match_by, match_value = scratch.get_match_props()
        except KeyError:
            return False
        match_fn = get_match_fn(match_by, match_value)
        hit = False
        for client in clients:
            if client["address"] == scratch.full_address:
                continue
            if match_fn(client[match_by], match_value):  # type: ignore[literal-required]
                address = client["address"]
                if address not in scratch.extra_addr:
                    scratch.extra_addr.add(address)
                    if scratch.conf.get("pinned"):
                        await self.backend.pin_window(address)
                    hit = True
        return hit

    async def run_hide(self, uid: str, flavor: HideFlavors = HideFlavors.NONE) -> None:
        """<name> hides scratchpad "name" (accepts "*").

        Args:
            uid: The scratchpad name, or "*" to hide all visible scratchpads
            flavor: Internal hide behavior flags (default: NONE)
        """
        if uid == "*":
            await asyncio.gather(*(self.run_hide(s.uid) for s in self.scratches.values() if s.visible))
            return

        scratch = self.scratches.get(uid)

        if not scratch:
            await self.backend.notify_error(f"Scratchpad '{uid}' not found, check your configuration or the hide parameter")
            self.log.warning("%s is not configured", uid)
            return

        if not await scratch.is_alive():
            return

        if scratch.client_info is not None and flavor & HideFlavors.IGNORE_TILED and not scratch.client_info["floating"]:
            return

        active_window = self.state.active_window
        active_workspace = self.state.active_workspace

        if not scratch.visible and not flavor & HideFlavors.FORCED and not flavor & HideFlavors.TRIGGERED_BY_AUTOHIDE:
            await self.backend.notify_error(f"Scratchpad '{uid}' is not visible, will not hide.")
            self.log.warning("%s is already hidden", uid)
            return

        await self._hide_scratch(scratch, active_window, active_workspace)

    async def _hide_scratch(self, scratch: Scratch, active_window: str, active_workspace: str) -> None:
        """Perform the actual hide operation.

        Args:
            scratch: The scratchpad object
            active_window: The active window address
            active_workspace: The active workspace name
        """
        clients = await self.backend.execute_json("clients")
        await scratch.update_client_info(clients=clients)
        # After update_client_info, client_info is guaranteed to be set (or KeyError raised)
        assert scratch.client_info is not None
        ref_position = scratch.client_info["at"]
        monitor_info = scratch.meta.monitor_info
        if monitor_info is None:
            self.log.error("Cannot hide %s: no monitor_info available", scratch.uid)
            return
        scratch.meta.extra_positions[scratch.address] = compute_offset(ref_position, (monitor_info["x"], monitor_info["y"]))
        # collects window which have been created by the app
        if scratch.conf.get_bool("multi"):
            await self._handle_multiwindow(scratch, clients)
            positions = {}
            for sub_client in clients:
                if sub_client["address"] in scratch.extra_addr:
                    positions[sub_client["address"]] = compute_offset(sub_client["at"], ref_position)
            scratch.meta.extra_positions.update(positions)
        scratch.visible = False
        scratch.meta.should_hide = False
        self.log.info("Hiding %s", scratch.uid)
        await self._pin_scratch(scratch)
        await self._hide_transition(scratch, monitor_info)

        if not scratch.conf.get_bool("close_on_hide"):
            await self.backend.move_window_to_workspace(scratch.full_address, mk_scratch_name(scratch.uid))

            for addr in scratch.extra_addr:
                await self.backend.move_window_to_workspace(addr, mk_scratch_name(scratch.uid))
                await asyncio.sleep(0.01)
        else:
            await self.backend.close_window(scratch.full_address)

            for addr in scratch.extra_addr:
                await self.backend.close_window(addr)
                await asyncio.sleep(0.01)

        for e_uid in scratch.excluded_scratches:
            await self.run_show(e_uid)
        scratch.excluded_scratches.clear()
        await self._handle_focus_tracking(scratch, active_window, active_workspace, clients)

    async def _handle_focus_tracking(self, scratch: Scratch, active_window: str, active_workspace: str, clients: ClientInfo | dict) -> None:
        """Handle focus tracking.

        Args:
            scratch: The scratchpad object
            active_window: The active window address
            active_workspace: The active workspace name
            clients: The list of clients
        """
        if not scratch.conf.get_bool("smart_focus"):
            return
        for track in self.focused_window_tracking.values():
            if scratch.have_address(track.prev_focused_window):
                track.clear()
        tracker = self.focused_window_tracking.get(scratch.uid)
        if tracker and not tracker.prev_focused_window_wrkspc.startswith("special:"):
            same_workspace = tracker.prev_focused_window_wrkspc == active_workspace
            t_pfw = tracker.prev_focused_window
            client = next(filter(lambda d: d.get("address") == t_pfw, cast("list[dict]", clients)), None)
            if (
                client
                and scratch.have_address(active_window)
                and same_workspace
                and not scratch.have_address(tracker.prev_focused_window)
                and not client["workspace"]["name"].startswith("special")
            ):
                self.log.debug("Previous scratch: %s", self.scratches.get(addr=tracker.prev_focused_window))
                await self.backend.focus_window(tracker.prev_focused_window)
