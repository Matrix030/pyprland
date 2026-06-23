---
---

# stash

Store single-window overlays in named stashes.

Each stash name is a single slot. A stashed window can be shown as a pinned overlay on top of your current workspace and stays visible while you switch workspaces, including special workspaces.

## Usage

```bash
bind = $mainMod,       S, exec, pypr stash_toggle S
bind = $mainMod SHIFT, S, exec, pypr stash_send   S
bind = $mainMod,       C, exec, pypr stash_toggle C
bind = $mainMod SHIFT, C, exec, pypr stash_send   C
```

`stash_send <name>`:

- sends the focused window into stash `<name>`
- if `<name>` is already occupied, releases the old window to the current workspace and replaces it
- if the focused window is already the shown stash window, releases it back to the current workspace

`stash_toggle <name>`:

- shows the named stash as a pinned floating overlay
- hides it back into its hidden special workspace

The first show uses the configured `size` and `position`. If `preserve_aspect = true`, later hide/show cycles keep the live size and position you last left the stash at.

### Animation

Set `animation` to make the overlay slide in on show and slide out on hide. Valid directions are `fromTop`, `fromBottom`, `fromLeft`, `fromRight`, `fromTopLeft`, `fromTopRight`, `fromBottomLeft` and `fromBottomRight` (case-insensitive; `-`, `_` and spaces are ignored). Leave it empty (the default) for an instant, non-animated show/hide.

- `offset` controls how far past the screen edge the window starts/ends the slide (default `100%`, relative to the window size; also accepts pixel values like `400px`).
- `hide_delay` is how long (in seconds, default `0.2`) the daemon waits for the slide-out to finish before tucking the window back into its hidden workspace.

Animations require Hyprland (the plugin registers a `no_anim` window rule for an internal tag the first time an animation runs).

## Commands

<PluginCommands plugin="stash" />

## Configuration

<PluginConfig plugin="stash" />

### Example

```toml
[pyprland]
plugins = ["stash"]

[stash.S]
animation = "fromBottom"
size = "24% 54%"
position = "76% 22%"
preserve_aspect = true

[stash.C]
animation = ""
size = "24% 54%"
position = "76% 22%"
preserve_aspect = true
```

## Notes

- `animation` slides the overlay in/out from the chosen edge; leave it empty for an instant show/hide. See [Animation](#animation) for `offset` and `hide_delay`.
- Stash windows are backed by hidden `special:st-<name>` workspaces when not shown.
- During a clean `pypr` shutdown, stash windows are released back to the active workspace as a best effort cleanup.
