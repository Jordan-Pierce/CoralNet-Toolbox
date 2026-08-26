#!/usr/bin/env bash
#
# Maps LOCKOUT_LEVEL onto the desktop-environment variables that Kasm's
# vnc_startup.sh reads. This has to happen in an entrypoint rather than in
# custom_startup.sh, because vnc_startup.sh chooses and launches the desktop
# BEFORE it ever calls the custom startup script.
#
set -eu

LOCKOUT_LEVEL="${LOCKOUT_LEVEL:-2}"

case "$LOCKOUT_LEVEL" in
    1)
        # Full XFCE desktop: panel, file manager, right-click menu, window
        # decorations. The toolbox is just one app among others. Use this for
        # development and debugging, not for handing to an untrusted user.
        export START_DE=xfce4-session
        export START_XFCE4=1
        echo "[entrypoint] LOCKOUT_LEVEL=1 - full XFCE desktop"
        ;;
    2)
        # Kiosk: openbox with no panel, no desktop, no file manager, no menus,
        # no window decorations, and no window-management keybindings. The
        # toolbox is the only thing on screen and cannot be closed or minimised.
        export START_DE=openbox
        export START_XFCE4=0
        echo "[entrypoint] LOCKOUT_LEVEL=2 - kiosk (openbox, no desktop)"
        ;;
    3)
        # Deliberately refuses rather than silently running as level 2. Level 3
        # is an OS-access boundary, and quietly delivering less isolation than
        # the operator asked for is worse than not starting.
        cat >&2 <<'MSG'
[entrypoint] FATAL: LOCKOUT_LEVEL=3 is not implemented.

Level 3 is reserved for OS-level isolation, which is not just a UI change:
  - read-only root filesystem, --cap-drop ALL, no-new-privileges
  - removal of Chrome / shells / file managers from the image
  - disabling the KasmVNC clipboard, upload and printer sidecars
  - constraining the toolbox's own file dialogs, which can browse /

Levels 1 and 2 lock the *interface*; they do not stop a determined user from
reaching the filesystem through the application. Refusing to start so that
nobody mistakes level 2 for level 3.

Use LOCKOUT_LEVEL=2 for kiosk mode.
MSG
        exit 1
        ;;
    *)
        echo "[entrypoint] FATAL: LOCKOUT_LEVEL must be 1, 2 or 3 (got '$LOCKOUT_LEVEL')" >&2
        exit 1
        ;;
esac

# Hand off to Kasm's original entrypoint chain, unchanged.
exec /dockerstartup/kasm_default_profile.sh \
     /dockerstartup/vnc_startup.sh \
     /dockerstartup/kasm_startup.sh "$@"
