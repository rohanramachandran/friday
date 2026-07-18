# Menu bar app setup

The frontend is a single Swift file plus an Info.plist, dropped into a
standard Xcode project.

1. Xcode, File, New, Project, macOS, App
2. Name it `FRIDAY`, Interface SwiftUI, Language Swift
3. Delete the generated `ContentView.swift` and `FRIDAYApp.swift`
4. Drag `app/FRIDAY/FRIDAY.swift` in (check "Copy items if needed")
5. Replace the project's `Info.plist` contents with `app/FRIDAY/Info.plist`
6. In Signing and Capabilities either turn App Sandbox off, or keep it on
   and add Microphone, Audio Input, Apple Events, and Outgoing Network
7. Run. The menu bar icon appears; there is no dock icon.

## Permissions

macOS prompts on first use:

- Accessibility (global hotkeys): System Settings, Privacy, Accessibility
- Microphone: prompted automatically
- Screen Recording: System Settings, Privacy, Screen Recording
- Automation (AppleScript): prompted per app

## Start at login

```bash
sed "s|__FRIDAY_PATH__|$PWD|g" scripts/com.rohan.friday.plist > ~/Library/LaunchAgents/com.rohan.friday.plist
launchctl load ~/Library/LaunchAgents/com.rohan.friday.plist
```

## Troubleshooting

- Daemon out of memory at startup: close Chrome, or point `MODEL_ID` in
  `daemon/orchestrator/brain.py` at a smaller model
- Slow first response: normal; kernel compilation plus first prefill
- No audio out: check microphone permission and the Kokoro voice download
- Hotkey dead: Accessibility permission not granted
- Vision tool fails: `pip install mlx-vlm --upgrade` inside the venv
