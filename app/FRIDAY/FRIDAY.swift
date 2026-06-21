// FRIDAY.swift: menu bar app, single file.
// Build in Xcode: macOS app, SwiftUI lifecycle, drop this in.

import SwiftUI
import AVFoundation
import ScreenCaptureKit
import Carbon.HIToolbox

@main
struct FRIDAYApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    var body: some Scene {
        Settings { EmptyView() }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem!
    var overlay: NSPanel?
    let engine = FridayEngine()
    var hotkeyRefs: [EventHotKeyRef?] = [nil, nil]

    func applicationDidFinishLaunching(_ n: Notification) {
        NSApp.setActivationPolicy(.accessory)
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.title = "◉"
        statusItem.button?.action = #selector(toggleOverlay)
        statusItem.button?.target = self
        engine.connect()
        registerHotkeys()
    }

    @objc func toggleOverlay() {
        if let o = overlay, o.isVisible { o.orderOut(nil) } else { showOverlay() }
    }

    func showOverlay() {
        if overlay == nil {
            let p = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 520, height: 200),
                            styleMask: [.borderless, .nonactivatingPanel],
                            backing: .buffered, defer: false)
            p.level = .floating
            p.isFloatingPanel = true
            p.isOpaque = false
            p.backgroundColor = .clear
            p.hasShadow = true
            p.contentView = NSHostingView(rootView: OverlayView(engine: engine))
            overlay = p
        }
        if let screen = NSScreen.main {
            let f = screen.frame
            overlay?.setFrameOrigin(NSPoint(x: f.midX - 260, y: f.midY - 100))
        }
        overlay?.makeKeyAndOrderFront(nil)
    }

    func registerHotkeys() {
        // ⌥Space = push-to-talk, ⌥⇧Space = screenshot+talk
        registerHotkey(id: 1, keyCode: UInt32(kVK_Space), mods: UInt32(optionKey)) { [weak self] press in
            if press { self?.engine.startRecording(withScreenshot: false); self?.showOverlay() }
            else { self?.engine.stopRecording() }
        }
        registerHotkey(id: 2, keyCode: UInt32(kVK_Space), mods: UInt32(optionKey | shiftKey)) { [weak self] press in
            if press { self?.engine.startRecording(withScreenshot: true); self?.showOverlay() }
            else { self?.engine.stopRecording() }
        }
    }

    func registerHotkey(id: UInt32, keyCode: UInt32, mods: UInt32, handler: @escaping (Bool) -> Void) {
        var ref: EventHotKeyRef?
        let hkID = EventHotKeyID(signature: OSType(0x46524459), id: id)  // "FRDY"
        RegisterEventHotKey(keyCode, mods, hkID, GetApplicationEventTarget(), 0, &ref)
        hotkeyRefs.append(ref)
        HotkeyManager.shared.handlers[id] = handler
        HotkeyManager.shared.install()
    }
}

final class HotkeyManager {
    static let shared = HotkeyManager()
    var handlers: [UInt32: (Bool) -> Void] = [:]
    private var installed = false

    func install() {
        guard !installed else { return }; installed = true
        var spec = [
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed)),
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased)),
        ]
        InstallEventHandler(GetApplicationEventTarget(), { _, event, _ in
            var hk = EventHotKeyID()
            GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                              nil, MemoryLayout<EventHotKeyID>.size, nil, &hk)
            let kind = GetEventKind(event)
            let pressed = kind == UInt32(kEventHotKeyPressed)
            DispatchQueue.main.async { HotkeyManager.shared.handlers[hk.id]?(pressed) }
            return noErr
        }, 2, &spec, nil, nil)
    }
}

// MARK: - Engine

final class FridayEngine: ObservableObject {
    @Published var state: String = "Idle"
    @Published var transcript: String = ""
    @Published var response: String = ""
    @Published var toolStatus: String = ""

    private var ws: URLSessionWebSocketTask?
    private let session = URLSession(configuration: .default)
    private var audioEngine: AVAudioEngine?
    private var audioBuffer = Data()
    private var pendingScreenshot: String?
    private let player = AudioStreamPlayer()

    func connect() {
        let url = URL(string: "ws://127.0.0.1:8765")!
        ws = session.webSocketTask(with: url)
        ws?.resume()
        receive()
    }

    private func receive() {
        ws?.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let msg):
                if case .string(let text) = msg {
                    self.handle(text)
                }
                self.receive()
            case .failure:
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) { self.connect() }
            }
        }
    }

    private func handle(_ text: String) {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else { return }
        DispatchQueue.main.async {
            switch type {
            case "ready": self.state = "Ready"
            case "transcript": self.transcript = json["text"] as? String ?? ""
            case "token": self.response += (json["text"] as? String ?? "")
            case "tool":
                let name = json["name"] as? String ?? ""
                let status = json["status"] as? String ?? ""
                self.toolStatus = status == "done" ? "" : "\(name)..."
            case "audio":
                if let b64 = json["data"] as? String, let d = Data(base64Encoded: b64) {
                    self.player.play(wav: d)
                }
            case "done": self.state = "Ready"; self.toolStatus = ""
            default: break
            }
        }
    }

    func startRecording(withScreenshot: Bool) {
        state = "Listening..."
        transcript = ""; response = ""
        audioBuffer = Data()
        if withScreenshot { captureScreen() }
        startMic()
    }

    func stopRecording() {
        stopMic()
        guard !audioBuffer.isEmpty else { return }
        state = "Thinking..."
        let wav = makeWav(pcm: audioBuffer, sampleRate: 16000)
        var msg: [String: Any] = ["type": "audio", "data": wav.base64EncodedString()]
        if let s = pendingScreenshot { msg["screenshot"] = s; pendingScreenshot = nil }
        if let data = try? JSONSerialization.data(withJSONObject: msg),
           let str = String(data: data, encoding: .utf8) {
            ws?.send(.string(str)) { _ in }
        }
    }

    private func startMic() {
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let inFormat = input.outputFormat(forBus: 0)
        let outFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: true)!
        let converter = AVAudioConverter(from: inFormat, to: outFormat)!
        input.installTap(onBus: 0, bufferSize: 4096, format: inFormat) { [weak self] buf, _ in
            let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: AVAudioFrameCount(outFormat.sampleRate) * 2)!
            var err: NSError?
            converter.convert(to: outBuf, error: &err) { _, status in
                status.pointee = .haveData; return buf
            }
            if let chans = outBuf.int16ChannelData {
                let frames = Int(outBuf.frameLength)
                let data = Data(bytes: chans[0], count: frames * 2)
                self?.audioBuffer.append(data)
            }
        }
        try? engine.start()
        audioEngine = engine
    }

    private func stopMic() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
    }

    private func captureScreen() {
        // Simple synchronous capture via CGWindowListCreateImage
        let img = CGWindowListCreateImage(.zero, .optionOnScreenOnly, kCGNullWindowID, .bestResolution)
        guard let cg = img else { return }
        let rep = NSBitmapImageRep(cgImage: cg)
        if let png = rep.representation(using: .png, properties: [:]) {
            pendingScreenshot = png.base64EncodedString()
        }
    }

    private func makeWav(pcm: Data, sampleRate: Int) -> Data {
        var header = Data()
        let dataLen = UInt32(pcm.count)
        let totalLen = dataLen + 36
        header.append("RIFF".data(using: .ascii)!)
        header.append(withUnsafeBytes(of: totalLen.littleEndian) { Data($0) })
        header.append("WAVE".data(using: .ascii)!)
        header.append("fmt ".data(using: .ascii)!)
        header.append(withUnsafeBytes(of: UInt32(16).littleEndian) { Data($0) })
        header.append(withUnsafeBytes(of: UInt16(1).littleEndian) { Data($0) })
        header.append(withUnsafeBytes(of: UInt16(1).littleEndian) { Data($0) })
        header.append(withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Data($0) })
        header.append(withUnsafeBytes(of: UInt32(sampleRate * 2).littleEndian) { Data($0) })
        header.append(withUnsafeBytes(of: UInt16(2).littleEndian) { Data($0) })
        header.append(withUnsafeBytes(of: UInt16(16).littleEndian) { Data($0) })
        header.append("data".data(using: .ascii)!)
        header.append(withUnsafeBytes(of: dataLen.littleEndian) { Data($0) })
        return header + pcm
    }
}

// MARK: - Audio playback (queues WAV chunks)

final class AudioStreamPlayer {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var setup = false

    func play(wav: Data) {
        if !setup { setupEngine(); setup = true }
        guard let url = saveTemp(wav: wav),
              let file = try? AVAudioFile(forReading: url) else { return }
        let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
        try? file.read(into: buf)
        player.scheduleBuffer(buf, completionHandler: nil)
        if !player.isPlaying { player.play() }
    }

    private func setupEngine() {
        engine.attach(player)
        let format = AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1)!
        engine.connect(player, to: engine.mainMixerNode, format: format)
        try? engine.start()
    }

    private func saveTemp(wav: Data) -> URL? {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).wav")
        try? wav.write(to: url)
        return url
    }
}

// MARK: - Overlay UI

struct OverlayView: View {
    @ObservedObject var engine: FridayEngine

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Circle()
                    .fill(engine.state == "Listening..." ? Color.red : engine.state == "Thinking..." ? Color.orange : Color.green)
                    .frame(width: 8, height: 8)
                Text(engine.state).font(.system(size: 11, weight: .medium)).foregroundColor(.secondary)
                Spacer()
                if !engine.toolStatus.isEmpty {
                    Text(engine.toolStatus).font(.system(size: 10)).foregroundColor(.blue)
                }
            }
            if !engine.transcript.isEmpty {
                Text(engine.transcript).font(.system(size: 12)).foregroundColor(.secondary).lineLimit(2)
            }
            ScrollView {
                Text(engine.response.isEmpty ? "Hold ⌥Space to talk" : engine.response)
                    .font(.system(size: 14))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 140)
        }
        .padding(16)
        .background(
            VisualEffectBlur().clipShape(RoundedRectangle(cornerRadius: 14))
        )
    }
}

struct VisualEffectBlur: NSViewRepresentable {
    func makeNSView(context: Context) -> NSVisualEffectView {
        let v = NSVisualEffectView()
        v.material = .hudWindow
        v.blendingMode = .behindWindow
        v.state = .active
        return v
    }
    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {}
}
