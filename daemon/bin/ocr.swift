// Minimal CLI wrapper around Apple's Vision OCR.
// Build: xcrun swiftc -O ocr.swift -o ocr (done automatically by scripts/setup.sh)
import AppKit
import Vision

let args = CommandLine.arguments
guard args.count == 2 else {
    print("usage: ocr <image_path>")
    exit(1)
}

guard let image = NSImage(contentsOf: URL(fileURLWithPath: args[1])),
      let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    print("OCR_ERROR: cannot load image")
    exit(1)
}

let request = VNRecognizeTextRequest { req, _ in
    guard let observations = req.results as? [VNRecognizedTextObservation] else { return }
    let lines = observations.compactMap { $0.topCandidates(1).first?.string }
    print(lines.joined(separator: "\n"))
}
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true

let handler = VNImageRequestHandler(cgImage: cg, options: [:])
do {
    try handler.perform([request])
} catch {
    print("OCR_ERROR: \(error.localizedDescription)")
    exit(1)
}
