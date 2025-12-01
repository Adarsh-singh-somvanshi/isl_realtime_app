# ISL Real-time Translation Mobile App - Flutter Setup Guide

## 📱 Project Overview

A complete production-ready Flutter mobile app with:
- **Real-time camera feed** using `camera` plugin
- **TensorFlow Lite inference** for ISL gesture recognition
- **Background isolate processing** for smooth 15-30 FPS performance
- **Modern, responsive UI** with smooth animations
- **Multi-screen architecture** (Home, Live Camera, Settings)
- **Optimized preprocessing** and model inference

---

## 🚀 Quick Start

### Prerequisites
- Flutter 3.7+ (verified on Windows with Dart 3.10+)
- Android SDK (API 21+) or iOS 11+
- Your `.tflite` model file: `isl_gesture_model.tflite`
- Label map JSON: `labels.json` (converted from `.pkl`)

### Step 1: Create Flutter Project

```bash
flutter create isl_realtime_mobile
cd isl_realtime_mobile
```

### Step 2: Update pubspec.yaml

See `pubspec.yaml` file in this branch.

```bash
flutter pub get
```

### Step 3: Add Model & Assets

```
isl_realtime_mobile/
├── assets/
│   └── model/
│       ├── isl_gesture_model.tflite  (your TFLite model)
│       └── labels.json              (converted label map)
└── lib/
    ├── main.dart
    ├── ...
```

### Step 4: Copy All Dart Files

Refer to the individual `.dart` files in this branch and place them in:
- `lib/main.dart`
- `lib/core/app_theme.dart`
- `lib/core/app_routes.dart`
- `lib/data/ml/label_repository.dart`
- `lib/data/ml/isl_interpreter.dart`
- `lib/domain/entities/prediction.dart`
- `lib/presentation/home/home_screen.dart`
- `lib/presentation/live/live_camera_screen.dart`
- `lib/presentation/live/live_viewmodel.dart`
- `lib/presentation/settings/settings_screen.dart`
- `lib/presentation/widgets/animated_title.dart`
- `lib/presentation/widgets/rounded_button.dart`

### Step 5: Run on Device

```bash
# Android
flutter run -d <device-id>

# Or use emulator
flutter emulators --launch <emulator-name>
flutter run
```

---

## 🔄 Architecture Overview

```
lib/
├── main.dart                    # App entry & provider setup
├── core/
│   ├── app_theme.dart          # Light/dark theme definitions
│   └── app_routes.dart         # Route constants
├── data/
│   └── ml/
│       ├── isl_interpreter.dart # TFLite model wrapper + background isolate
│       └── label_repository.dart # Load & manage labels from JSON
├── domain/
│   ├── entities/
│   │   └── prediction.dart      # Prediction data model
│   └── usecases/
│       └── (future ML use cases)
└── presentation/
    ├── home/
    │   └── home_screen.dart     # Welcome & start button
    ├── live/
    │   ├── live_camera_screen.dart  # Full-screen camera + overlay
    │   └── live_viewmodel.dart      # State & isolate orchestration
    ├── settings/
    │   └── settings_screen.dart  # Theme, resolution, model info
    └── widgets/
        ├── animated_title.dart   # Lottie intro animation
        └── rounded_button.dart   # Reusable button component
```

---

## 🧠 ML Pipeline

### Background Isolate Pattern

1. **Main UI Thread**: Captures camera frames at ~30 FPS
2. **Isolate Worker**: Runs model inference without blocking UI
3. **Communication**: Lightweight message passing (frame data → prediction)

### Frame Processing

- **Capture**: `camera` plugin streams YUV/RGBA frames
- **Resize**: Scale to model input (e.g., 224×224)
- **Normalize**: Convert pixel values (0-255) to (0-1) range
- **Inference**: Run TFLite interpreter
- **Post-process**: Map output indices to label text

---

## 📋 File Descriptions

### Core Files

**lib/main.dart**
- MultiProvider setup with Provider for state management
- Theme switching (Light/Dark/System)
- Route configuration

**lib/core/app_theme.dart**
- Material3 theme with Indigo seed color
- Separate light & dark themes

**lib/core/app_routes.dart**
- Route string constants
- Easy navigation management

### ML Layer

**lib/data/ml/isl_interpreter.dart**
- TFLite model loading & initialization
- Background isolate entry point
- Frame preprocessing (YUV→RGB, resizing, normalization)
- Inference execution
- Output post-processing

**lib/data/ml/label_repository.dart**
- JSON label loading
- Index-to-label mapping

**lib/domain/entities/prediction.dart**
- Data class for predictions
- Immutable structure for state management

### UI Layer

**lib/presentation/live/live_viewmodel.dart**
- Provider ChangeNotifier for reactive updates
- Isolate lifecycle management
- Frame submission & result handling
- Confidence threshold filtering

**lib/presentation/home/home_screen.dart**
- Welcome screen with animated title
- Start camera button
- Theme mode selector

**lib/presentation/live/live_camera_screen.dart**
- Full-screen camera preview
- Real-time prediction overlay (bottom panel)
- Animated confidence ring
- Back button & error handling

**lib/presentation/settings/settings_screen.dart**
- Theme selection (Light/Dark/System)
- Camera resolution options
- Model information display

---

## 🔧 Customization

### Adjusting Input Shape

If your model expects different dimensions, edit `isl_interpreter.dart`:

```dart
// Change these based on your model
final inputH = _inputShape[1];  // Height (default: 224)
final inputW = _inputShape[2];  // Width (default: 224)
```

### Changing Confidence Threshold

Edit `live_viewmodel.dart`:

```dart
confidenceThreshold = 0.55;  // Adjust this value
```

### Adjusting FPS Performance

In `live_camera_screen.dart`, throttle frame submission:

```dart
static int _frameCounter = 0;

void _onFrame(CameraImage image) {
  if (_frameCounter++ % 2 == 0) {  // Process every 2nd frame
    vm.submitFrame(...);
  }
}
```

---

## 📊 Performance Targets

- **Camera FPS**: 30 FPS (preview smooth & responsive)
- **Inference FPS**: 15 FPS (model inference in background)
- **Frame Latency**: <100ms (from capture to prediction display)
- **Memory**: ~80-150 MB (depending on model size)

---

## 🐛 Troubleshooting

### Model Loading Fails

1. Verify `isl_gesture_model.tflite` exists in `assets/model/`
2. Check `pubspec.yaml` includes: `assets: - assets/model/`
3. Run: `flutter clean && flutter pub get`

### Labels Not Loading

1. Ensure `labels.json` is in `assets/model/`
2. Verify JSON format: `{"0": "label_name", "1": "another_label"}`
3. Check character encoding (UTF-8)

### Camera Permission Denied (Android)

1. Add to `android/app/src/main/AndroidManifest.xml`:
   ```xml
   <uses-permission android:name="android.permission.CAMERA" />
   ```
2. Rebuild: `flutter clean && flutter run`

### Performance Issues / Lag

1. Reduce camera resolution: `ResolutionPreset.low` or `.medium`
2. Throttle frame processing (process every Nth frame)
3. Profile with DevTools: `flutter pub global activate devtools && devtools`

---

## 📝 Converting `.pkl` Labels to JSON

**On your dev machine (Python):**

```python
import pickle, json

with open('label_map.pkl', 'rb') as f:
    label_map = pickle.load(f)  # {0: 'help_you', 1: 'congratulation', ...}

with open('labels.json', 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)
```

Place the resulting `labels.json` in `assets/model/labels.json`.

---

## 🎯 Next Steps

1. ✅ Clone the `flutter-mobile-app` branch
2. ✅ Create Flutter project
3. ✅ Copy all `.dart` files from this branch
4. ✅ Add your `.tflite` model and `labels.json`
5. ✅ Update `pubspec.yaml`
6. ✅ Run `flutter pub get`
7. ✅ Connect device/emulator
8. ✅ Run `flutter run`

---

## 📚 Resources

- Flutter Docs: https://flutter.dev/docs
- TensorFlow Lite for Flutter: https://pub.dev/packages/tflite_flutter
- Camera Plugin: https://pub.dev/packages/camera
- Provider State Management: https://pub.dev/packages/provider

---

**Happy coding! 🚀**
