# 🖐️ ISL Live Real-Time Translation

A production-ready Streamlit web application for real-time Indian Sign Language (ISL) gesture recognition using TensorFlow Lite and MediaPipe.

## ✨ Features

✅ **Real-Time Gesture Recognition** - Live WebRTC video, MediaPipe hand detection, 5 ISL gestures
✅ **Interactive Interface** - Camera feed, adjustable settings, real-time results
✅ **Production Ready** - Error handling, robust loading, TFLite optimization

## 🎯 Supported Gestures (5 ISL Signs)
1. Help You | 2. Congratulation | 3. Hi How Are You | 4. I Am Hungry | 5. Take Care of Yourself

## 🚀 Quick Start

### Live Web App: https://islrealtimeapp-1212.streamlit.app/
No installation required! Visit and allow camera access.

### Local Setup
```bash
git clone https://github.com/Adarsh-singh-somvanshi/isl_realtime_app.git
cd isl_realtime_app
pip install -r requirements.txt
streamlit run app.py
```

## 📋 Requirements
- Python 3.8+
- Webcam
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🐛 Troubleshooting

**Error loading resources**: Refresh page in 10 seconds
**Camera not working**: Allow permissions, check if in use, try different browser
**Gestures not detected**: Ensure hand visible, adjust lighting, lower confidence threshold

## 📊 Recent Fixes (Nov 29, 2025)

✅ Fixed corrupted label_map.pkl file
✅ Changed from Keras H5 to TensorFlow Lite
✅ Robust text-based label loading
✅ Fixed sidebar AttributeError
✅ Production-ready deployment

## 🚀 Deployment on Streamlit Cloud
1. Fork repository
2. Go to https://share.streamlit.io
3. Select repository
4. Deploy!

## 📚 Resources
- [Streamlit Docs](https://docs.streamlit.io)
- [MediaPipe Solutions](https://google.github.io/mediapipe/)
- [TensorFlow Lite](https://www.tensorflow.org/lite)

## 👤 Author
**Adarsh Singh Somvanshi**
- GitHub: [@Adarsh-singh-somvanshi](https://github.com/Adarsh-singh-somvanshi)

---
**Made with ❤️ for accessibility and inclusion**
*Last Updated: November 29, 2025 | Status: ✅ Production Ready*
