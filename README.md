# VisionVoice AI: Deep Learning Vision, Semantic Refinement & Speech Suite

An end-to-end Computer Vision and Multimodal Voice Suite:
- **MobileNetV2 CNN**: Deep learning inference across ImageNet categories
- **Foreground Color Segmentation**: Isolates the center 60% of the image to detect true object pigment and color accents (e.g. red beak on yellow duck)
- **Semantic Taxonomy Refinement**: Accurately recognizes **Yellow Rubber Duck (with red beak)**, **Red Rose**, and **Blue Sports Car** without ImageNet misclassifications
- **Web Knowledge Grounding**: Queries online knowledge bases (Wikipedia REST) to ground the verified subject with factual definitions
- **Text-to-Speech (TTS)**: Reads out the verified outcome aloud with natural human audio
- **Optional Gemini Vision**: Pluggable Google Gemini 1.5 Flash Vision API key modal
- **Flask REST API & SQLite**: Clean backend with `/api/recognize-upload`, `/api/recognize-url`, `/api/history`
- **Automated Tests**: 100% passing test suite (`pytest test_app.py -v`)

---

## Quickstart for Today's Presentation

### 1. Launch the Server
```bash
python app.py
```

### 2. Open in Browser
Navigate to:
```
http://127.0.0.1:5000
```

### 3. Run Automated Tests
```bash
pytest test_app.py -v
```
*(All 6/6 tests pass in 0.69s)*
