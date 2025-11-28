# 🧠 LỘ TRÌNH PHÁT TRIỂN ML CHO GOGAME

## 📋 TỔNG QUAN

Tài liệu này mô tả chi tiết hướng phát triển Machine Learning cho GoGame, tập trung vào việc tạo ra các tính năng phân tích thế cờ thông minh và trực quan, giúp người chơi học hỏi và cải thiện kỹ năng.

---

## 🎯 MỤC TIÊU VÀ GIÁ TRỊ

### Mục tiêu chính:
1. **Phân tích thế cờ chi tiết** - Đánh giá vị trí, nhận biết điểm mạnh/yếu
2. **Phát hiện mối đe dọa** - Nhận biết các vùng bị đe dọa, nhóm quân yếu
3. **Cơ hội tấn công** - Xác định các vùng có thể tấn công, giành lấy
4. **Nhận biết ý định** - Dự đoán ý định của đối thủ, chiến thuật đang sử dụng
5. **Trực quan hóa** - Hiển thị kết quả trên UI bằng cách khoanh vùng, đánh dấu, chú thích

### Giá trị mang lại:
- ✅ **Học tập hiệu quả**: Người chơi hiểu rõ hơn về thế cờ
- ✅ **Tính năng độc đáo**: Không chỉ gợi ý nước đi mà còn giải thích tại sao
- ✅ **Monetization**: Premium feature có giá trị thực sự
- ✅ **Khác biệt với đối thủ**: Tính năng mà các game cờ vây khác chưa có

---

## 🏗️ KIẾN TRÚC ML MỚI

### 1. Multi-Task Learning Architecture

Thay vì chỉ có Policy/Value networks, chúng ta sẽ xây dựng một hệ thống **Multi-Task Learning** với các model chuyên biệt:

```
┌─────────────────────────────────────────────────────────┐
│              INPUT: Board State (17 planes)              │
│  - Stone positions (Black/White)                         │
│  - Liberties, Groups, Territory, etc.                   │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Shared Feature Extractor    │
        │   (CNN Backbone - ResNet-like)│
        └───────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Threat       │ │ Attack       │ │ Intent       │
│ Detection    │ │ Opportunity   │ │ Recognition  │
│ Head        │ │ Head          │ │ Head         │
└──────────────┘ └──────────────┘ └──────────────┘
        │               │               │
        ▼               ▼               ▼
  Threat Map      Attack Map      Intent Map
  (Heatmap)       (Heatmap)       (Heatmap)
```

### 2. Các Model Cần Train

#### 2.1. **Threat Detection Model** (Nhận biết mối đe dọa)
- **Input**: Board state (17 planes)
- **Output**: Threat heatmap (board_size × board_size)
  - Giá trị cao = vùng bị đe dọa cao
  - Nhận biết: nhóm quân yếu, vùng có thể bị bao vây, mắt giả
- **Loss function**: Binary cross-entropy với ground truth từ game analysis

#### 2.2. **Attack Opportunity Model** (Cơ hội tấn công)
- **Input**: Board state (17 planes)
- **Output**: Attack heatmap (board_size × board_size)
  - Giá trị cao = vùng có thể tấn công
  - Nhận biết: nhóm đối thủ yếu, vùng có thể xâm nhập, cơ hội bắt quân
- **Loss function**: Binary cross-entropy

#### 2.3. **Intent Recognition Model** (Nhận biết ý định)
- **Input**: Board state + last N moves (17 + N planes)
- **Output**: Intent classification + heatmap
  - Classification: ["territory", "attack", "defense", "connection", "cut"]
  - Heatmap: Vùng liên quan đến ý định
- **Loss function**: Multi-task (classification + regression)

#### 2.4. **Position Evaluation Model** (Nâng cấp Value Network)
- **Input**: Board state (17 planes)
- **Output**: 
  - Win probability (scalar)
  - Territory map (heatmap)
  - Influence map (heatmap)
- **Loss function**: MSE cho win prob + L1 cho maps

---

## 📊 DỮ LIỆU TRAINING

### 1. Data Sources

#### 1.1. Self-Play Games (Chính)
- Sử dụng MCTS engine hiện có để generate games
- Mỗi game → nhiều training samples (mỗi move)
- Ưu điểm: Dữ liệu phong phú, đa dạng

#### 1.2. Professional Games (Bổ sung)
- Download SGF files từ KGS, OGS, GoGoD
- Parse và extract positions
- Ưu điểm: Chất lượng cao, chiến thuật thực tế

#### 1.3. Annotated Positions (Ground Truth)
- Tạo ground truth bằng cách:
  - Chạy MCTS với nhiều playouts để đánh giá threats
  - Sử dụng rule-based heuristics để label attacks
  - Human annotation cho một số positions quan trọng

### 2. Data Format

```python
{
    "board_state": Tensor[17, board_size, board_size],
    "threat_map": Tensor[board_size, board_size],  # 0-1, threat level
    "attack_map": Tensor[board_size, board_size],  # 0-1, attack opportunity
    "intent": {
        "type": "attack" | "defense" | "territory" | "connection" | "cut",
        "confidence": float,
        "region": [[x1, y1], [x2, y2], ...]  # Bounding box
    },
    "evaluation": {
        "win_probability": float,
        "territory_map": Tensor[board_size, board_size],
        "influence_map": Tensor[board_size, board_size]
    },
    "metadata": {
        "game_id": str,
        "move_number": int,
        "board_size": int,
        "current_player": "B" | "W"
    }
}
```

### 3. Data Augmentation

- **Rotation**: Xoay bàn cờ 90°, 180°, 270°
- **Reflection**: Lật ngang/dọc
- **Color swap**: Đổi màu (Black ↔ White)
- **Noise**: Thêm noise nhỏ vào features

---

## 🚀 LỘ TRÌNH TRIỂN KHAI

### **PHASE 1: Data Collection & Preparation** (1-2 tuần)

#### Bước 1.1: Tạo Data Collection Pipeline
- [ ] Tạo script `collect_self_play_data.py`
- [ ] Chạy MCTS self-play để generate 10,000+ games
- [ ] Lưu raw data vào MongoDB/S3

#### Bước 1.2: Tạo Ground Truth Labels
- [ ] Implement `threat_analyzer.py` (rule-based)
- [ ] Implement `attack_analyzer.py` (rule-based)
- [ ] Label data với rule-based heuristics
- [ ] Validate labels với một số positions thủ công

#### Bước 1.3: Data Preprocessing
- [ ] Tạo `data_loader.py` để load và preprocess
- [ ] Implement data augmentation
- [ ] Split train/val/test (80/10/10)
- [ ] Tạo PyTorch Dataset class

**Deliverables:**
- ✅ Dataset với 50,000+ labeled positions
- ✅ Data loader và preprocessing pipeline
- ✅ Validation script

---

### **PHASE 2: Model Architecture** (1 tuần)

#### Bước 2.1: Shared Backbone
- [ ] Implement `shared_backbone.py` (ResNet-like CNN)
- [ ] Test với dummy data
- [ ] Benchmark performance

#### Bước 2.2: Task-Specific Heads
- [ ] Implement `threat_head.py`
- [ ] Implement `attack_head.py`
- [ ] Implement `intent_head.py`
- [ ] Implement `evaluation_head.py`

#### Bước 2.3: Multi-Task Model
- [ ] Combine backbone + heads trong `multi_task_model.py`
- [ ] Implement forward pass
- [ ] Test end-to-end

**Deliverables:**
- ✅ Model architecture hoàn chỉnh
- ✅ Unit tests cho từng component
- ✅ Model size < 50MB (lightweight)

---

### **PHASE 3: Training Pipeline** (2-3 tuần)

#### Bước 3.1: Training Script
- [ ] Tạo `train_multi_task.py`
- [ ] Implement loss functions (weighted multi-task loss)
- [ ] Implement training loop với validation
- [ ] Add TensorBoard logging

#### Bước 3.2: Hyperparameter Tuning
- [ ] Learning rate scheduling
- [ ] Loss weights cho các tasks
- [ ] Batch size, optimizer (Adam/AdamW)
- [ ] Early stopping

#### Bước 3.3: Model Evaluation
- [ ] Metrics: Accuracy, Precision, Recall cho classification
- [ ] Metrics: MSE, MAE cho regression
- [ ] Visual evaluation: Plot heatmaps
- [ ] Compare với rule-based baselines

**Deliverables:**
- ✅ Trained model với validation accuracy > 70%
- ✅ Training logs và metrics
- ✅ Model checkpoint

---

### **PHASE 4: Inference Service** (1 tuần)

#### Bước 4.1: Model Serving
- [ ] Tạo `ml_analysis_service.py` trong backend
- [ ] Load model và implement inference
- [ ] Optimize với ONNX/TorchScript (optional)
- [ ] Add caching cho performance

#### Bước 4.2: API Endpoints
- [ ] `POST /ml/analyze-position` - Phân tích thế cờ
- [ ] `POST /ml/detect-threats` - Phát hiện mối đe dọa
- [ ] `POST /ml/find-attacks` - Tìm cơ hội tấn công
- [ ] `POST /ml/recognize-intent` - Nhận biết ý định

#### Bước 4.3: Response Format
```json
{
    "threats": {
        "heatmap": [[0.1, 0.3, ...], ...],  // board_size × board_size
        "regions": [
            {
                "type": "weak_group",
                "positions": [[3, 3], [3, 4], [4, 3]],
                "severity": 0.8,
                "description": "Nhóm quân đen yếu, thiếu mắt"
            }
        ]
    },
    "attacks": {
        "heatmap": [[0.2, 0.5, ...], ...],
        "opportunities": [
            {
                "type": "capture",
                "position": [5, 5],
                "confidence": 0.9,
                "description": "Có thể bắt 3 quân trắng"
            }
        ]
    },
    "intent": {
        "primary_intent": "attack",
        "confidence": 0.85,
        "regions": [
            {
                "type": "attack",
                "positions": [[7, 7], [7, 8], [8, 7]],
                "description": "Đối thủ đang tấn công nhóm quân đen"
            }
        ]
    },
    "evaluation": {
        "win_probability": 0.65,
        "territory_map": [[0.1, 0.2, ...], ...],
        "influence_map": [[0.3, 0.4, ...], ...]
    }
}
```

**Deliverables:**
- ✅ ML service hoàn chỉnh
- ✅ API endpoints với response format chuẩn
- ✅ Performance: < 500ms per request (với caching)

---

### **PHASE 5: Frontend Integration** (1-2 tuần)

#### Bước 5.1: UI Components
- [ ] Tạo `MLAnalysisPanel.jsx` component
- [ ] Implement heatmap visualization (Canvas/SVG)
- [ ] Implement region highlighting
- [ ] Add tooltips với descriptions

#### Bước 5.2: Visualization Features
- [ ] **Threat visualization**: 
  - Màu đỏ gradient cho vùng bị đe dọa
  - Độ đậm = mức độ đe dọa
- [ ] **Attack visualization**:
  - Màu xanh lá cho cơ hội tấn công
  - Mũi tên chỉ hướng tấn công
- [ ] **Intent visualization**:
  - Icon khác nhau cho mỗi loại intent
  - Text annotations
- [ ] **Evaluation visualization**:
  - Territory overlay (màu xanh/đỏ)
  - Influence gradient

#### Bước 5.3: User Interaction
- [ ] Toggle on/off từng loại visualization
- [ ] Click vào region để xem chi tiết
- [ ] Animation khi chuyển đổi giữa các moves
- [ ] Settings panel để điều chỉnh opacity, colors

**Deliverables:**
- ✅ UI components hoàn chỉnh
- ✅ Smooth visualization với performance tốt
- ✅ User-friendly interface

---

### **PHASE 6: Premium Feature Integration** (1 tuần)

#### Bước 6.1: Shop Integration
- [ ] Thêm "ML Analysis" vào shop (50 coins)
- [ ] Update `premium_service.py` để gọi ML service
- [ ] Add usage tracking

#### Bước 6.2: Match Integration
- [ ] Add "Analyze Position" button trong game UI
- [ ] Show analysis results trong side panel
- [ ] Save analysis results để xem lại sau

#### Bước 6.3: Statistics
- [ ] Track số lần sử dụng ML analysis
- [ ] Show trong user statistics
- [ ] Leaderboard cho "most improved" (dựa trên analysis usage)

**Deliverables:**
- ✅ Premium feature hoàn chỉnh
- ✅ Monetization working
- ✅ User tracking và statistics

---

## 🔧 CHI TIẾT IMPLEMENTATION

### 1. Model Architecture Code Structure

```
src/ml/
├── models/
│   ├── __init__.py
│   ├── shared_backbone.py      # ResNet-like backbone
│   ├── threat_head.py          # Threat detection head
│   ├── attack_head.py          # Attack opportunity head
│   ├── intent_head.py          # Intent recognition head
│   ├── evaluation_head.py      # Position evaluation head
│   └── multi_task_model.py     # Combined model
├── training/
│   ├── __init__.py
│   ├── data_collector.py       # Self-play data collection
│   ├── label_generator.py      # Ground truth generation
│   ├── dataset.py              # PyTorch Dataset
│   ├── train_multi_task.py     # Training script
│   └── evaluator.py            # Model evaluation
├── inference/
│   ├── __init__.py
│   ├── analyzer.py             # Main analysis service
│   └── postprocessor.py        # Process model outputs
└── utils/
    ├── visualization.py        # Heatmap generation
    └── metrics.py              # Evaluation metrics
```

### 2. Backend Service Structure

```
backend/app/services/
├── ml_service.py               # Main ML service (existing)
└── ml_analysis_service.py     # NEW: Position analysis service

backend/app/routers/
└── ml.py                       # Update với analysis endpoints
```

### 3. Frontend Components

```
frontend-web/src/components/
├── MLAnalysisPanel.jsx         # NEW: Main analysis panel
├── ThreatVisualization.jsx     # NEW: Threat heatmap
├── AttackVisualization.jsx     # NEW: Attack opportunities
├── IntentVisualization.jsx     # NEW: Intent display
└── EvaluationOverlay.jsx       # NEW: Territory/influence overlay
```

---

## 📈 METRICS & EVALUATION

### Training Metrics:
- **Threat Detection**: Precision, Recall, F1-score
- **Attack Detection**: Precision, Recall, F1-score
- **Intent Recognition**: Accuracy, Confusion matrix
- **Position Evaluation**: MSE, MAE vs ground truth

### Business Metrics:
- **Usage rate**: % users sử dụng ML analysis
- **Retention**: Users sử dụng analysis có chơi lâu hơn không?
- **Revenue**: Coins spent on ML analysis
- **User satisfaction**: Feedback scores

---

## 🎓 KIẾN THỨC CẦN THIẾT

### Machine Learning:
- ✅ PyTorch basics
- ✅ CNN architecture (ResNet)
- ✅ Multi-task learning
- ✅ Loss functions (BCE, MSE, etc.)
- ✅ Training loops, optimizers

### Go Game Knowledge:
- ✅ Threat detection rules
- ✅ Attack patterns
- ✅ Strategic concepts (territory, influence, etc.)

### Software Engineering:
- ✅ API design
- ✅ Data pipelines
- ✅ Model serving
- ✅ Frontend visualization

---

## ⚠️ CHALLENGES & SOLUTIONS

### Challenge 1: Data Quality
**Problem**: Ground truth labels khó tạo chính xác
**Solution**: 
- Bắt đầu với rule-based heuristics
- Iteratively improve với human feedback
- Sử dụng MCTS evaluation làm weak supervision

### Challenge 2: Model Size
**Problem**: Model quá lớn → slow inference
**Solution**:
- Lightweight architecture (< 50MB)
- Model quantization (INT8)
- ONNX conversion
- Caching frequent positions

### Challenge 3: Real-time Performance
**Problem**: Inference quá chậm cho real-time
**Solution**:
- Async processing
- Background jobs cho heavy analysis
- Progressive loading (show partial results)

### Challenge 4: Accuracy
**Problem**: Model chưa đủ chính xác
**Solution**:
- More training data
- Better architecture
- Ensemble methods
- Fine-tuning với human-annotated data

---

## 🎯 SUCCESS CRITERIA

### Technical:
- ✅ Model accuracy > 70% trên validation set
- ✅ Inference time < 500ms (với caching)
- ✅ Model size < 50MB
- ✅ API response time < 1s

### Product:
- ✅ 30%+ users sử dụng ML analysis
- ✅ Positive user feedback (> 4/5 stars)
- ✅ Revenue từ ML features > 20% tổng premium revenue

### Learning:
- ✅ Team hiểu được multi-task learning
- ✅ Có thể extend model với tasks mới
- ✅ Documentation đầy đủ cho future development

---

## 📚 TÀI LIỆU THAM KHẢO

1. **AlphaGo Paper**: DeepMind's approach to Go AI
2. **Multi-Task Learning**: Caruana (1997)
3. **Go Strategy Books**: "Lessons in the Fundamentals of Go" by Kageyama
4. **PyTorch Tutorials**: Official PyTorch documentation
5. **Computer Go**: Sensei's Library, GoBase.org

---

## 🚦 NEXT STEPS

1. **Bắt đầu với Phase 1**: Data Collection
2. **Review architecture**: Đảm bảo phù hợp với requirements
3. **Setup development environment**: PyTorch, CUDA (nếu có GPU)
4. **Create GitHub issues**: Break down tasks thành issues nhỏ
5. **Start coding!** 🚀

---

**Tài liệu này sẽ được update thường xuyên khi có tiến triển!**

