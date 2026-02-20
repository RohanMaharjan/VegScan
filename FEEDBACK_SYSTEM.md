# 🎯 Feedback System for Model Improvement

## Overview
The feedback system allows users to correct misclassifications in real-time, creating a dataset for future model retraining.

## How It Works

### User Flow
1. **Capture & Scan**: User takes image and AI predicts the vegetable
2. **Two Buttons Appear**: Below valid predictions (>60% confidence)
   - **✓ Right** → Image was classified correctly
   - **✗ Wrong** → Image was misclassified
3. **Modal Form** (for Wrong predictions):
   - Shows the image and what the model predicted
   - Dropdown list of 13 vegetables
   - User selects the correct vegetable
4. **Submission**:
   - Image is saved with timestamp
   - Metadata stored in JSON file
   - Success toast confirms submission

### Data Structure
```
feedback/
├── images/
│   ├── tomato_20260215_143022_456.jpg
│   ├── potato_20260215_143045_789.jpg
│   └── ...
└── feedback_data.json
```

### Feedback Metadata (`feedback_data.json`)
```json
[
  {
    "timestamp": "2026-02-15T14:30:22.456789",
    "image_filename": "tomato_20260215_143022_456.jpg",
    "predicted_label": "potato",
    "correct_label": "tomato",
    "image_path": "feedback/images/tomato_20260215_143022_456.jpg"
  }
]
```

## Frontend Components

### Buttons (scanner.html)
- Appear only for valid predictions (confidence ≥ 60%)
- Right button: Shows success toast
- Wrong button: Opens feedback modal

### Feedback Modal
- Image preview
- Model's original prediction
- Dropdown with all 13 vegetables
- Cancel & Submit buttons
- Closes on backdrop click

### Success Toast
- Shows confirmation message
- Auto-hides after 3 seconds
- Appears bottom-right

## Backend Components

### API Endpoint: `/api/feedback/`
**POST Request**
```
Form Data:
- image: File blob
- predicted_label: String (what model predicted)
- correct_label: String (what user selected)
```

**Response**
```json
{
  "success": true,
  "message": "Thank you! Image saved as tomato for future training.",
  "entry": {
    "timestamp": "2026-02-15T...",
    "image_filename": "...",
    ...
  }
}
```

### View: `save_feedback_api(request)`
**Location**: `scanner/views.py`

**Features**:
- Validates vegetable name against class list
- Creates `feedback/` directory if missing
- Generates unique filename with timestamp (millisecond precision)
- Saves image at 95% JPEG quality
- Appends metadata to JSON
- Returns validation errors if needed

**Error Handling**:
- Missing image/label → 400 Bad Request
- Invalid vegetable name → 400 with list of valid options
- File I/O errors → 500 with error message

## Vegetables (13 Classes)
1. Bean
2. Broccoli
3. Bottle Gourd
4. Brinjal
5. Bitter Gourd
6. Cabbage
7. Capsicum
8. Carrot
9. Cauliflower
10. Potato
11. Pumpkin
12. Radish
13. Tomato

## Using Feedback Data for Retraining

When you have collected enough feedback images, you can:

1. **Copy images to training folder**:
   ```
   data_split/train/{vegetable_name}/
   ```

2. **Run retraining script**:
   ```bash
   python train_model_vgg16.py
   # or
   python train_model.py
   ```

3. **Update models**:
   - Fine-tune on new feedback dataset
   - Validate on test set
   - Deploy improved models

## Storage Location
- **Frontend submission**: Triggered from scanner.html
- **Backend processing**: `scanner/views.py` → `save_feedback_api()`
- **Data saved**: `{PROJECT_ROOT}/feedback/` directory

## Statistics
- **Image Quality**: Saved at 95% JPEG quality (good balance)
- **Filename**: `{correct_label}_{YYYYMMDD_HHMMSS_mmm}.jpg`
- **Metadata**: JSON format for easy programmatic access
- **Validation**: Ensures correct_label is from known categories

## Example Workflow
```
User captures potato image
↓
AI predicts: "Carrot" (67% confidence)
↓
User sees "✗ Wrong" button
↓
Clicks "Wrong" → Modal opens
↓
User selects "Potato" from dropdown
↓
Clicks "Submit Feedback ✓"
↓
Image saved: feedback/images/potato_20260215_143022_456.jpg
Metadata saved: feedback/feedback_data.json
↓
Toast: "✓ Image classified successfully! Thank you for helping improve our AI"
```

## Future Enhancements
- [ ] User authentication for feedback attribution
- [ ] Confidence tracking (only save low-confidence misclassifications)
- [ ] Batch feedback processing dashboard
- [ ] Automatic model retraining when threshold reached
- [ ] User feedback statistics and leaderboard
- [ ] Feedback categorization (partial, subtle differences, etc.)
