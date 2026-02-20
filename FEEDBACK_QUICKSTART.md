# 🚀 Feedback System - Quick Start Guide

## What You Just Got

Your VegScan AI now has a **crowdsourced feedback system** that lets users correct misclassifications on-the-fly. These corrections are automatically saved for future model retraining!

## User Experience

### On the Scanner Page:
1. User captures image of vegetable
2. AI predicts with confidence %
3. Two buttons appear below the prediction:
   - **✓ Right** – Confirms correct prediction (shows success message)
   - **✗ Wrong** – Opens modal to report mistake

### When User Clicks "Wrong":
1. Modal opens showing:
   - The captured image
   - What the AI predicted
   - Dropdown list of 13 vegetables
2. User selects the **correct** vegetable name
3. Clicks **"Submit Feedback ✓"**
4. Success toast confirms: "Image classified successfully! Thank you for helping improve our AI"

### Behind the Scenes:
- Image is saved to `feedback/images/` with timestamp
- Metadata recorded in `feedback/feedback_data.json`
- Ready for future retraining!

## File Locations

```
feedback/
├── images/
│   ├── tomato_20260215_143022_456.jpg
│   ├── potato_20260215_143045_789.jpg
│   └── (user-corrected images)
└── feedback_data.json  (metadata about corrections)
```

## Managing Feedback Data

### View Statistics
```bash
python manage_feedback.py
# Select option 1 to see:
# - Total feedback collected
# - How many were misclassifications
# - Breakdown by vegetable
```

### Copy to Training Folder
```bash
python manage_feedback.py
# Select option 3 to:
# - Move feedback images to data_split/train/
# - Ready for model retraining
```

### Clear Feedback
```bash
python manage_feedback.py
# Select option 4 to:
# - Archive current feedback
# - Start fresh
```

## Retraining with Feedback

After collecting enough feedback (e.g., 100+ images):

```bash
# 1. Copy feedback images to training folder
python manage_feedback.py  # Select option 3

# 2. Retrain your models
python train_model_vgg16.py  # or python train_model.py

# 3. Models will improve with new data!
```

## Technical Details

### API Endpoint
- URL: `/api/feedback/`
- Method: POST
- Sends: image blob + predicted label + user's correct label
- Returns: success confirmation with metadata

### Vegetable Classes (13)
1. Bean          7. Capsicum
2. Broccoli      8. Carrot
3. Bottle Gourd  9. Cauliflower
4. Brinjal       10. Potato
5. Bitter Gourd  11. Pumpkin
6. Cabbage       12. Radish
                 13. Tomato

### File Format
Each image saved with format:
```
{correct_vegetable_name}_{YYYYMMDD_HHMMSS_milliseconds}.jpg
```

Metadata stored in JSON:
```json
{
  "timestamp": "2026-02-15T14:30:22.456789",
  "image_filename": "tomato_20260215_143022_456.jpg",
  "predicted_label": "potato",
  "correct_label": "tomato",
  "image_path": "feedback/images/tomato_20260215_143022_456.jpg"
}
```

## Why This Matters

✅ **Active Learning**: Users help train the model while using it
✅ **Crowd-sourced Data**: Real-world corrections improve accuracy
✅ **Easy Integration**: Simply copy images to training folder
✅ **No Manual Labeling**: User corrections ARE the labels
✅ **Future-proof**: Build your training dataset gradually

## Example Workflow

```
User 1: Captures tomato, AI says "Potato" → Corrects to "Tomato" ✓
User 2: Captures carrot, AI says "Orange Gourd" → Corrects to "Carrot" ✓
User 3: Captures broccoli, AI says "Cabbage" → Corrects to "Broccoli" ✓

(After collecting 50+ examples of each mistake)

Run: python manage_feedback.py → Copy to training folder
Run: python train_model_vgg16.py → Fine-tune models
Result: Model now accurately handles edge cases it was struggling with!
```

## Troubleshooting

**Q: Feedback button not showing?**
A: Only shows for valid predictions (confidence ≥ 60%)

**Q: Modal not opening?**
A: Browser console (F12) for errors; may need to allow popup windows

**Q: Images not saving?**
A: Check folder permissions on `feedback/` directory

**Q: Want to disable feedback temporarily?**
A: Comment out the feedback buttons in `templates/scanner.html` lines ~150-160

## Next Steps

1. ✅ Deploy and let users help improve your model
2. ✅ Monitor feedback statistics with `manage_feedback.py`
3. ✅ Periodically retrain with collected feedback
4. ✅ Watch accuracy improve over time!

---

**The feedback system is now live! 🎉 Users can help train your model starting today.**
