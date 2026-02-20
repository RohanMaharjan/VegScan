#!/usr/bin/env python3
"""
Feedback Data Management Script

This script helps you:
1. View collected feedback statistics
2. Copy feedback images to training folder
3. Clean up old feedback
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent
FEEDBACK_DIR = PROJECT_ROOT / 'feedback'
FEEDBACK_JSON = FEEDBACK_DIR / 'feedback_data.json'
FEEDBACK_IMAGES = FEEDBACK_DIR / 'images'
TRAIN_DIR = PROJECT_ROOT / 'data_split' / 'train'


def load_feedback():
    """Load feedback data from JSON"""
    if not FEEDBACK_JSON.exists():
        return []
    
    with open(FEEDBACK_JSON, 'r') as f:
        return json.load(f)


def get_statistics():
    """Display feedback statistics"""
    data = load_feedback()
    
    if not data:
        print("❌ No feedback data found yet!")
        return
    
    # Count by vegetable
    by_veggie = defaultdict(int)
    for entry in data:
        veggie = entry.get('correct_label', 'unknown')
        by_veggie[veggie] += 1
    
    # Count misclassifications
    misclassified = sum(1 for entry in data if entry.get('predicted_label') != entry.get('correct_label'))
    
    print("\n" + "="*60)
    print("📊 FEEDBACK STATISTICS")
    print("="*60)
    print(f"Total Feedback Entries: {len(data)}")
    print(f"Misclassified Images: {misclassified}")
    print(f"Correct Predictions: {len(data) - misclassified}")
    print(f"\nBreakdown by Vegetable:")
    print("-" * 40)
    
    for veggie in sorted(by_veggie.keys()):
        count = by_veggie[veggie]
        percentage = (count / len(data)) * 100
        print(f"  {veggie:20} → {count:3} images ({percentage:5.1f}%)")
    
    print("\n" + "="*60 + "\n")


def copy_to_training():
    """Copy feedback images to training folder"""
    data = load_feedback()
    
    if not data:
        print("❌ No feedback to copy!")
        return
    
    copied_count = 0
    
    for entry in data:
        src = FEEDBACK_IMAGES / entry.get('image_filename', '')
        correct_label = entry.get('correct_label', 'unknown')
        
        if not src.exists():
            print(f"⚠️  Missing: {src}")
            continue
        
        # Create destination in training folder
        dst_dir = TRAIN_DIR / correct_label
        os.makedirs(dst_dir, exist_ok=True)
        
        # Copy with timestamp to avoid conflicts
        timestamp = entry.get('timestamp', '').replace(':', '-')[:19]
        dst_filename = f"{correct_label}_{timestamp}_{src.stem}.jpg"
        dst = dst_dir / dst_filename
        
        try:
            shutil.copy2(src, dst)
            copied_count += 1
            print(f"✓ {src.name} → {correct_label}/")
        except Exception as e:
            print(f"✗ Error copying {src.name}: {e}")
    
    print(f"\n✅ Copied {copied_count} images to training dataset!")
    print(f"Next: Run 'python train_model_vgg16.py' to retrain")


def clear_feedback():
    """Clear all feedback (with confirmation)"""
    response = input("\n⚠️  This will DELETE all feedback images and data!\nType 'YES' to confirm: ")
    
    if response != "YES":
        print("Cancelled.")
        return
    
    try:
        # Archive into timestamped folder before deletion
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_dir = PROJECT_ROOT / f'feedback_archive_{timestamp}'
        shutil.copytree(FEEDBACK_DIR, archive_dir)
        print(f"✓ Archived to: {archive_dir}")
        
        # Clear feedback
        shutil.rmtree(FEEDBACK_DIR)
        os.makedirs(FEEDBACK_IMAGES, exist_ok=True)
        
        # Reset JSON
        with open(FEEDBACK_JSON, 'w') as f:
            json.dump([], f)
        
        print("✅ Feedback cleared!")
    except Exception as e:
        print(f"✗ Error: {e}")


def list_recent():
    """Show recent feedback entries"""
    data = load_feedback()
    
    if not data:
        print("No feedback yet!")
        return
    
    print("\n📝 Recent Feedback (last 10):")
    print("-" * 80)
    
    for entry in reversed(data[-10:]):
        timestamp = entry.get('timestamp', 'unknown')[:19]
        predicted = entry.get('predicted_label', '?')
        correct = entry.get('correct_label', '?')
        status = "✓" if predicted == correct else "✗"
        
        print(f"{status} {timestamp} | Predicted: {predicted:15} → Actual: {correct:15}")
    
    print()


def main():
    print("""
\033[1m╔════════════════════════════════════════════════╗
║   Feedback Data Management Tool                  ║
╚════════════════════════════════════════════════╝\033[0m
""")
    
    while True:
        print("""
Options:
  1. View Statistics
  2. Show Recent Feedback
  3. Copy to Training Folder
  4. Clear All Feedback
  5. Exit
""")
        choice = input("Select option (1-5): ").strip()
        
        if choice == '1':
            get_statistics()
        elif choice == '2':
            list_recent()
        elif choice == '3':
            copy_to_training()
        elif choice == '4':
            clear_feedback()
        elif choice == '5':
            print("\nGoodbye! 👋")
            break
        else:
            print("Invalid option!")


if __name__ == '__main__':
    main()
