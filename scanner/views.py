from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse, FileResponse, Http404
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import io
import os
from django.conf import settings
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from django.conf import settings
import numpy as np
import random
import mimetypes
import json
from datetime import datetime
from .api_client import classify_image_with_external_api, APIClientError

# Simple in-process model cache to avoid reloading weights per request
MODEL_CACHE = {}

# small tensor transform used for crops (we handle resizing/crops manually)
tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =========================
# CLASS LABELS (VERY IMPORTANT)
# Must match your training order!
# =========================
class_names = ['bean', 'broccoli', 'bottle_gourd', 'brinjal', 'bitter_gourd', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'potato', 'pumpkin', 'radish', 'tomato']


# =========================
# IMAGE TRANSFORM - Standard (no augmentation)
# Must match training transform
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# =========================
# IMAGE TRANSFORMS FOR TEST-TIME AUGMENTATION
# Multiple variations to increase prediction confidence
# =========================
tta_transforms = [
    # Original
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    # Horizontal flip
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    # Slight rotation
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    # Color jitter
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
]


# =========================
# LOAD MODEL FUNCTION
# =========================
def load_model(model_name):
    BASE_DIR = settings.BASE_DIR
    num_classes = len(class_names)
    # Use a simple cache to keep models in memory between requests
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    if model_name == "model1":
        model_path = os.path.join(BASE_DIR, "models", "vgg16_veggies.pth")
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif model_name == "model2":
        model_path = os.path.join(BASE_DIR, "models", "mobilenetv2_veggies.pth")
        model = models.mobilenet_v2(weights=None)
        # Fix: Use proper in_features instead of last_channel
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError("Invalid model selected")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    MODEL_CACHE[model_name] = model
    return model


# =========================
# TEST-TIME AUGMENTATION PREDICTION
# Averages predictions across multiple augmented versions
# =========================
def _preprocess_image(image_pil):
    """Apply lightweight enhancements to make the image clearer for inference."""
    try:
        im = image_pil.convert('RGB')
        im = ImageOps.autocontrast(im, cutoff=1)
        im = im.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        return im
    except Exception:
        return image_pil


def predict_with_tta(image_pil, model, return_probs=False):
    """
    Extended TTA: preprocess, create variations + five crops per variation, and average probabilities.
    If `return_probs` is True, returns the averaged probability vector instead of index/confidence.
    """
    im = _preprocess_image(image_pil)

    variations = []
    # original
    variations.append(im)
    # hflip
    variations.append(ImageOps.mirror(im))
    # slight rotate
    variations.append(im.rotate(8, resample=Image.BILINEAR))
    # slight autocontrast variant
    variations.append(ImageOps.autocontrast(im, cutoff=2))

    all_probs = []

    # For multi-crop: resize to 256 then take 5 crops of 224
    for var in variations:
        try:
            var_resized = var.resize((256, 256))
        except Exception:
            var_resized = var

        crop_size = 224
        margin = 256 - crop_size
        coords = [
            (0, 0),
            (margin, 0),
            (0, margin),
            (margin, margin),
            (margin//2, margin//2)
        ]

        crops = [var_resized.crop((l, t, l + crop_size, t + crop_size)) for (l, t) in coords]

        batch_tensors = torch.stack([tensor_norm(crop) for crop in crops])
        with torch.no_grad():
            outputs = model(batch_tensors)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            # average over crops for this variation
            avg_var = np.mean(probs, axis=0)
            all_probs.append(avg_var)

    if len(all_probs) == 0:
        # fallback: single transform
        t = transform(image_pil).unsqueeze(0)
        with torch.no_grad():
            out = model(t)
            p = torch.softmax(out, dim=1).cpu().numpy()[0]
        if return_probs:
            return p
        idx = int(np.argmax(p))
        return idx, float(p[idx])

    # average across variations
    avg_probs = np.mean(all_probs, axis=0)

    if return_probs:
        return avg_probs

    predicted_idx = int(np.argmax(avg_probs))
    confidence = float(avg_probs[predicted_idx])
    return predicted_idx, confidence


# =========================
# ENSEMBLE PREDICTION
# Combines predictions from both VGG16 and MobileNetV2
# =========================
def predict_with_ensemble(image_pil):
    """
    Use ensemble of both models for better accuracy.
    Averages predictions from VGG16 and MobileNetV2.
    """
    vgg16_model = load_model("model1")
    mobilenetv2_model = load_model("model2")

    # Get full averaged probability distributions from both models using extended TTA
    vgg_probs = predict_with_tta(image_pil, vgg16_model, return_probs=True)
    mobile_probs = predict_with_tta(image_pil, mobilenetv2_model, return_probs=True)

    # Weight each model by its peak confidence to prefer the more-certain model per-image
    vgg_conf = float(np.max(vgg_probs))
    mobile_conf = float(np.max(mobile_probs))
    total = vgg_conf + mobile_conf + 1e-12
    w_vgg = vgg_conf / total
    w_mobile = mobile_conf / total

    ensemble_probs = (w_vgg * vgg_probs) + (w_mobile * mobile_probs)

    predicted_idx = int(np.argmax(ensemble_probs))
    confidence = float(ensemble_probs[predicted_idx])
    return predicted_idx, confidence



# =========================
# HOME PAGE
# =========================
def home_view(request):
    return render(request, 'home.html')


def slideshow_api(request):
    """Return a small set of random image URLs from data_split/test.
    The frontend will request these URLs; images are served by `slideshow_image` view.
    """
    try:
        # Prefer curated static slideshow images if present
        static_dir = os.path.join(settings.BASE_DIR, 'static', 'slideshow')
        payload = []
        if os.path.isdir(static_dir):
            files = [f for f in os.listdir(static_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            files.sort()
            random.shuffle(files)
            for fname in files[:12]:
                url = settings.STATIC_URL.rstrip('/') + '/slideshow/' + fname
                payload.append({'url': url, 'name': fname})
            if payload:
                return JsonResponse({'images': payload})

        # Fallback: serve directly from data_split/test via slideshow_image view
        base_dir = os.path.join(settings.BASE_DIR, 'data_split', 'test')
        images = []
        if os.path.isdir(base_dir):
            for cls in os.listdir(base_dir):
                cls_dir = os.path.join(base_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        images.append({'class': cls, 'name': fname})

        random.shuffle(images)
        selected = images[:8]
        for it in selected:
            url = f'/slideshow/image/{it["class"]}/{it["name"]}/'
            payload.append({'url': url, 'name': it['name']})

        return JsonResponse({'images': payload})
    except Exception as e:
        return JsonResponse({'images': [], 'error': str(e)})


def slideshow_image(request, cls, fname):
    """Serve a file from data_split/test/<cls>/<fname> with safe checks.
    """
    # Prevent path traversal and ensure class is recognized
    if cls not in class_names:
        raise Http404()

    safe_fname = os.path.basename(fname)
    file_path = os.path.join(settings.BASE_DIR, 'data_split', 'test', cls, safe_fname)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise Http404()

    content_type, _ = mimetypes.guess_type(file_path)
    try:
        return FileResponse(open(file_path, 'rb'), content_type=content_type or 'application/octet-stream')
    except Exception:
        raise Http404()


# =========================
# LOGIN
# =========================
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password')

    return render(request, 'login.html')


# =========================
# REGISTER
# =========================
def register_view(request):
    if request.method == 'POST':
        full_name = request.POST.get('full_name', '').strip()
        email = request.POST.get('email', '').strip()
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        confirm_password = request.POST.get('confirm_password', '')

        # Validation
        errors = []
        
        if not full_name:
            errors.append('Full name is required')
        if not email:
            errors.append('Email is required')
        elif '@' not in email or '.' not in email.split('@')[-1]:
            errors.append('Please enter a valid email address')
        if not username:
            errors.append('Username is required')
        elif len(username) < 3:
            errors.append('Username must be at least 3 characters')
        if len(password) < 8:
            errors.append('Password must be at least 8 characters')
        if password != confirm_password:
            errors.append('Passwords do not match')
        if User.objects.filter(username=username).exists():
            errors.append('Username already exists')
        if User.objects.filter(email=email).exists():
            errors.append('Email already registered')

        if errors:
            for error in errors:
                messages.error(request, error)
            return render(request, 'register.html', {
                'full_name': full_name,
                'email': email,
                'username': username,
            })
        
        # Create user
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=full_name
        )
        messages.success(request, 'Account created successfully! Please log in.')
        return redirect('login')

    return render(request, 'register.html')


# =========================
# LOGOUT
# =========================
def logout_view(request):
    logout(request)
    return redirect('home')


# =========================
# SCANNER PAGE
# =========================
def scanner_view(request):
    return render(request, 'scanner.html')


# =========================
# BATCH SCAN API
# =========================
def scan_batch_api(request):
    if request.method == 'POST':
        images = request.FILES.getlist('images')
        selected_model = request.POST.get('model')
        use_ensemble = request.POST.get('ensemble', 'false').lower() == 'true'

        results = []

        try:
            for img in images:
                image = Image.open(img).convert("RGB")
                # If an external precise API URL is configured, call it (hidden from UI).
                # If not configured, fall back to local models instead of failing.
                api_used = False
                api_result = None
                api_url = getattr(settings, 'PRECISE_VEG_API_URL', '')
                api_configured = bool(str(api_url).strip())

                if api_configured:
                    try:
                        api_result = classify_image_with_external_api(image)
                    except APIClientError:
                        api_result = None

                if api_result is not None:
                    api_used = True
                    # Normalize returned values
                    label = api_result.get('label') or 'Unknown'
                    confidence = api_result.get('confidence')
                    is_veg = api_result.get('is_vegetable')

                    # If API did not provide is_vegetable, infer from label membership
                    if is_veg is None:
                        is_veg = (label in class_names)

                    # Ensure confidence is a float 0..1, fallback to 0 if missing
                    try:
                        confidence = float(confidence) if confidence is not None else 0.0
                        if confidence > 1.0:
                            confidence = confidence / 100.0
                    except Exception:
                        confidence = 0.0

                    # Build result according to existing frontend expectations
                    if is_veg and confidence >= 0.60 and label in class_names:
                        results.append({
                            'label': label,
                            'confidence': round(confidence, 4),
                            'confidence_percent': round(confidence * 100, 2),
                            'is_valid': True,
                            'source': 'external_api'
                        })
                    else:
                        results.append({
                            'label': 'not_vegetable',
                            'confidence': round(confidence, 4),
                            'confidence_percent': round(confidence * 100, 2),
                            'is_valid': False,
                            'warning': 'Image not recognized or confidence too low',
                            'source': 'external_api'
                        })
                    # continue to next image without using local models
                    continue
                # If API was configured but returned no result, return error
                if api_configured and api_result is None:
                    return JsonResponse({'error': 'External precise classification API configured but unavailable.'}, status=500)

                if use_ensemble:
                    # Use ensemble of both models with TTA
                    predicted_idx, confidence = predict_with_ensemble(image)
                elif selected_model == "ensemble":
                    # Ensemble mode selected
                    predicted_idx, confidence = predict_with_ensemble(image)
                else:
                    # Single model with TTA
                    try:
                        model = load_model(selected_model)
                    except Exception as e:
                        return JsonResponse({'error': str(e)}, status=500)
                    
                    predicted_idx, confidence = predict_with_tta(image, model)

                label = class_names[int(predicted_idx)]
                
                # Check if confidence is high enough (>60%)
                # If below 60%, it's likely an invalid/non-vegetable image or wrong vegetable
                if confidence >= 0.60:
                    results.append({
                        'label': label,
                        'confidence': round(confidence, 4),
                        'confidence_percent': round(confidence * 100, 2),
                        'is_valid': True
                    })
                else:
                    results.append({
                        'label': 'not_vegetable',
                        'confidence': round(confidence, 4),
                        'confidence_percent': round(confidence * 100, 2),
                        'is_valid': False,
                        'warning': 'Image not recognized or confidence too low'
                    })

        except Exception as e:
            return JsonResponse({'error': f'Processing error: {str(e)}'}, status=500)

        return JsonResponse({'items': results})

    return JsonResponse({'error': 'Invalid request method'}, status=400)


# ============================
# FEEDBACK SYSTEM - Save user corrections for future training
# ============================
def save_feedback_api(request):
    """
    Accept user feedback on predictions:
    - predicted_label: what the model predicted
    - correct_label: what the user says it should be (or 'not_vegetable')
    - image: the image file blob
    
    Saves image + metadata to feedback/ folder for future model training
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=400)
    
    try:
        predicted_label = request.POST.get('predicted_label', '')
        correct_label = request.POST.get('correct_label', '')
        image_file = request.FILES.get('image')
        
        if not image_file or not correct_label:
            return JsonResponse({'error': 'Missing image or correct_label'}, status=400)
        
        # Validate labels
        correct_label = correct_label.lower().strip()
        valid_labels = class_names + ['not_vegetable']  # Allow 'not_vegetable' option
        
        if correct_label not in valid_labels:
            valid_options = ', '.join(class_names)
            return JsonResponse({
                'error': f'Invalid option. Valid: {valid_options}, or "not_vegetable"'
            }, status=400)
        
        # Create feedback directory structure
        feedback_dir = os.path.join(settings.BASE_DIR, 'feedback')
        images_dir = os.path.join(feedback_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # milliseconds
        filename = f"{correct_label}_{timestamp}.jpg"
        filepath = os.path.join(images_dir, filename)
        
        # Save image
        image = Image.open(image_file).convert('RGB')
        image.save(filepath, 'JPEG', quality=95)
        
        # Create/update feedback_data.json metadata
        metadata_file = os.path.join(feedback_dir, 'feedback_data.json')
        
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'image_filename': filename,
            'predicted_label': predicted_label,
            'correct_label': correct_label,
            'image_path': f'feedback/images/{filename}'
        }
        
        # Load existing feedback or create new list
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                feedback_data = json.load(f)
        else:
            feedback_data = []
        
        feedback_data.append(feedback_entry)
        
        # Save updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        return JsonResponse({
            'success': True,
            'message': f'Thank you! Image saved for future training.',
            'entry': feedback_entry
        })
    
    except Exception as e:
        return JsonResponse({'error': f'Error saving feedback: {str(e)}'}, status=500)

