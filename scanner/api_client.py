import io
import requests
from django.conf import settings


class APIClientError(Exception):
    pass


def classify_image_with_external_api(image_pil):
    """
    Send image to configured external precise vegetable classification API.

    Expected API behavior (flexible):
    - Endpoint URL: settings.PRECISE_VEG_API_URL
    - Optional header: Authorization: Bearer <PRECISE_VEG_API_KEY> or x-api-key
    - Multipart form field: 'image' (file)

    Returns dict:
    {
      'label': 'tomato',
      'confidence': 0.92,    # 0..1
      'is_vegetable': True   # boolean
    }

    If API not configured, returns None so caller can fallback.
    """
    # Primary: call configured precise API endpoint if present
    url = getattr(settings, 'PRECISE_VEG_API_URL', '')
    key = getattr(settings, 'PRECISE_VEG_API_KEY', '')

    def _normalize_result(label, confidence, is_veg=None):
        # Normalize confidence to 0..1
        try:
            confidence = float(confidence) if confidence is not None else 0.0
            if confidence > 1.0:
                confidence = confidence / 100.0
        except Exception:
            confidence = 0.0

        return {
            'label': label,
            'confidence': confidence,
            'is_vegetable': bool(is_veg) if is_veg is not None else None
        }

    if url:
        try:
            buf = io.BytesIO()
            image_pil.save(buf, format='JPEG')
            buf.seek(0)
            files = {'image': ('image.jpg', buf, 'image/jpeg')}
            headers = {}
            if key:
                headers['Authorization'] = f'Bearer {key}'
                headers['x-api-key'] = key

            resp = requests.post(url, files=files, headers=headers, timeout=10)
            if resp.status_code != 200:
                raise APIClientError(f'API error: status {resp.status_code}')

            data = resp.json()
            if isinstance(data, dict):
                label = data.get('label') or data.get('predicted_label') or data.get('prediction')
                confidence = data.get('confidence')
                is_veg = data.get('is_vegetable')
                return _normalize_result(label, confidence, is_veg)

            raise APIClientError('Unexpected API response format')
        except requests.exceptions.RequestException as e:
            raise APIClientError(f'Network error: {str(e)}')
        except ValueError:
            raise APIClientError('Invalid JSON from API')

    # Fallback: try Hugging Face Inference API if configured via settings
    hf_model = getattr(settings, 'HF_MODEL', '') or getattr(settings, 'PRECISE_VEG_HF_MODEL', '')
    hf_token = getattr(settings, 'HF_TOKEN', '') or getattr(settings, 'PRECISE_VEG_HF_TOKEN', '')
    if not hf_model:
        return None

    # Simple mapping from common labels to your 13 classes
    label_map = {
        'bean': ['bean', 'green bean', 'string bean'],
        'broccoli': ['broccoli'],
        'bottle_gourd': ['bottle gourd', 'calabash', 'bottle-gourd', 'gourd'],
        'brinjal': ['eggplant', 'aubergine', 'brinjal'],
        'bitter_gourd': ['bitter gourd', 'bitter melon', 'bitter-gourd'],
        'cabbage': ['cabbage'],
        'capsicum': ['capsicum', 'bell pepper', 'pepper'],
        'carrot': ['carrot'],
        'cauliflower': ['cauliflower'],
        'potato': ['potato', 'irish potato'],
        'pumpkin': ['pumpkin', 'squash'],
        'radish': ['radish'],
        'tomato': ['tomato']
    }

    # Build reverse lookup
    reverse_map = {}
    for k, vals in label_map.items():
        for v in vals:
            reverse_map[v.lower()] = k

    # Call Hugging Face inference
    hf_url = f'https://api-inference.huggingface.co/models/{hf_model}'
    headers = {'Accept': 'application/json'}
    if hf_token:
        headers['Authorization'] = f'Bearer {hf_token}'

    try:
        buf = io.BytesIO()
        image_pil.save(buf, format='JPEG')
        data = buf.getvalue()
        resp = requests.post(hf_url, headers=headers, data=data, timeout=20)
        if resp.status_code == 503:
            # model is loading
            raise APIClientError('HuggingFace model is loading')
        if resp.status_code == 401 or resp.status_code == 403:
            raise APIClientError('Unauthorized to access HuggingFace model')
        if resp.status_code != 200:
            raise APIClientError(f'HF error: status {resp.status_code}')

        preds = resp.json()
        # HF image-classification returns a list of {label,score}
        if isinstance(preds, list) and len(preds) > 0:
            top = preds[0]
            label_text = top.get('label','').lower()
            score = float(top.get('score', 0.0))

            # sometimes labels come like 'LABEL_0' or 'tomato (food)'; try to extract words
            cleaned = label_text
            # remove parentheses content
            if '(' in cleaned:
                cleaned = cleaned.split('(')[0].strip()

            mapped = reverse_map.get(cleaned)
            if mapped:
                return _normalize_result(mapped, score, True)

            # If no direct mapping, try substring matching
            for key_word, mapped_class in reverse_map.items():
                if key_word in cleaned:
                    return _normalize_result(mapped_class, score, True)

            # Not one of the 13 classes
            return _normalize_result('not_vegetable', score, False)

        raise APIClientError('Unexpected HF response format')
    except requests.exceptions.RequestException as e:
        raise APIClientError(f'HF network error: {str(e)}')
    except ValueError:
        raise APIClientError('Invalid JSON from HF')
