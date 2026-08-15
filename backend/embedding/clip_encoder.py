"""Hàm hỗ trợ cũ để tương thích ngược."""

def translate_vi_to_en(text: str) -> str:
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source='vi', target='en').translate(text)
    except Exception:
        return text
