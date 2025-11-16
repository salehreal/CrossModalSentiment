import emoji

def extract_emojis(text: str):
    return [c for c in text if c in emoji.EMOJI_DATA]

def emoji_sentiment_score(emojis: list):
    positive = {"😊", "😍", "😂", "😄", "😁", "🥰", "😎", "🤩"}
    negative = {"😢", "😠", "😞", "😭", "😡", "💔", "😣", "😤"}

    score = 0
    for e in emojis:
        if e in positive:
            score += 1
        elif e in negative:
            score -= 1

    return score / max(len(emojis), 1)

def count_emojis(emojis: list):
    return len(emojis)

def process_text(text: str):
    emojis = extract_emojis(text)
    return {
        "emojis": emojis,
        "count": count_emojis(emojis),
        "sentiment_score": emoji_sentiment_score(emojis)
    }
