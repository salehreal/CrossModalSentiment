import emoji

def extract_emojis(text):
    return [c for c in text if c in emoji.EMOJI_DATA]

def emoji_sentiment_score(emojis):
    positive = {"😊", "😍", "😂", "😄", "😁"}
    negative = {"😢", "😠", "😞", "😭", "😡"}
    score = 0
    for e in emojis:
        if e in positive:
            score += 1
        elif e in negative:
            score -= 1
    return score / max(len(emojis), 1)
