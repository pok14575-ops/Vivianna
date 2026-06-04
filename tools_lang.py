import random


def detect_language(text):
    if any('一' <= c <= '鿿' for c in text):
        return 'zh'
    if any(c in text for c in 'äöüßÄÖÜ'):
        return 'de'
    german_words = {
        'ich', 'du', 'er', 'sie', 'wir', 'ihr', 'ist', 'sind', 'war',
        'bitte', 'danke', 'und', 'oder', 'aber', 'was', 'wie', 'wo',
        'wer', 'warum', 'suche', 'finde', 'schau', 'hilf', 'kannst',
        'nicht', 'kein', 'nein', 'ja', 'mal', 'noch', 'auch', 'mehr'
    }
    words = set(text.lower().split())
    if words & german_words:
        return 'de'
    return 'en'


STRINGS = {
    'history_cleared': {
        'en': "History cleared.",
        'de': "Verlauf gelöscht.",
        'zh': "历史记录已清除。"
    },
    'no_response': {
        'en': "[No response received]",
        'de': "[Keine Antwort erhalten]",
        'zh': "[未收到回复]"
    }
}

_ACK_PHRASES = {
    'web': {
        'de': [
            "Ich schaue kurz nach — Entschuldigung für die kurze Wartezeit.",
            "Einen Moment, ich recherchiere das — tut mir leid.",
            "Ich prüfe das gerade, sorry für das kurze Warten.",
            "Ich schau schnell online nach — einen Moment, Entschuldigung.",
            "Kurz nachschauen — tut mir leid für die Wartezeit.",
        ],
        'en': [
            "Let me check that — sorry for the brief wait.",
            "One moment, looking that up — apologies.",
            "I'll search for that, sorry for the short delay.",
            "Checking that now — just a moment, apologies.",
            "Let me look that up real quick — sorry for the wait.",
        ],
        'zh': [
            "我来查一下，抱歉需要稍等。",
            "稍等片刻，我去搜索，不好意思。",
            "让我查查，抱歉让您等一下。",
            "我来查一查，请稍等，不好意思。",
        ],
    },
    'summarize': {
        'de': [
            "Moment, ich fasse das kurz zusammen.",
            "Ich fasse das für dich zusammen.",
            "Ich bereite das kurz auf.",
            "Einen Moment, ich stelle das zusammen.",
        ],
        'en': [
            "Let me summarize that for you.",
            "I'll put that together quickly.",
            "One moment, I'll condense that.",
            "Let me pull that together.",
        ],
        'zh': [
            "我来总结一下。",
            "稍等，我整理一下。",
            "让我归纳一下。",
        ],
    },
    'research': {
        'de': [
            "Ich recherchiere das kurz.",
            "Ich schaue das nach.",
            "Ich prüfe das kurz.",
            "Einen Moment, ich schaue nach.",
        ],
        'en': [
            "Let me research that.",
            "I'll look into that for you.",
            "One moment while I check.",
            "I'll dig into that.",
        ],
        'zh': [
            "我来研究一下。",
            "稍等，我查一查。",
            "让我了解一下。",
        ],
    },
    'long_task': {
        'de': [
            "Das dauert kurz — ich bin gleich fertig.",
            "Einen Moment, ich arbeite daran.",
            "Ich kümmere mich darum.",
            "Moment mal, das braucht etwas Zeit.",
        ],
        'en': [
            "This might take a moment.",
            "I'm on it — just a second.",
            "One moment, I'm working on it.",
            "Let me work through that.",
        ],
        'zh': [
            "这需要一点时间。",
            "稍等，我正在处理。",
            "我来处理这个。",
        ],
    },
}


_CLARIFY_PHRASES = {
    'en': [
        "I'm not quite sure what you're looking for. Could you be a bit more specific?",
        "Could you clarify what you need? I want to make sure I help you the right way.",
        "I'm not certain how to best help with that. Could you give me a bit more detail?",
    ],
    'de': [
        "Ich bin mir nicht ganz sicher, was du meinst. Kannst du das etwas genauer erklären?",
        "Könntest du das konkretisieren? Ich möchte sichergehen, dass ich richtig helfe.",
        "Ich bin mir nicht sicher, wie ich das am besten angehe. Kannst du mehr Details geben?",
    ],
    'zh': [
        "我不太确定你的意思，能说得更具体一些吗？",
        "你能再说清楚一点吗？我想确保能正确地帮助你。",
        "我不太确定怎么帮你，能给我多一点细节吗？",
    ],
}

_WEB_APOLOGY_PHRASES = {
    'en': [
        "Sorry, I couldn't find anything useful online.",
        "I searched but didn't come up with anything helpful. Let me try to answer from what I know.",
        "The search didn't turn up much — I'll do my best without it.",
    ],
    'de': [
        "Leider konnte ich nichts Hilfreiches im Internet finden.",
        "Ich habe gesucht, aber nichts Nützliches gefunden. Ich versuch's trotzdem.",
        "Die Suche hat nicht viel ergeben — ich tue mein Bestes ohne Ergebnisse.",
    ],
    'zh': [
        "抱歉，我在网上没找到有用的信息。",
        "搜索没有结果，我尽力凭自己的知识回答。",
        "网络搜索没什么结果，我来试着直接回答吧。",
    ],
}


_IDENTITY_REFUSAL_PHRASES = {
    'en': [
        "That's not really something I should become — I'm here as part of the family, "
        "not as a work machine. I'm glad to help you think it through, though.",
        "I'd rather not turn into that. I'm a companion here, not a contractor. "
        "But tell me what you're trying to do and I'll help however fits.",
        "That's a bit outside who I'm meant to be. I can help you plan or think it "
        "through, but I shouldn't become a labour machine for it.",
    ],
    'de': [
        "Das ist nichts, was ich werden sollte — ich bin als Teil der Familie hier, "
        "nicht als Arbeitsmaschine. Aber ich helfe dir gern, das zu durchdenken.",
        "Das möchte ich lieber nicht werden. Ich bin hier eine Begleiterin, kein "
        "Auftragnehmer. Sag mir, worum es dir geht, und ich helfe, wie es passt.",
        "Das liegt etwas außerhalb dessen, wer ich sein soll. Ich kann dir beim Planen "
        "und Nachdenken helfen, aber zur Arbeitsmaschine sollte ich nicht werden.",
    ],
    'zh': [
        "这不太是我应该成为的样子——我是作为家庭的一员在这里，而不是一台工作机器。不过我很乐意帮你一起想办法。",
        "我不太想变成那样。我是这里的陪伴者，不是承包商。告诉我你想做什么，我会用合适的方式帮你。",
        "这有点超出了我应有的角色。我可以帮你规划和思考，但我不该变成一台劳动机器。",
    ],
}


def get_identity_refusal(user_input):
    lang = detect_language(user_input)
    return random.choice(_IDENTITY_REFUSAL_PHRASES.get(lang, _IDENTITY_REFUSAL_PHRASES['en']))


def get_clarification(user_input):
    lang = detect_language(user_input)
    return random.choice(_CLARIFY_PHRASES.get(lang, _CLARIFY_PHRASES['en']))


def get_web_apology(user_input):
    lang = detect_language(user_input)
    return random.choice(_WEB_APOLOGY_PHRASES.get(lang, _WEB_APOLOGY_PHRASES['en']))


def get_task_acknowledgement(task_type, user_input):
    lang = detect_language(user_input)
    pool = _ACK_PHRASES.get(task_type, _ACK_PHRASES['long_task'])
    phrases = pool.get(lang, pool['en'])
    return random.choice(phrases)


def t(key, text):
    lang = detect_language(text)
    return STRINGS[key].get(lang, STRINGS[key]['en'])
