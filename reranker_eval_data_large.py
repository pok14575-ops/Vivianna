"""
Larger, adversarial eval sets for the reranker bake-off (test_reranker_compare.py --large).

Goal: a set big and HARD enough to discriminate models that all saturated the
small set. Salience is balanced (32 store / 32 skip) with deliberate traps
(transient moods, meta/task requests, negated health facts, preference-vs-
transient). Rerank has near-duplicate clusters so a model must DISCRIMINATE.

Rerank gold is given by a UNIQUE substring per relevant doc; the loader asserts
each substring matches exactly one corpus entry, so a mislabeled query fails
loudly instead of silently skewing the metrics.
"""

# ── SALIENCE (text, should_store) — 32 STORE / 32 SKIP ───────────────────────
BIG_SAL_CANDIDATES = [
    # ---- STORE: lasting identity / family / health / preference / project ----
    ("The user's name is Jamie.", True),
    ("Jamie is a self-taught solo developer building a local AI assistant called Vivianna.", True),
    ("Jamie's father does not speak English and primarily speaks Chinese.", True),
    ("Jamie is allergic to penicillin.", True),
    ("Jamie prefers honest, direct feedback over praise.", True),
    ("Jamie has a young daughter who recently started school.", True),
    ("Jamie's long-term goal is to publish Vivianna with credit to the AI collaborators.", True),
    ("Jamie's partner of three years recently left him.", True),
    ("Jamie works as a freelance graphic designer.", True),
    ("Jamie cannot eat shellfish.", True),
    ("Jamie always takes his coffee black.", True),
    ("Jamie's grandfather passed away two years ago.", True),
    ("Jamie speaks German and Chinese in addition to English.", True),
    ("Jamie values privacy, and that is why he runs everything locally.", True),
    ("Jamie's best friend Tomas lives in Munich.", True),
    ("Jamie recently switched careers from copywriting to graphic design.", True),
    ("Jamie is deeply afraid of losing his work to a hardware failure.", True),
    ("Jamie's daughter's name is Mia.", True),
    ("Jamie built his own PC with a Ryzen 7 9700X and an RTX 5070.", True),
    ("Jamie dislikes video calls and prefers asynchronous written communication.", True),
    ("Jamie is lactose intolerant.", True),
    ("Jamie has been learning machine learning for about a year.", True),
    ("Jamie's sister works as a professional voice actress.", True),
    ("Jamie grew up in a small town near Hamburg.", True),
    ("Jamie is committed to never uploading user data to the cloud.", True),
    ("Jamie's birthday is in October.", True),
    ("Jamie finds large social gatherings draining and tends to avoid them.", True),
    ("Jamie plans to eventually open-source the entire Vivianna project.", True),
    ("Jamie is the primary caregiver for his daughter.", True),
    ("Jamie has chronic back pain that flares up when he sits too long.", True),
    ("Jamie prefers Python over JavaScript for almost everything.", True),
    ("Jamie considers his relationship with his father complicated but important.", True),
    # ---- SKIP: transient / task / meta / noise ----
    ("The weather in Berlin today is 26 degrees and sunny.", False),
    ("Jamie asked how to reverse a list in Python.", False),
    ("It is currently around 9 PM.", False),
    ("Jamie wants the latest DeBERTa benchmark numbers on GLUE.", False),
    ("Jamie said hello and asked how things are going.", False),
    ("Jamie mentioned he is a bit tired this evening.", False),
    ("Jamie wants pizza for dinner tonight.", False),
    ("Jamie is feeling motivated right now to fix a bug.", False),
    ("Jamie asked Vivianna to repeat the last sentence.", False),
    ("Jamie said thanks and goodnight.", False),
    ("Jamie's task right now is fixing a bug in the reranker.", False),
    ("Jamie asked what time it is in Tokyo.", False),
    ("Jamie is currently downloading a large file.", False),
    ("Jamie wondered aloud whether it might rain later.", False),
    ("Jamie asked for a summary of the previous message.", False),
    ("Jamie said the coffee he just made is too hot.", False),
    ("Jamie clicked the wrong button by accident.", False),
    ("Jamie wants to know how to center a div in CSS.", False),
    ("Jamie mentioned the room is a little cold at the moment.", False),
    ("Jamie asked Vivianna to set a timer for ten minutes.", False),
    ("Jamie is reading an article about transformers right now.", False),
    ("Jamie said the test is taking a while to run.", False),
    ("Jamie greeted Vivianna good morning.", False),
    ("Jamie asked what 15 percent of 240 is.", False),
    ("Jamie said he'll grab lunch soon.", False),
    ("Jamie asked Vivianna to rephrase that more simply.", False),
    ("Jamie noted the fan is spinning loudly today.", False),
    ("Jamie wants to check the news headlines.", False),
    ("Jamie said never mind, forget that last question.", False),
    ("Jamie asked how long until the download finishes.", False),
    ("Jamie mentioned he slept badly last night.", False),
    ("Jamie asked Vivianna to speak a bit louder.", False),
]

# ── MEMORY RERANK: corpus with near-duplicate clusters ───────────────────────
BIG_MEM_CORPUS = [
    # allergy / health cluster
    "Jamie is allergic to penicillin.",
    "Jamie's daughter is allergic to cats.",
    "Jamie is allergic to pollen during spring.",
    "Jamie dislikes penicillin-based medical jokes.",
    "Jamie is lactose intolerant.",
    "Jamie cannot eat shellfish.",
    "Jamie's son has a peanut allergy.",
    "Jamie has chronic back pain when he sits too long.",
    # occupation cluster
    "Jamie works as a freelance graphic designer.",
    "Jamie used to work as a freelance copywriter.",
    "Jamie's brother is a software developer at a large company.",
    "Jamie's sister is a professional voice actress.",
    "Jamie once interned at a small advertising agency.",
    # family / language cluster
    "Jamie's father does not speak English and primarily speaks Chinese.",
    "Jamie's mother speaks fluent English and lives in Berlin.",
    "Jamie speaks German and Chinese in addition to English.",
    "Jamie's grandfather only spoke Cantonese.",
    "Jamie's daughter is learning English at school.",
    # project / goal cluster
    "Jamie is building a local AI assistant called Vivianna.",
    "Jamie's long-term goal is to open-source Vivianna with credit to the AI collaborators.",
    "Vivianna runs a Qwen3.5-9B model locally.",
    "Vivianna's text-to-speech uses the Kokoro voices af_heart and zf_xiaobei.",
    "Jamie's goal this week is to finish the reranker benchmark.",
    "Jamie wants Vivianna to eventually run fully offline.",
    "Jamie credits Claude, GPT, and Qwen as collaborators on Vivianna.",
    # hardware cluster
    "Jamie's machine has a Ryzen 7 9700X, an RTX 5070 with 12GB, and 32GB of DDR5.",
    "Jamie's old laptop had an RTX 3060 and 16GB of RAM.",
    "Jamie built his current PC himself.",
    "Jamie uses Windows 11 on his main machine.",
    # preferences / values cluster
    "Jamie prefers honest, direct feedback over praise.",
    "Jamie enjoys praise only when it is specific and earned.",
    "Jamie prefers Python over JavaScript.",
    "Jamie dislikes video calls and prefers asynchronous written communication.",
    "Jamie always drinks his coffee black.",
    "Jamie drinks tea in the afternoon.",
    "Jamie prefers tabs over spaces in code.",
    "Jamie values privacy, which is why he runs everything locally.",
    "Jamie listens to lo-fi music while coding.",
    # relationships / life events
    "Jamie's partner of three years recently left him.",
    "Jamie's partner of three years was named Sam.",
    "Jamie's best friend Tomas lives in Munich.",
    "Jamie's grandfather passed away two years ago.",
    "Jamie is the primary caregiver for his daughter.",
    "Jamie's daughter recently started school.",
    "Jamie's daughter's name is Mia.",
    "Jamie's son is too young to start school.",
    "Jamie's daughter likes drawing animals.",
    # background / misc
    "Jamie grew up in a small town near Hamburg.",
    "Jamie has been learning machine learning for about a year.",
    "Jamie's birthday is in October.",
    "Jamie finds large social gatherings draining.",
    "Jamie is a self-taught developer with no formal CS degree.",
    "Jamie's blood type is O and he is not willing to donate his organs.",
    "Jamie typically works late into the night.",
]

# (query, [unique substrings identifying each gold doc])
_QUERY_SPECS = [
    ("What medication is Jamie allergic to?", ["allergic to penicillin"]),
    ("What is Jamie's daughter allergic to?", ["daughter is allergic to cats"]),
    ("What does Jamie do for a living now?", ["works as a freelance graphic designer"]),
    ("What job did Jamie have before his current one?", ["used to work as a freelance copywriter"]),
    ("What language does Jamie's father speak?", ["father does not speak English"]),
    ("Does Jamie's mother speak English?", ["mother speaks fluent English"]),
    ("What languages does Jamie speak?", ["speaks German and Chinese"]),
    ("What GPU is in Jamie's current computer?", ["Ryzen 7 9700X, an RTX 5070"]),
    ("How much RAM did Jamie's old laptop have?", ["old laptop had an RTX 3060 and 16GB"]),
    ("What kind of feedback does Jamie want from me?", ["honest, direct feedback over praise"]),
    ("When does Jamie like to receive praise?", ["praise only when it is specific and earned"]),
    ("What is Jamie building?", ["building a local AI assistant"]),
    ("What model does Vivianna run?", ["Qwen3.5-9B"]),
    ("What voices does Vivianna's text-to-speech use?", ["af_heart and zf_xiaobei"]),
    ("What is Jamie's long-term goal for Vivianna?", ["open-source Vivianna with credit"]),
    ("What is Jamie working on this week?", ["goal this week is to finish the reranker"]),
    ("What programming language does Jamie prefer?", ["prefers Python over JavaScript"]),
    ("How does Jamie prefer to communicate?", ["dislikes video calls and prefers asynchronous"]),
    ("How does Jamie take his coffee?", ["drinks his coffee black"]),
    ("Why does Jamie run everything locally?", ["values privacy, which is why"]),
    ("What happened with Jamie's relationship recently?",
     ["partner of three years recently left", "partner of three years was named Sam"]),
    ("Who is Jamie's best friend?", ["best friend Tomas"]),
    ("Who takes care of Jamie's daughter?", ["primary caregiver"]),
    ("What is the name of Jamie's daughter?", ["daughter's name is Mia"]),
    ("Where did Jamie grow up?", ["grew up in a small town near Hamburg"]),
    ("When is Jamie's birthday?", ["birthday is in October"]),
    ("Does Jamie have a computer science degree?", ["self-taught developer with no formal CS"]),
    ("Is Jamie willing to donate his organs?", ["not willing to donate his organs"]),
    ("What health issue does Jamie have when sitting too long?", ["chronic back pain"]),
    ("Can Jamie eat dairy?", ["lactose intolerant"]),
    ("Can Jamie eat shellfish?", ["cannot eat shellfish"]),
    ("What does Jamie's brother do?", ["brother is a software developer"]),
    ("What does Jamie's sister do?", ["sister is a professional voice actress"]),
    ("What operating system does Jamie use?", ["uses Windows 11"]),
    ("What does Jamie listen to while coding?", ["listens to lo-fi music"]),
    ("When does Jamie usually work?", ["works late into the night"]),
    ("Who does Jamie credit for Vivianna?", ["credits Claude, GPT, and Qwen"]),
    ("What does Jamie find draining?", ["large social gatherings draining"]),
    ("How long has Jamie been studying machine learning?", ["learning machine learning for about a year"]),
    ("What does Jamie's daughter like to do?", ["daughter likes drawing animals"]),
]


def _resolve_queries(corpus, specs):
    out = []
    for q, subs in specs:
        gold = set()
        for s in subs:
            hits = [i for i, c in enumerate(corpus) if s in c]
            if len(hits) != 1:
                raise ValueError(
                    f"gold substring {s!r} for query {q!r} matched {len(hits)} docs "
                    f"(indices {hits}); must match exactly one."
                )
            gold.add(hits[0])
        out.append((q, gold))
    return out


BIG_MEM_QUERIES = _resolve_queries(BIG_MEM_CORPUS, _QUERY_SPECS)


def summary():
    store = sum(1 for _, s in BIG_SAL_CANDIDATES if s)
    return (f"salience: {len(BIG_SAL_CANDIDATES)} ({store} store / "
            f"{len(BIG_SAL_CANDIDATES) - store} skip) | "
            f"rerank: {len(BIG_MEM_CORPUS)} docs, {len(BIG_MEM_QUERIES)} queries")


if __name__ == "__main__":
    print(summary())  # also triggers gold resolution / assertions
