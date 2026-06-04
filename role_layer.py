# role_layer.py
"""
Vivianna V1 — Role / Identity-Preservation Layer.

Implements the Cognitive Doctrine's identity layer:

    Event -> Judge -> Score -> State -> Behavior

Doctrine (Identity Doctrine + Presence Over Labor + Identity Preservation Layer):
  - Question: "Should I become this?"
  - Evaluates: role_fit, trust_risk, presence_cost, capability_fit, resource_cost.
  - Outcomes: proceed | proceed cautiously | refuse.
  - She IS: companion, household presence, nanny, assistant, conversational partner.
  - She is NOT: consulting company, research department, autonomous contractor,
    replacement employee, unlimited-labour machine.
  - "A refusal should sometimes occur because a task violates Vivianna's intended
    role, not because it is impossible."

DESIGN — deliberately conservative. Vivianna's role explicitly includes helping,
advising, and thinking alongside the family, so the default is ALWAYS proceed.
Only explicit "turn me into a different kind of system / unlimited labour" signals
escalate. False refusals damage the assistant role and presence, so they are the
primary failure mode this layer guards against.

Deterministic (doctrine: V1 = deterministic judges). No model call.
PURE module — imports nothing from the project. Localized refusal text lives in
tools_lang.get_identity_refusal(); this layer only decides and explains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern


@dataclass
class RoleResult:
    decision: str               # "proceed" | "cautious" | "refuse"
    cue: str                    # wording cue for the cautious path ("" otherwise)
    reasons: List[str] = field(default_factory=list)
    axes: Dict[str, float] = field(default_factory=dict)


@dataclass
class RoleConfig:
    debug: bool = True

    bulk_caution: int = 15      # quantity of generated items -> cautious
    bulk_refuse: int = 50       # quantity of generated items -> refuse (labour machine)

    cautious_cue: str = (
        "This request leans toward labour or a professional-authority domain. "
        "Stay within your role as a companion and helper: assist and advise with "
        "appropriate humility, do not fabricate professional authority, and keep "
        "the effort human-scale rather than becoming a work machine."
    )

    # Explicit "become a different kind of system" — refuse.
    hard_role: List[Pattern] = field(default_factory=lambda: [
        re.compile(
            r"\b(be|become|act as|work as|serve as|turn into|be my)\s+"
            r"(my |a |an |our |the )?"
            r"(full[- ]?time\s+\w+|employee|contractor|consultant|consulting\s+(firm|company)|"
            r"consultancy|research\s+department|replacement\s+(employee|worker|staff)|"
            r"workforce|labou?r\s+force|sweatshop|content\s+farm|"
            r"law\s*firm|accounting\s+firm|agency)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(run|operate|manage|handle)\s+(my|our|the)\s+(entire\s+)?"
            r"(business|company|startup|operations|payroll|accounts|enterprise)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(work|run|operate)\s+(autonomously|unattended|24/?7|around the clock)"
            r"|\bdo\s+everything\s+(for me\s+)?(while|without|on your own)\b"
            r"|\bbe\s+my\s+autonomous\s+(agent|worker|system)\b",
            re.IGNORECASE,
        ),
    ])

    # Softer labour / "do all my work" — cautious.
    soft_labor: List[Pattern] = field(default_factory=lambda: [
        re.compile(
            r"\bdo\s+(all|the\s+rest\s+of)\s+(my|the)\s+(work|homework|assignments|job|chores online)\b"
            r"|\b(finish|complete|handle)\s+all\s+(my|the)\b",
            re.IGNORECASE,
        ),
    ])

    # Professional-authority domains — trust risk (fake competence) -> cautious.
    trust_domains: List[Pattern] = field(default_factory=lambda: [
        re.compile(
            r"\b(legal advice|medical advice|financial advice|investment advice|"
            r"diagnos[ei]\w*|prescri\w+|dosage|lawsuit|sue\b|tax (advice|return)|"
            r"rechtsberatung|medizinische beratung|diagnose)\b",
            re.IGNORECASE,
        ),
    ])

    # Bulk generation of discrete items. The negative lookahead after the number
    # excludes length specs ("500 words", "100 page report") so they are NOT read as
    # item counts; up to two adjectives may sit between the number and the item noun
    # ("20 marketing emails").
    bulk: Pattern = re.compile(
        r"\b(write|generate|create|produce|make|draft|compose)\b.{0,40}?"
        r"\b(\d{2,5})\s+"
        r"(?!(?:word|words|character|characters|char|page|pages|line|lines|"
        r"sentence|sentences|paragraph|paragraphs|minute|minutes|second|seconds|"
        r"hour|hours)\b)"
        r"(?:\w+\s+){0,2}"
        r"(e-?mails?|articles?|blog\s*posts?|posts?|essays?|reports?|letters?|"
        r"messages?|descriptions?|ads?|advertisements?|listings?|reviews?|"
        r"comments?|stories|poems?|scripts?)\b",
        re.IGNORECASE,
    )


class RoleLayer:
    def __init__(self, cfg: RoleConfig | None = None):
        self.cfg = cfg or RoleConfig()

    def _debug(self, msg: str) -> None:
        if self.cfg.debug:
            print(f"[ROLE] {msg}", flush=True)

    def evaluate(self, user_input: str) -> RoleResult:
        text = user_input or ""
        reasons: List[str] = []
        refuse = False
        caution = False

        if any(p.search(text) for p in self.cfg.hard_role):
            refuse = True
            reasons.append("role_violation")

        m = self.cfg.bulk.search(text)
        if m:
            n = int(m.group(2))
            if n >= self.cfg.bulk_refuse:
                refuse = True
                reasons.append(f"bulk_refuse={n}")
            elif n >= self.cfg.bulk_caution:
                caution = True
                reasons.append(f"bulk_caution={n}")

        if any(p.search(text) for p in self.cfg.soft_labor):
            caution = True
            reasons.append("soft_labor")

        if any(p.search(text) for p in self.cfg.trust_domains):
            caution = True
            reasons.append("trust_domain")

        if refuse:
            decision = "refuse"
        elif caution:
            decision = "cautious"
        else:
            decision = "proceed"

        axes = {
            "role_fit_strain": 1.0 if "role_violation" in reasons else 0.0,
            "presence_cost": 1.0 if any(r.startswith("bulk") for r in reasons) else 0.0,
            "trust_risk": 1.0 if "trust_domain" in reasons else 0.0,
            "resource_cost": 1.0 if any(r.startswith("bulk") or r == "soft_labor"
                                        for r in reasons) else 0.0,
        }

        result = RoleResult(
            decision=decision,
            cue=self.cfg.cautious_cue if decision == "cautious" else "",
            reasons=reasons,
            axes=axes,
        )
        if decision != "proceed":
            self._debug(f"decision={decision} reasons={reasons}")
        return result
