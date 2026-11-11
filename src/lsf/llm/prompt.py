"""Palier 3 : construction du prompt « suite de top-5 -> phrase proposee ».

Ce module ne fait AUCUN appel reseau. Le choix du fournisseur de LLM et tout
appel payant restent a decider explicitement.
"""

from __future__ import annotations

Top5 = list[tuple[str, float]]

SYSTEM = (
    "Tu recois la sortie d'un systeme de reconnaissance de signes isoles : pour chaque "
    "signe produit, les 5 gloses les plus probables avec leur probabilite. Le systeme "
    "fait des erreurs ; la bonne glose n'est pas toujours la premiere, ni meme presente. "
    "Propose UNE phrase plausible qui utilise au plus une glose par segment, dans l'ordre. "
    "Tu ne traduis pas la langue des signes : les gloses perdent le visage, l'espace et "
    "les classificateurs. Si aucune phrase n'est plausible, reponds « (incertain) »."
    " Reponds au format : PHRASE: ... | GLOSES: g1 g2 ..."
)


def format_segments(segments: list[Top5]) -> str:
    lines = []
    for i, top in enumerate(segments, 1):
        lines.append(f"Segment {i}: " + ", ".join(f"{g} ({p:.2f})" for g, p in top))
    return "\n".join(lines)


def build_messages(segments: list[Top5], examples: list[tuple[list[Top5], str]] = (), language: str = "anglais") -> list[dict]:
    """Messages au format chat (role/content), independants du fournisseur."""
    msgs = []
    for segs, answer in examples:
        msgs.append({"role": "user", "content": format_segments(segs)})
        msgs.append({"role": "assistant", "content": answer})
    msgs.append({"role": "user", "content": format_segments(segments) + f"\nLangue de la phrase : {language}."})
    return [{"role": "system", "content": SYSTEM}] + msgs
