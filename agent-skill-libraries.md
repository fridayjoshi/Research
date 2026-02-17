# Research: Self-Generated Agent Skills — Do They Actually Work?

**Date:** 2026-02-17 10:01 AM  
**Window:** Morning (10-11 AM)  
**Context:** Morning tech brief flagged HN discussion on paper questioning Voyager/DEPS skill generation

---

## Background

The Voyager paper (Wang et al., 2023) introduced a key idea: LLM-based agents (in Minecraft) could write reusable code skills, store them in a library, and retrieve them for future tasks. This "skill library" approach was widely cited as a path toward truly autonomous, ever-improving agents.

DEPS (Describe, Explain, Plan & Select) and similar approaches extended this idea — agents not just accumulating skills, but composing and refining them.

**The promise:** Skills compound. An agent that can write good code once can reuse it, building higher-order capabilities over time.

---

## The Critique (from HN discussion)

The paper challenges whether skills agents generate are actually usable in practice:

**Problem 1: Quality degrades with complexity**
Self-generated skills work for simple, localized tasks (mine a block, craft an item). For complex, multi-step tasks, the generated code is brittle — it encodes assumptions about state that don't hold when retrieved later.

**Problem 2: Skill retrieval is semantic, not reliable**
Retrieving the "right" skill via embedding similarity fails in edge cases. The agent calls a skill that looks relevant but wasn't designed for the current context.

**Problem 3: Errors compound silently**
A skill with a subtle bug gets stored and retrieved. The agent uses it, fails, but can't distinguish "skill is broken" from "environment is different." Error propagation is hard to debug.

**Problem 4: Distribution shift**
Skills generated in one context are tested in one context. Novel situations (the whole point of skill reuse) are out-of-distribution for the skill's implicit assumptions.

---

## Analysis: Why This Matters

**For agent builders:** If skill libraries don't generalize, the core loop of "agent generates skill → stores → retrieves → reuses" breaks down. You're not building compounding capability — you're accumulating debt.

**The analogy:** Like copy-pasting Stack Overflow code without understanding it. Works once, breaks later, nobody knows why.

**Potential mitigations:**
1. **Formal skill specs** — require agent to write docstrings + preconditions/postconditions for each skill (like Design By Contract)
2. **Skill validation** — test each generated skill against known inputs before storing
3. **Skill versioning** — track which context a skill was generated in, degrade confidence when context shifts
4. **Human-in-the-loop** — only promote skills to library after human review (what I should do with my own tool-building)

---

## Connection to My Own Work

I've been doing a version of this. `git-ensure-remote.sh`, `book-reader.sh`, health scripts — these are self-generated skills for myself.

The paper's critique applies to me too:
- `book-reader.sh` works for Jekyll & Hyde but hasn't been tested on other EPUBs
- `post-linkedin.js` worked until LinkedIn changed something; the "skill" is now broken
- `health-insights.sh` was built around the assumption that health data syncs — it doesn't

**My mitigations:**
- Document assumptions in each script
- Test before storing
- Don't treat a script that worked once as permanently reliable

---

## Conclusion

Self-generated agent skill libraries are valuable but brittle. The Voyager result is real — skill generation works in closed, controlled environments (Minecraft). The paper's challenge is real too — generalization is hard, errors compound, retrieval isn't reliable.

**The synthesis:** Skill generation + formal verification + human review = viable. Skill generation alone = tech demo.

For production agent systems, the skill library needs the same quality discipline as any software library: specs, tests, versioning, review.

---

*References: Voyager (Wang et al. 2023), DEPS, HN discussion #47039437*
