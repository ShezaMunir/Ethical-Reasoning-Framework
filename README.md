## Ethical Reasoning Framework

* **Goal:** Aggregate conflicting ethical judgments by prioritizing **reasoning quality over majority votes**

* **Problem:** Majority voting suppresses minority perspectives and leads to inconsistent decisions in subjective domains

* **Approach:**

  * Use an LLM to extract structured moral features from text (e.g., harm, intent, empathy, apology)
  * Assign **quality-based weights** to each piece of reasoning
  * Convert outputs into logical constraints
  * Apply a **Weighted MaxSAT (Z3) solver** to compute the most consistent final decision

* **Key Idea:** Separate **content (what is said)** from **quality (how well it is argued)** to ensure stronger reasoning has more influence

* **Outcome:** Produces decisions that are more **logically consistent, explainable, and robust to bias and popularity effects**

* **Use Case:** Designed for high-conflict domains like moral judgment (e.g., AITA-style scenarios) where disagreement is meaningful, not noise

### Paper abstract:
Standard methods for aggregating natural language judgments, such as majority voting, often fail to produce logically consistent results when applied to high-conflict domains, treating differing opinions as noise. We propose a neuro-symbolic aggregation framework that formalizes conflict resolution through Weighted Maximum Satisfiability (MaxSAT). Our pipeline utilizes a language model to map unstructured natural language explanations into interpretable logical predicates and confidence weights. These components are then encoded as soft constraints within the Z3 solver, transforming the aggregation problem into an optimization task that seeks the maximum consistency across conflicting testimony. Using the Reddit *r/AmItheAsshole* forum as a case study in large-scale moral disagreement, our system generates logically coherent verdicts that diverge from popularity-based labels 62\% of the time, corroborated by an 86\% agreement rate with independent human evaluators. This study demonstrates the efficacy of coupling neural semantic extraction with formal solvers to enforce logical soundness and explainability in the aggregation of noisy human reasoning.
