LLM Evaluation Tasks
====================

ProgramAsWeights excels at evaluation tasks that researchers need but are tedious to implement manually. These programs replace expensive API calls with fast, local evaluation.

Answer Equivalence Checking
----------------------------

**Problem:** Exact string matching gives 0% accuracy for semantically correct but differently worded answers.

**Solution:** Use a neural program that understands semantic equivalence.

.. code-block:: python

   import programasweights as paw
   
   # Load the answer equivalence checker
   equiv_checker = paw.function("programasweights/answer-equiv")
   
   # Examples where exact match fails but semantic match succeeds
   examples = [
       ("42", "forty-two"),                    # Number formats
       ("Paris", "Paris, France"),            # Geographic variants  
       ("Yes", "Correct"),                    # Synonymous answers
       ("2023-01-15", "January 15, 2023"),    # Date formats
   ]
   
   for pred, target in examples:
       # Format as evaluation prompt
       prompt = f"Prediction: {pred}\nTarget: {target}"
       result = equiv_checker(prompt)
       print(f"{pred} ≟ {target} → {result}")

**Output:**
::

   42 ≟ forty-two → equivalent
   Paris ≟ Paris, France → equivalent  
   Yes ≟ Correct → equivalent
   2023-01-15 ≟ January 15, 2023 → equivalent

Response Quality Scoring
-------------------------

**Problem:** Need to automatically grade LLM responses for quality, completeness, and relevance.

.. code-block:: python

   import programasweights as paw
   
   quality_scorer = paw.function("programasweights/quality-scorer")
   
   responses = [
       "The capital of France is Paris.",                    # Good: direct, correct
       "Well, um, I think it might be Paris maybe?",        # Poor: uncertain, wordy
       "Paris is the capital and largest city of France.",  # Excellent: complete, informative
       "I don't know.",                                     # Poor: unhelpful
   ]
   
   for response in responses:
       score = quality_scorer(f"Question: What is the capital of France?\nAnswer: {response}")
       print(f"'{response[:30]}...' → Score: {score}/10")

Factual Accuracy Assessment
---------------------------

.. code-block:: python

   factual_checker = paw.function("programasweights/factual-checker")
   
   claims = [
       "The Earth is round.",                    # Factual
       "Paris is the capital of Germany.",      # Incorrect
       "Water boils at 100°C at sea level.",    # Factual
       "Humans can breathe underwater.",        # Incorrect
   ]
   
   for claim in claims:
       accuracy = factual_checker(claim)
       print(f"'{claim}' → {accuracy}")

Automated Research Pipeline
---------------------------

**Complete evaluation pipeline for research papers:**

.. code-block:: python

   import programasweights as paw
   import json
   from typing import Dict, List
   
   class AutoEvaluator:
       def __init__(self):
           self.equiv_checker = paw.function("programasweights/answer-equiv")
           self.quality_scorer = paw.function("programasweights/quality-scorer") 
           self.factual_checker = paw.function("programasweights/factual-checker")
           self.consistency_checker = paw.function("programasweights/consistency-checker")
       
       def evaluate_response(self, question: str, prediction: str, 
                           target: str) -> Dict[str, float]:
           # Check answer equivalence
           equiv_prompt = f"Prediction: {prediction}\nTarget: {target}"
           is_equivalent = self.equiv_checker(equiv_prompt) == "equivalent"
           
           # Score response quality
           quality_prompt = f"Question: {question}\nAnswer: {prediction}"
           quality = float(self.quality_scorer(quality_prompt))
           
           # Check factual accuracy
           factual = self.factual_checker(prediction) == "factual"
           
           return {
               "exact_match": prediction.strip() == target.strip(),
               "semantic_match": is_equivalent,
               "quality_score": quality,
               "factual_accuracy": factual,
           }
       
       def evaluate_dataset(self, qa_pairs: List[Dict]) -> Dict[str, float]:
           results = []
           
           # Process in batches for efficiency
           for item in qa_pairs:
               metrics = self.evaluate_response(
                   item["question"], 
                   item["prediction"], 
                   item["target"]
               )
               results.append(metrics)
           
           # Aggregate metrics
           return {
               "exact_match_acc": sum(r["exact_match"] for r in results) / len(results),
               "semantic_match_acc": sum(r["semantic_match"] for r in results) / len(results),
               "avg_quality": sum(r["quality_score"] for r in results) / len(results),
               "factual_acc": sum(r["factual_accuracy"] for r in results) / len(results),
               "total_examples": len(results)
           }

   # Usage
   evaluator = AutoEvaluator()
   
   # Your model's outputs
   test_data = [
       {
           "question": "What is the capital of France?",
           "prediction": "The capital city is Paris",
           "target": "Paris"
       },
       # ... more examples
   ]
   
   metrics = evaluator.evaluate_dataset(test_data)
   print(json.dumps(metrics, indent=2))

Comparison with Traditional Evaluation
---------------------------------------

**Before (manual/API-based):**

.. code-block:: python

   import openai
   
   def evaluate_manually(pred, target):
       # Expensive API call for each comparison
       response = openai.chat.completions.create(
           model="gpt-4",
           messages=[{
               "role": "user",
               "content": f"Are these equivalent?\nA: {pred}\nB: {target}"
           }]
       )
       return "yes" in response.choices[0].message.content.lower()
   
   # Cost: $0.01-0.10 per evaluation
   # Speed: 1-3 seconds per evaluation
   # Reliability: Depends on API availability

**After (ProgramAsWeights):**

.. code-block:: python

   import programasweights as paw
   
   equiv_checker = paw.function("programasweights/answer-equiv")
   
   def evaluate_with_paw(pred, target):
       prompt = f"Prediction: {pred}\nTarget: {target}"
       return equiv_checker(prompt) == "equivalent"
   
   # Cost: Free after initial download
   # Speed: 10-50ms per evaluation  
   # Reliability: Always available offline

Benefits for Researchers
------------------------

- **Reproducible:** Same program version gives identical results
- **Fast:** 100x faster than API calls for large evaluations
- **Offline:** No internet dependency during evaluation
- **Consistent:** No API changes breaking your evaluation pipeline
- **Cost-effective:** No per-evaluation charges

Next: :doc:`bias-analysis` 