---
title: Why DSPy?
layout: intro-image-right
image: image.png
theme: apple-basic
class: text-white
clicks: 0
author: Justin Irabor
keywords: ai, python, dspy, llm
exportFilename: why-dspy
monaco: true
colorSchema: dark
---
# Why DSPy?
An introduction to rigor, testability<br/>modularity and algorithmic optimization<br/>of LLM workflows.

<div class="absolute bottom-10">
  <p class="text-xs leading-loose">
    Technical Exchange @ IU<br/>September 19, 2024
  </p>
</div>

---
layout: statement

---
# The problem with prompting

<div class="text-left">

At sufficient levels of complexity, prompts are difficult to:

* Maintain: multiple LLMs working together at various steps have an implicit dependency 

* Modularize: it's difficult to reason about individual steps of an LLM workflow

* Optimize: getting the best out of your prompts might require manual trial and error to get right.
</div>

---
layout: center

---

```python
    # Pseudocode for illustration purposes
    from llms import base_assistant
    import prompts from .prompts_library
    from api_services import coursebook, coursebook_vector_db

    # Multi-step LLM process:
        # 1. Fetch from coursebook API
        # 2. Create a summary of coursebook
        # 3. Use RAG for student queries

    
    course_book = await coursebook.get("DLM1238") # 1
    course_summary = base_assistant.run(prompt=prompts["summarize"], document=course_book) # 2
    # 3?

```
---
layout: quote

---
"DSPy is a framework for algorithmically optimizing LM prompts and weights"
<br/>
— Official DSPy Docs

---
layout: section

---

# Components

1. Signatures

2. Modules

3. Optimizers

---
layout: section

---

# Signatures

Declarative specs for defining the behaviour of DSPy modules. <br/> Signatures allow us to move from telling LLMs what to do without specifying how*

``` python
    document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a..."""
    summarize = dspy.ChainOfThought('document -> summary')
    response = summarize(document=document)

    print(response.summary) # The 21-year-old Lee made seven appearances and scored one goal for West Ham last season...
    print(response.rationale) # Rationale: produce the summary. We need to highlight the key points about Lee's performance...
```

---
layout: center

---
## With signatures, we can specify our desired output, and leave it to the LLM to plan its path to the outcome.

<div class="text-left">
<h3>Examples:</h3>
<li>QA: "question -> answer"</li>
<li>Summaization: "document -> summary", "longtext -> gist", "verbose -> tl;dr"</li>
<li>RAG: "context, query -> answer"</li>
</div>
(That is, declaratively)
---
layout: section

---

# Modules

Abstractions over specific prompting techniques (eg Chain of Thought, ReAct, etc).
\
\
The design of DSPy modules takes inspiration from PyTorch’s NN modules.
\
\
Modules can be configured to your design requirements (`n` for number of passes/completions, `temperature`, `max_len` of output, etc)

---

# Module Example
### A demonstration with Chain of Thought and a RAG pipeline

```python

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words") # signature description


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages) # retrieval model
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

```

---

# Optimizers

This is where the magic happens.
Optimizers are DSPy algorithms that tune the parameters of a DSPy program (prompts/weights) to maximize a goal metric (accuracy, performance, etc)

Optimizers work by evaluating the following:

* Your DSPy program, which it evaluates against
* Your metric (a function that scores the output of your program), which is tested against
* Your training inputs.

Optimizers optimize three internal properties of the DSPy modules as they pertain to your use case: the LM weights, the instructions and input/output demonstrations. _This is what ends up leading to superior prompts than may be possible by hand-crafting._

DSPy offers a handful of Optimizers. Let's take a look at a simple one: BootstrapFewShot.

---

# BootstrapFewShot
This optimizer works by using a _teacher_ module (your DSPy program, by default), to generate and end-to-end run of your LLM workflow step, as well as the labeled examples you supply in your trainset.

The optimizer then uses the metric you specified to validate the demonstrations (ie, data from the trainset and data generated by the _teacher_), and then only uses the successful demonstrations to compile your program's prompt.

---

# Demonstration

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=250)
dspy.settings.configure(lm=turbo)

gms8k = GSM8K()

trainset, devset = gms8k.train, gms8k.dev

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)


evaluate = Evaluate(devset=devset[:], metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=False)

cot_baseline = CoT()

evaluate(cot_baseline, devset=devset[:])

teleprompter = BootstrapFewShotWithRandomSearch(
    metric=gsm8k_metric, 
    max_bootstrapped_demos=8, 
    max_labeled_demos=8,
)

cot_compiled = teleprompter.compile(CoT(), trainset=trainset, valset=devset)

cot_compiled.save('iu_demo.json')

```


---
layout: center

---
# Who needs DSPy?
Short answer, not most people! DSPy is designed with a specific kind of builder in mind:
- One who has a clearly-defined, multi-step LLM pipeline.
- One who has a task that can otherwise be articulated as a good ol' ML problem. 

### To this effect, you must be able to:
- Define the task in terms of expected inputs and outputs as well as resource constraints.
-  Define your pipeline.
-  Get training and validation data.
- Clearly define your metric to optimize against.

---

# Thank you


### repo link: https://github.com/vunderkind/dspy-presentation
<br>
<br>


### References
- The DSPy paper: https://arxiv.org/pdf/2310.03714
- DSPy docs: https://dspy-docs.vercel.app/docs
- Building a RAG framework with auto reasoning and prompting with DSPy: https://www.youtube.com/watch?v=6rN9ozzdT3A
- Replit's few-shot code diff pipeline with DSPy: https://blog.replit.com/code-repair
