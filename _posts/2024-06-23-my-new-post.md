---
layout: post
title: "How to do few-shot learning with GPT like Stavropoulos et al.(2024) (in python) - 1"
date: 2024-06-23 16:09 +0800
toc: true
comments: true
categories: [Paper explaining, LLM]
tags: ['METHODS','BRM','few-shot','PRACTICE','support set'] 

---
# Intro
Recently, I read a fascinating paper published in BRM titled [Shadows of Wisdom: Classifying Meta-Cognitive and Morally Grounded Narrative Content via Large Language Models](https://link.springer.com/article/10.3758/s13428-024-02441-0). The paper explores the potential of using large language models (LLMs) for coding in psychology research. Among the various models tested, the open-sourced 'RoBERTa' and few-shot LLMs produced the best results. Interestingly, the zero-shot LLM (GPT-4) also demonstrated adequate accuracy when compared to human coding.

In my opinion, the few-shot LLM stands out as the most viable alternative to human coders. While open-sourced models like RoBERTa rely on fine-tuning and may require more extensive training data, few-shot LLMs can achieve impressive results with fewer examples. Moreover, few-shot LLMs benefit from a broader range of knowledge and simpler training procedures.

In this blog post, I aim to introduce briefly about few-shot learning, and to replicate the study discussed in the paper, focusing specifically on the 'Perspective' dimension, which performed particularly well in the few-shot scenario. You can find the code and data [here](https://osf.io/9dr8p/?view_only=8f6f2df837994e209ff546561fd92e5a).


# Few-shot learning: What is it ?

Few-shot Learning (FSL) is a machine learning framework that uses a few examples to help a more general model learn new concepts. As far as I know, FSL originated from computer vision. It is quite similar to the generalization ability in humans: once we learn 1+1 = 2, we can understand that 10+22 = 32 by applying the patterns or regularities from the examples to similar patterns. This is akin to how teachers use examples in class to help students grasp complex concepts.

FSL has multiple advantages: it is data-efficient, reduces the effort required for data annotation, and is fast to train. With advanced large language models (LLMs) like GPT and Claude, which are relatively easy to use, we can leverage few-shot learning to accomplish many tasks, particularly in psychology, such as annotation or even simulation. For those interested in simulating user behavior, refer to the work by Dr. Xu Chen and his team [here](https://arxiv.org/pdf/2306.02552)

Before diving into few-shot learning, it's important to understand a few key terms:

* **Support Set**: This is the set of labeled data with the new category, serving as the material for the LLM to learn from, which are the examples used in this replication. In traditional machine learning, this is similar to the 'training set'.

* **Query Set**: This is the set of unlabeled data that requires the LLM to label them, akin to the 'test set' in traditional machine learning.

* **N-way K-shot Learning Scheme**: In this scheme, 'N-way' indicates the number of categories to select from, and 'K-shot' refers to the number of examples per category. In this replication, there are only 2 categories, so N is 2.

I will test the annotation accuracy here with different values of K (ranging from 3 to 5) to see if there are significant differences in performance.


# Replication🙌: How to do the few-shot learning like Stavropoulos et al.(2024) in python
*This replication focuses solely on the 'perspective' dimension. All the codes and prompts are available on their[osf](https://osf.io/9dr8p/?view_only=8f6f2df837994e209ff546561fd92e5a)*
![The procedure of use both open-sourced and close-sourced models to annotate in Stavropoulos et al.(2024) ](/assets/img/image.png)
The image above shows how the paper trained their models to annotate text. 

## Preparation of the prompt👩🏻‍💻
The first step is to define the category, construct the coding guide, and **use GPT-4 to clarify the prompt** to ensure there is no ambiguity. For the 'perspective' dimension, the label and definition are as follows:
![alt text](/assets/img/image-2.png)

### Instruction for annotation
With the coding guide in hand, combine it with the prompt to construct the instruction for the LLM (acting as the 'system', which is crucial for understanding the user's intention. I may write a blog later about the differences between system, user, and assistant roles). The following is the prompt construct. The prompt uses the **'personas creation'** technique, widely employed in prompt engineering, to focus the context on a certain area, thus returning answers more relevant to the question. Here, by describing the personas as *'intelligent assistant'* and *specifying the function*, the LLM follows the task more closely.

Also, note that **"###"** is used in the prompt to separate the instructions. For more information, refer to the [OpenAI help](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api)).
Additionally, the prompt restricts the output format

![prompt used in Stavropoulos et al.(2024)](/assets/img/image-1.png)

### Prompt for examples generation
Before they actually sent it to GPT, they asked GPT to generate examples for few-shot training using the examples in the coding guide. Since I haven't found the exact prompt used in Stavropoulos et al. (2024), I edited the prompt from the instruction for this task (which I saved as 'prompt_exp_gen.txt'):

```  

As an intelligent assistant, your primary function is to process and interpret instructions provided in JSON format. Your specific task is to create five examples for the presence and absence of a certain label, respectively, based on the instruction and examples I provided. You should then deliver a succinct one-sentence justification for each decision. Follow the coding guide below and respond with a JSON object in the specified output format.  
  
###

CODING GUIDE:
<CODING GUIDE>

###

OUTPUT FORMAT:
IF PRESENT:
{"code": "present”,”example”:”<EXAMPLE>”, "explanation": "<YOUR EXPLANATION HERE>"}
IF ABSENT:
{"code": "absent",”example”:”<EXAMPLE>”, "explanation": "<YOUR EXPLANATION HERE>"}

```

## Summon GPT with Python 🏴󠁧󠁢󠁷󠁬󠁳󠁿
To start, ensure your Python environment has packages like openai installed and you know your OpenAI API key. For those unfamiliar with the process, refer to OpenAI's [tutorial](https://platform.openai.com/docs/quickstart). 

If you haven't added the API key to your environment, you can add it directly in the project setup:

```python
import openai
client = openai.OpenAI(api_key='<ENTER YOUR OPENAI API KEY HERE>'
```
Download the prompt and examples into your environment. Now, let's begin by letting GPT generate examples for later few-shot learning.

### Ask GPT to generate examples
The goal here is to generate examples for few-shot learning. I requested GPT to generate 5 examples each for the presence and absence of 'perspective'. Here is the code I used to summon GPT:

```python

# generate examples

prompt_exp_gen = os.path.join(material_folder, 'prompt_exp_gen.txt')
with open(prompt_exp_gen,'r') as f:
    prompt_exp_gen = f.readlines()

with open(os.path.join(material_folder,'code_guide.json'),'r') as f:
    code_guide = f.read()

ind_code = prompt_exp_gen.index('<CODING GUIDE>\n')
prompt_exp_gen[ind_code] = code_guide
prompt_exp_gen = ''.join(prompt_exp_gen) # combine them as one text

exp_gen = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": prompt_exp_gen},
  ]
)

examples = json.loads(exp_gen.choices[0].message.content.replace('```','').replace('\n','').replace('json',''))
file_path = os.path.join(material_folder,'pers_exp.json')
with open(file_path, 'w') as file:
    json.dump(examples, file, indent=4) # save the examples in pers_exp.json
```

The examples (or the support set) I generated were like: 
```json
[
    {
        "code": "present",
        "example": "I tried to see things from her perspective, maybe she acted out of stress or fear.",
        "explanation": "The person makes an effort to understand and justify the actions of the other party by considering their possible emotional state."
    },
    {
        "code": "present",
        "example": "Understanding their struggle is key; they might be acting out because of pressures we don't see.",
        "explanation": "The speaker is acknowledging unseen pressures that may influence the other person's behavior, demonstrating empathy."
    },
    {
        "code": "present",
        "example": "From his point of view, the decision might have seemed necessary under the circumstances.",
        "explanation": "The person considers how the circumstances might have influenced the other person\u2019s decision, showing an attempt to understand their reasoning."
    },
    {
        "code": "present",
        "example": "I\u2019ve been thinking about how my actions might affect my colleague's workload and overall stress levels.",
        "explanation": "The person is reflecting on the impact of their own actions on others, indicating an awareness of different perspectives."
    },
    {
        "code": "present",
        "example": "She seemed upset after the meeting, perhaps because she felt ignored.",
        "explanation": "The person is interpreting another's emotional response based on observed behavior, showing empathy and consideration of their feelings."
    },
    {
        "code": "absent",
        "example": "He needs to just follow the rules without making a fuss.",
        "explanation": "There is no attempt to understand or empathize with the other person's viewpoint, focusing only on compliance."
    },
    {
        "code": "absent",
        "example": "I don\u2019t really care why she did it, rules are rules.",
        "explanation": "The person dismisses any consideration of the other person\u2019s reasons or feelings, demonstrating a lack of empathy."
    },
    {
        "code": "absent",
        "example": "This is not the time to question motives, just get the job done.",
        "explanation": "The focus is solely on task completion without any consideration of the other party's perspective or underlying motivations."
    },
    {
        "code": "absent",
        "example": "He could have his reasons, but it doesn't matter here.",
        "explanation": "While acknowledging the possibility of other reasons, there is no effort to understand or consider them."
    },
    {
        "code": "absent",
        "example": "We\u2019ve always done it this way and it works, no need to change anything.",
        "explanation": "The person is dismissive of different perspectives by insisting on the status quo without considering the views or feelings of others."
    }
]
```

Time Limits, see you in the part two, which I will get into the main dish --- Few-shot training!


# References
1. https://spotintelligence.com/2023/06/29/few-shot-learning/#What_is_few-shot_learning
2. https://blog.paperspace.com/few-shot-learning/