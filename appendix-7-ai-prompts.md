# Appendix 7. Prompts used with the AI tool

Prompts used with the AI tool Claude (Anthropic) in the bachelor thesis *Automating Citable-Fact Detection in Academic Texts* (Riga Technical University, 2026).

The chats were held in a Claude account that is not the author’s own and that cannot create public share links, so the chats cannot be viewed by others. As the supervisor’s rules allow in this case, the full prompts are given below.

E-mails and comments from the supervisor that were part of some prompts are not reproduced, to protect personal data; they are marked in square brackets.

## Anthropic, 2026a — chat “Creating a reference list for semester report”, 13 June 2026

**Prompt:**

> I attached my paper and these are supervisors sugestions for extending the paper so the chapter two would be more than 7 pages:

> [The supervisor’s suggestions for Chapter 2 followed here; it is not reproduced.]

## Anthropic, 2026b — chat “Dataset sizes and best reported results from research papers”, 17 June 2026

**Prompt:**

> Prompt 1: Zeng & Acuna, 2020 (PMOA-CITE) , Roostaee, 2022 , find the dataset sizes for table 2.1. and Find best reported results for table 2.2

> Prompt 2: you did not give me the dataset sizes of Färber et al., 2018 , Bonab et al., 2018 (SEPID-cite) ,Gosangi et al., 2021 (ACL-cite) , Redi et al., 2019 , Hafid et al., 2024

**Additional prompts (about 12 July 2026, chat “Dataset sizes and domains from reference papers”):**

> I would like you to tell me the size each dataset that was used in every paper that was included in the reference list. Because my dataset will need to be around the same size. I will need to construct a dataset from freely available sources, in which existing citation markers serve as ground-truth labels for citable facts. I would like to know the domain of the each dataset that was used in every paper that is in the reference list as well.

> yea I will go with balanced set. I have one question regarding balancing by augmenting. Also I need some guidance with my dataset construction task: construct a dataset from freely available sources, in which existing citation markers serve as ground-truth labels for citable facts. I need clear steps on how to construct a dataset.

## Anthropic, 2026c — chat “Continuing thesis draft 1 and GPU script porting”, 1–6 October 2026

**Prompt:**

> Prompt 1 (1 October 2026): the answer to the sugesstion is yes. Instead of injserting fig.3.1 as png create it in words with figures and etc.

> Prompt 2 (3 October 2026): 1. create the table 2.3 in the document. 2. you have written what fixes need to be done but it is really hard for me to locate where exactly these fixes should be done. please indicate location for each one.

## Anthropic, 2026d — chat of 11 October – 2 November 2026

**Prompt:**

> Prompt 1 (11 October 2026): My supervisor gave answers. Also there is structure problem and for fixing that i attached the thesis template as well. So lets start fixing everything he wants fixed. Also this is his answer for AI tool usage: [the document provided by supervisor about declaring AI tool use followed].

> [The supervisor’s comments in the attached file asked for a definition of each reason in Table 2.2 and of each pattern in Table 2.3, and for Fig. 2.1 as a bitmap; they are not reproduced.]

> Prompt 2 (11 October 2026), the answer “3. yes” to the tool’s proposal: “Move the group bullets of 3.3.1–3.3.2, Table 3.3 and the examples in 3.4.1 to Appendix 2, which now has the full table. Keep 2–3 sentences in 3.3 that point there. Keep Tables 3.1–3.2 and the main numbers in the chapter. Yes or no?”

> Prompt 3 (25 October 2026): i have edited so change this docx

## Anthropic, 2026e — chat “Coursera courses for thesis machine learning skills”, to 13 September 2026

**Prompt:**

> Prompt 1: I want to understand which courses in coursera are the best for me to learn so that I will be able to do my thesis myself. I have some understanding ding of machine learning but I am not advanced. So do a deep research on every course in coursers and let me know which courses I need to complete.

> Prompt 2: give me extract py as well I deleted by mistake

## Anthropic, 2026f — chat “Bachelor thesis on automated citable-fact detection: engineering brief”, 13–19 September 2026

**Prompt :**

> Part two — then start the work in the same reply. Do not wait for me to approve part one before beginning. Go straight to the blocker in section 4: regenerate both dataset variants from one version of `extract_and_label.py` and verify the keep-holes output is a strict superset, then score the two annotation sheets with `compare_annotators.py`. Give me the code and the commands to run, since the data sits on my machine, and tell me exactly what output to paste back.

> [The prompt continued with a description of the thesis, the schedule, the environment, the planned experiments and the formatting and writing rules. It contains personal data and is not reproduced in full.]



**Additional prompts (19 September 2026 and subsequent exchanges):**

> Prompt 1: Activity
>
> [pasted: the university's deadline table for the bachelor thesis (progress report, 75% thesis submission, pre-defence, topic and supervisor approval, final submission, defence), about 60 lines]
>
> and the attached file is the progress report I submitted last semester. The pasted text shows the requirements for this semester. According to the deadlines create me a roadmap and plan so that I can complete each stage(50%, 75%) successfully [attached: 5thsem_report-\_2026_06_09.docx]

> Prompt 2: I want to continue with chapter 2. Please provide me with guideline

> Prompt 3: check the fig3.2 also fill the table 3.4 from the check_rerun.py output also check the skeleton as whole and fix/update what needs to be fixed/updated.

> Prompt 4: yes I want a more readable graph also take into account that I as a reader should be able to match the values of the dots on axises. right now it is hard to read as well.

> Prompt 5: [pasted: her own draft of the Chapter 4 introduction (the section before 4.1), about 1 paragraph] this is what I wrote for introduction to chapter4(the section before 4.1). I want you to tell me what I can add, change, improve etc.


## Anthropic, 2026g — chat “KeyError in data split script”, 18–25 September 2026

**Prompt:**

> Prompt 1: [terminal output of `make_splits.py` ending in `KeyError: '2209.15314'`] what is the reason of this error and how to fix it? this is the first step of the roadmap
