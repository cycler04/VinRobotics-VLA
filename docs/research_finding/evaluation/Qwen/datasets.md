# Evaluation Datasets for Qwen Models

> **Question:** What kinds of data sit behind the main Qwen evaluation suites,
> how large are they, and what does each example contain?
>
> **Scope:** Public evaluation data for language and vision-language Qwen
> checkpoints. This is not a reconstruction of Qwen's proprietary training
> corpus. Dataset facts were checked on 2026-07-22.

## Scope boundary

This file describes only the examples, modalities, annotations, splits, scale,
access and licenses. Prompts, preprocessing, evaluators and reported Qwen
results belong in [benchmarks.md](benchmarks.md); score definitions belong in
[metrics.md](metrics.md); training objectives belong in [loss.md](loss.md).

## Dataset map

| Dataset | Type and modalities | Published scale | Principal splits/configurations | What one example contains |
|---|---|---:|---|---|
| MMLU-Pro | Text, difficult multiple-choice knowledge/reasoning | 12,032 test + 70 validation questions | 14 subject domains; 10 answer choices | Question, options, answer and subject |
| GPQA | Text, graduate-level science multiple choice | 448 questions total; Diamond has 198 | Biology, physics and chemistry; Diamond is the strictest subset | Expert-written question, choices and answer |
| IFEval | Text instruction-following constraints | 541 prompts, 25 verifiable instruction types | Public evaluation collection; each prompt has 1–3 constraints | Prompt plus machine-checkable instruction IDs |
| MMMU-Pro | Image-plus-text expert reasoning | 1,730 examples in each released configuration | Standard 10-choice, standard 4-choice and vision-only | Question, one or more images, options, subject metadata |
| MathVista | Visual mathematical reasoning | 6,141 examples from 31 sources | `testmini` 1,000; full `test` 5,141 | Image, question, answer type and source/task metadata |
| OCRBench | OCR and text-rich visual QA | 1,000 manually verified QA pairs | Five task families; original OCRBench | Image, text-related question and reference answer |
| RefCOCO | Referring-expression grounding in COCO images | 19,994 images, 50,000 objects, 142,210 expressions | Train 120,624; val 10,834; testA 5,657; testB 5,095 expressions | Image, phrase and referred object box/region |
| Video-MME | Video question answering | 900 videos, 254 hours, 2,700 QA pairs | Short, medium and long; 6 domains/30 subfields | Video, three multiple-choice questions, optional subtitles |

Counts are version-specific. In particular, original OCRBench is not OCRBench
v2, and original Video-MME is not Video-MME v2. A local dataset manifest must
pin the named release and configuration.

## Text datasets

### MMLU-Pro

MMLU-Pro is a harder revision of broad academic multiple-choice evaluation. It
expands the choice set from four to ten, removes many noisy or trivial items,
and emphasizes questions that require reasoning rather than direct recall. Its
14 domains include mathematics, physics, chemistry, law, engineering, health,
history, psychology and business. The public release exposes 12,032 test rows
and a small 70-row validation set.

It is useful as a broad regression set, but it remains text-only and public. It
does not test whether a visual reference is correct or whether a robot action is
safe. [MMLU-Pro paper][mmlu-pro] [Official repository][mmlu-pro-repo]

### GPQA and GPQA-Diamond

GPQA contains 448 multiple-choice questions written and validated by domain
experts in biology, physics and chemistry. The authors designed them to resist
answers obtained by ordinary web search. `GPQA-Diamond` is a 198-question subset
selected for the highest agreement and quality; it is not a separate large
corpus.

It is a small, high-difficulty science collection rather than a broad-coverage
knowledge dataset.
[GPQA paper][gpqa]

### IFEval

IFEval has 541 prompts carrying one to three constraints drawn from 25
machine-verifiable instruction types. Examples request properties such as exact
formatting, word occurrence, casing, list structure or the presence/absence of
specified content. The annotation is therefore a set of constraint checkers,
not a free-form human preference label.

This construction makes the annotations machine-verifiable, but it
covers only instructions with deterministic checks. [IFEval paper][ifeval]
[Official implementation][ifeval-repo]

## Vision-language datasets

### MMMU-Pro

MMMU-Pro revises expert multimodal questions to reduce shortcuts. The release
has three 1,730-example configurations:

- `standard (10 options)`, the primary harder multiple-choice form;
- `standard (4 options)`, retained for comparison with older protocols;
- `vision`, where question and options are rendered into the image so that the
  input cannot be solved through a text-only path.

It spans many university and professional subjects and may attach one or more
figures to a question. The three configurations are alternative renderings of
the task, not 5,190 independent semantic questions. [MMMU-Pro paper][mmmu-pro]
[Dataset card][mmmu-pro-data]

### MathVista

MathVista combines 28 existing sources with three newly created sources:
IQTest, FunctionQA and PaperQA. Its 6,141 questions cover figures, charts,
geometry, scientific plots, documents and puzzle-like images. The compact
`testmini` split has 1,000 examples; the remaining 5,141 form the full test set.
Test answers are withheld for server evaluation, so local files may not contain
everything needed for standalone scoring. [MathVista paper][mathvista-paper]
[Project][mathvista]

### OCRBench

Original OCRBench is a compact, manually verified set of 1,000 image-question
pairs organized into five groups:

1. text recognition;
2. scene-text visual question answering;
3. document visual question answering;
4. key-information extraction;
5. handwritten mathematical expression recognition.

It mixes natural scenes, documents and handwriting, so a single aggregate can
hide camera-domain failures. OCRBench v2 is a separate, much larger bilingual
benchmark with 10,000 QA pairs and 31 scenarios; results from the two versions
must not share a column. [OCRBench paper][ocrbench-paper]
[Official repository][ocrbench]

### RefCOCO

RefCOCO is built on MS-COCO images and crowd-written referring expressions. It
contains 142,210 expressions for 50,000 object instances in 19,994 images.
`testA` is dominated by people, while `testB` contains other objects.

The target is an object region, so the dataset tests whether a phrase such as
“the cup to the left of the plate” resolves to the intended object. It is a
static 2D grounding dataset, not a 3D pose or action dataset.
[RefCOCO paper][refcoco]

### Video-MME

Original Video-MME contains 900 videos totaling about 254 hours and 2,700
question-answer pairs. Durations range from roughly 11 seconds to one hour. The
collection covers six top-level domains and 30 subfields and is stratified into
short, medium and long video groups. Audio and subtitle tracks are retained so
that protocols can evaluate with or without subtitle information.

The raw dataset contract is richer than a sampled-frame tensor. A local copy
should preserve the original video, timestamps, duration class, subtitles and
question IDs before a model-specific frame sampler is applied. Video-MME v2 is
a later dataset and must be versioned separately. [Video-MME paper][video-mme-paper]
[Project][video-mme]

## Access and reproduction details

Most datasets above have public code or data entry points, but “public” does not
mean every evaluation is fully local:

- MathVista keeps full-test labels behind an evaluation service.
- GPQA distribution has access and license conditions that must be checked at
  download time.
- image/video datasets may inherit licenses or access terms from COCO, source
  videos, or component datasets;

Record at least the following in a local dataset manifest:

```yaml
name: Video-MME
version: original
source_revision: <commit-or-dataset-revision>
split: test
example_ids: <path-or-hash>
modalities: [video, audio, subtitles, text]
native_scale: {videos: 900, qa_pairs: 2700}
native_media_metadata: <timestamps-duration-audio-subtitle-tracks>
license_or_terms: <checked-url-and-date>
```

For every dataset, preserve example IDs, native media metadata, split, source
revision and any filtered/excluded rows. Prompting and metric settings belong in
the run manifest, not in the dataset description.

## Training overlap and local status

Qwen reports broad text, image, OCR, grounding and video training mixtures, but
not a sample-level manifest sufficient to rule out overlap with every public
evaluation set. Contamination should therefore be recorded as **unknown**, not
assumed absent.

No dataset in this report was downloaded or ingested into this workspace during
this research. The current repository has canonical dataset inspection and
conversion tools, but no Qwen evaluation runner.

## Sources

- Wang et al. *MMLU-Pro*. [Paper][mmlu-pro] · [Repository][mmlu-pro-repo]
- Rein et al. *GPQA*. [Paper][gpqa]
- Zhou et al. *IFEval*. [Paper][ifeval] · [Implementation][ifeval-repo]
- Yue et al. *MMMU-Pro*. [Paper][mmmu-pro] · [Dataset][mmmu-pro-data]
- Lu et al. *MathVista*. [Paper][mathvista-paper] · [Project][mathvista]
- Liu et al. *OCRBench*. [Paper][ocrbench-paper] · [Repository][ocrbench]
- Yu et al. *Modeling Context in Referring Expressions*. [Paper][refcoco]
- Fu et al. *Video-MME*. [Paper][video-mme-paper] · [Project][video-mme]

[mmlu-pro]: https://arxiv.org/abs/2406.01574
[mmlu-pro-repo]: https://github.com/TIGER-AI-Lab/MMLU-Pro
[gpqa]: https://arxiv.org/abs/2311.12022
[ifeval]: https://arxiv.org/abs/2311.07911
[ifeval-repo]: https://github.com/google-research/google-research/tree/master/instruction_following_eval
[mmmu-pro]: https://arxiv.org/abs/2409.02813
[mmmu-pro-data]: https://huggingface.co/datasets/MMMU/MMMU_Pro
[mathvista-paper]: https://arxiv.org/abs/2310.02255
[mathvista]: https://mathvista.github.io/
[ocrbench-paper]: https://arxiv.org/abs/2305.07895
[ocrbench]: https://github.com/qywh2023/OCRbench
[refcoco]: https://openaccess.thecvf.com/content_cvpr_2016/html/Yu_Modeling_Context_in_CVPR_2016_paper.html
[video-mme-paper]: https://arxiv.org/abs/2405.21075
[video-mme]: https://video-mme.github.io/home_page.html
