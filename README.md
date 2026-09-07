# Saya Comfy Couple+

> [!WARNING]
> **Work in Progress**
>
> Saya Comfy Couple+ is still under active development.
>
> The project already works and can be used in real ComfyUI workflows, but the internal routing, node layout, ports, behavior and documentation may still change while the project is being refined.
>
> If you build an important workflow around it, keep a backup before updating.

## What is Saya Comfy Couple+?

**Saya Comfy Couple+** is a modified and expanded Comfy Couple implementation for ComfyUI.

It is designed primarily for workflows where one image contains **one or two distinct characters** and you want to keep:

* the global scene
* Person 1 identity
* Person 2 identity
* regional attention
* IPAdapter references
* detailer prompts

separated from each other instead of mixing everything into the same conditioning.

The basic idea is simple:

```text
MAIN
Shared scene, composition, action, pose, lighting, background and mood

PERSON 1
Identity and appearance of the first character

PERSON 2
Identity and appearance of the second character

NEGATIVE
Shared negative prompt
```

The goal is not simply to create two masks.

The goal is to give ComfyUI a cleaner way to understand:

```text
What belongs to the whole image?

What belongs specifically to Person 1?

What belongs specifically to Person 2?
```

This becomes especially useful in larger automatic workflows using regional prompting, IPAdapter and detailers.

---

# Why does this project exist?

The original Comfy Couple approach is useful for regional two-character generation, but its prompt structure is relatively simple:

```text
positive_1
positive_2
negative
```

In a complex workflow, that often means each character prompt must contain several unrelated things at once:

```text
scene
+ composition
+ character identity
+ regional information
+ detailer information
```

That works, but it becomes harder to maintain and harder to reason about.

Saya Comfy Couple+ instead separates the prompt into:

```text
main_positive
person_1_positive
person_2_positive
negative
```

The node then combines the relevant information internally.

For Person 1:

```text
Person 1 context
=
main_positive
+
person_1_positive
```

For Person 2:

```text
Person 2 context
=
main_positive
+
person_2_positive
```

Both characters therefore receive the same scene information while keeping their own identity information.

---

# How it works

A simplified generation looks like this:

```text
                    MAIN
                     │
             ┌───────┴───────┐
             │               │
             ▼               ▼
         PERSON 1         PERSON 2
             │               │
             ▼               ▼
      MAIN + PERSON 1   MAIN + PERSON 2
             │               │
             └───────┬───────┘
                     │
                     ▼
              Regional Couple
                conditioning
                     │
                     ▼
                  Sampler
```

The important part is that `MAIN` is not treated as a weak unrelated condition.

It becomes part of both regional character contexts.

This makes the prompt structure much easier to understand:

```text
MAIN tells the image what is happening.

PERSON 1 tells the first region who Person 1 is.

PERSON 2 tells the second region who Person 2 is.
```

---

# Example

Imagine this image:

```text
Two characters sitting together on a bed in a gamer bedroom.
Soft evening lighting.
Medium shot.
```

Person 1 is:

```text
short blue hair,
white eyes,
rabbit ears,
petite body,
white hoodie
```

Person 2 is:

```text
long pink hair,
red eyes,
black horns,
tall body,
black dress
```

Instead of repeating the bedroom, pose and lighting in both character prompts:

```text
MAIN

two characters,
sitting together on a bed,
modern gamer bedroom,
soft evening lighting,
medium shot
```

```text
PERSON 1

short blue hair,
white eyes,
rabbit ears,
petite body,
white hoodie
```

```text
PERSON 2

long pink hair,
red eyes,
black horns,
tall body,
black dress
```

Internally, Saya Comfy Couple+ builds:

```text
REGION 1

two characters,
sitting together on a bed,
modern gamer bedroom,
soft evening lighting,
medium shot,
short blue hair,
white eyes,
rabbit ears,
petite body,
white hoodie
```

and:

```text
REGION 2

two characters,
sitting together on a bed,
modern gamer bedroom,
soft evening lighting,
medium shot,
long pink hair,
red eyes,
black horns,
tall body,
black dress
```

This keeps the scene shared without forcing the two identities into the same prompt.

---

# Main features

Saya Comfy Couple+ currently provides:

* separate `MAIN`, `PERSON 1` and `PERSON 2` conditioning
* shared negative conditioning
* regional couple attention
* automatic Person 1 and Person 2 masks
* horizontal and vertical region layouts
* configurable region split position
* dedicated outputs for workflow routing
* IPAdapter `attn_mask` support
* detailer-oriented conditioning outputs
* solo and duo workflow support
* safer handling of different encoded conditioning lengths

---

# Outputs and routing

The node exposes several outputs because different parts of a large workflow usually need different information.

## `model`

The model patched with Saya Comfy Couple attention logic.

Normally:

```text
Saya Comfy Couple+ model
→ KSampler model
```

---

## `full_positive`

This is the main positive conditioning used for generation.

It contains the regional structure built from:

```text
MAIN + PERSON 1
MAIN + PERSON 2
```

Normally:

```text
full_positive
→ KSampler positive
```

---

## `negative`

Shared negative conditioning.

Normally:

```text
negative
→ KSampler negative
```

It can also be reused by detailers and other workflow branches.

---

## `main_positive`

The original shared scene conditioning.

Useful when another part of your workflow needs only global information.

Examples:

```text
scene
pose
composition
background
lighting
shared action
style
```

---

## `person_1_positive`

The original Person 1 conditioning.

Useful for:

```text
Person 1 detailers
debugging
identity-specific routing
custom workflow branches
```

---

## `person_2_positive`

The original Person 2 conditioning.

Useful for the same purposes as Person 1, but for the second character.

---

## `duo_positive`

Combined Person 1 and Person 2 identity conditioning without the full scene prompt.

A common use is:

```text
duo_positive
→ detailer positive
```

This is useful when a detailer should know which characters exist without receiving every background, composition or lighting instruction from `MAIN`.

---

## `mask_positive_1`

Automatic regional mask for Person 1.

Typical usage:

```text
mask_positive_1
→ Person 1 IPAdapter attn_mask
```

---

## `mask_positive_2`

Automatic regional mask for Person 2.

Typical usage:

```text
mask_positive_2
→ Person 2 IPAdapter attn_mask
```

This allows two different IPAdapter references to be spatially routed toward their respective characters.

---

# Inputs

## `model`

Connect the model you want Saya Comfy Couple+ to patch.

For example:

```text
Checkpoint Loader
→ LoRA
→ Saya Comfy Couple+
```

or simply:

```text
Checkpoint Loader
→ Saya Comfy Couple+
```

---

## `main_positive`

Use this for anything that applies to the image as a whole.

Good examples:

```text
number of characters
scene
pose
composition
camera framing
camera angle
shared action
background
lighting
mood
global style
```

Example:

```text
two characters,
sitting together,
bedroom,
soft lighting,
medium shot
```

Avoid putting Person 1 or Person 2 identity information here unless that characteristic should genuinely apply to both characters.

---

## `person_1_positive`

Use this for Person 1 identity.

Examples:

```text
hair
eyes
ears
horns
body type
clothes
accessories
character-specific traits
```

---

## `person_2_positive`

Same idea, but for Person 2.

For a solo workflow, this input can be empty or disabled depending on how your workflow handles empty conditioning.

---

## `negative`

Your shared negative conditioning.

Use the same negative prompt you would normally use for your model and workflow.

---

## `orientation`

Controls the direction of the automatic regional split.

Available modes:

```text
horizontal
vertical
```

Choose the mode that best matches the expected placement of the characters.

---

## `center`

Controls where the separation between the two regions happens.

Examples:

```text
0.50
equal split

0.40
Person 1 side becomes smaller and Person 2 side becomes larger

0.60
Person 1 side becomes larger and Person 2 side becomes smaller
```

The exact visual result depends on orientation.

---

## `width` / `height`

Resolution used to build the internal automatic masks.

These values should correspond to the canvas used by your generation workflow.

---

# Quick start

A minimal duo workflow looks like this:

```text
Checkpoint / LoRA
        │
        ▼
Saya Comfy Couple+
        │
        ├── model ────────────→ KSampler model
        │
        ├── full_positive ────→ KSampler positive
        │
        └── negative ─────────→ KSampler negative
```

Prompt connections:

```text
Shared scene CLIP
→ main_positive

Person 1 CLIP
→ person_1_positive

Person 2 CLIP
→ person_2_positive

Negative CLIP
→ negative
```

That is enough to use the main regional generation system.

---

# Tutorial 1 - Solo generation

Saya Comfy Couple+ can also be used when only one character is present.

Use:

```text
MAIN

solo,
one character,
bedroom,
sitting on bed,
medium shot,
soft lighting
```

```text
PERSON 1

short blue hair,
white eyes,
rabbit ears,
white hoodie
```

```text
PERSON 2

empty / disabled conditioning
```

```text
NEGATIVE

your normal negative prompt
```

The important principle stays the same:

```text
MAIN
=
what the image is doing

PERSON 1
=
who the character is
```

This allows the workflow to keep the same prompt architecture when switching between solo and duo generations.

---

# Tutorial 2 - Duo generation

For two characters:

```text
MAIN

two characters,
sitting together,
bedroom,
medium shot,
soft evening lighting
```

```text
PERSON 1

short blue hair,
white eyes,
rabbit ears,
petite body
```

```text
PERSON 2

long pink hair,
red eyes,
black horns,
tall body
```

Then connect:

```text
Saya model
→ sampler model

full_positive
→ sampler positive

negative
→ sampler negative
```

The node internally creates two regional contexts:

```text
MAIN + PERSON 1
MAIN + PERSON 2
```

This is the core behavior of Saya Comfy Couple+.

---

# Tutorial 3 - Two-character IPAdapter

Saya Comfy Couple+ also provides masks that can be used as IPAdapter attention masks.

For Person 1:

```text
Person 1 reference image
→ IPAdapter Person 1

mask_positive_1
→ IPAdapter Person 1 attn_mask
```

For Person 2:

```text
Person 2 reference image
→ IPAdapter Person 2

mask_positive_2
→ IPAdapter Person 2 attn_mask
```

Conceptually:

```text
PERSON 1 reference
        │
        ▼
   IPAdapter P1
        ▲
        │
 mask_positive_1


PERSON 2 reference
        │
        ▼
   IPAdapter P2
        ▲
        │
 mask_positive_2
```

The goal is to prevent both references from blindly influencing the entire image.

Each reference instead receives the regional mask associated with its character.

---

# Tutorial 4 - Detailers

Detailers often need different prompt information from the main sampler.

The main sampler needs:

```text
scene
composition
background
characters
regional information
```

A face or body detailer often cares much more about:

```text
character identity
appearance
clothing
character-specific features
```

For a simple shared detailer setup:

```text
duo_positive
→ detailer positive

negative
→ detailer negative
```

`duo_positive` contains the character information without forcing the detailer to reread the entire shared scene prompt.

This is especially useful in automatic workflows where using separate detailer branches for every character would unnecessarily increase complexity and generation time.

If your workflow uses separate Person 1 and Person 2 detailers, the raw person outputs are also available:

```text
person_1_positive
→ Person 1 detailer

person_2_positive
→ Person 2 detailer
```

---

# Recommended prompt organization

A good rule is:

## MAIN

Describe:

```text
WHAT is happening
WHERE it happens
HOW the image is framed
HOW the scene is lit
```

Example:

```text
two characters,
sitting together,
modern bedroom,
medium shot,
warm evening lighting
```

## PERSON 1

Describe:

```text
WHO Person 1 is
```

Example:

```text
short blue hair,
white eyes,
rabbit ears,
petite body,
white hoodie
```

## PERSON 2

Describe:

```text
WHO Person 2 is
```

Example:

```text
long pink hair,
red eyes,
black horns,
tall body,
black dress
```

Do not put:

```text
blue hair,
white eyes
```

inside `MAIN` unless both characters are supposed to receive those traits.

---

# Why prompt separation matters

Consider this prompt:

```text
two girls,
bedroom,
blue hair,
pink hair,
red eyes,
white eyes,
rabbit ears,
horns
```

A diffusion model sees all of those concepts together.

It does not automatically know that:

```text
blue hair
white eyes
rabbit ears
```

belong exclusively to Person 1 while:

```text
pink hair
red eyes
horns
```

belong exclusively to Person 2.

Saya Comfy Couple+ gives the workflow explicit structure for separating those concepts spatially.

It cannot guarantee perfect character binding in every generation, but it gives the model and workflow much cleaner information to work with.

---

# Automatic regional masks

Saya Comfy Couple+ creates two masks corresponding to the two character regions.

Their layout is controlled by:

```text
orientation
center
width
height
```

A simplified horizontal example:

```text
┌──────────────────────────────┐
│                              │
│     PERSON 1 | PERSON 2      │
│                              │
└──────────────────────────────┘
```

A different `center` value changes the relative size of the two regions.

These masks are also exposed to the workflow so other systems such as IPAdapter can reuse the same spatial organization.

---

# Conditioning length safety

Character prompts do not always encode to the same context length.

For example:

```text
Person 1
short prompt

Person 2
much longer prompt with many identity details
```

Saya Comfy Couple+ includes handling for different encoded conditioning lengths.

Shorter context tensors are padded where necessary before internal concatenation.

This helps prevent tensor-size mismatch errors when regional conditioning lengths differ.

This is a safety mechanism.

It does **not** remove or bypass the underlying CLIP token limits of the model.

---

# Difference from the original Comfy Couple

Original structure:

```text
positive_1
positive_2
negative
```

Saya Comfy Couple+:

```text
main_positive
person_1_positive
person_2_positive
negative
```

Additional routing:

```text
full_positive
main_positive
person_1_positive
person_2_positive
duo_positive
mask_positive_1
mask_positive_2
```

Core idea:

```text
PERSON 1 REGION
=
MAIN + PERSON 1

PERSON 2 REGION
=
MAIN + PERSON 2
```

This gives complex workflows a cleaner separation between global scene information and character-specific identity information.

---

# Installation

## Git

Clone the repository inside your ComfyUI `custom_nodes` directory.

Linux / macOS:

```bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/alphaziod/saya-comfy-couple-plus.git
```

Windows PowerShell:

```powershell
cd C:\ComfyUI\custom_nodes
git clone https://github.com/alphaziod/saya-comfy-couple-plus.git
```

ComfyUI Portable:

```powershell
cd C:\ComfyUI_windows_portable\ComfyUI\custom_nodes
git clone https://github.com/alphaziod/saya-comfy-couple-plus.git
```

Then restart ComfyUI.

If ComfyUI is installed somewhere else, use your actual `custom_nodes` path.

---

# Updating

Linux / macOS:

```bash
cd ~/ComfyUI/custom_nodes/saya-comfy-couple-plus
git pull
```

Windows:

```powershell
cd C:\ComfyUI\custom_nodes\saya-comfy-couple-plus
git pull
```

Restart ComfyUI after updating.

Because the project is still WIP, keeping a backup of important workflows before major updates is recommended.

---

# Uninstalling

Remove the repository from `custom_nodes`.

Linux / macOS:

```bash
rm -rf ~/ComfyUI/custom_nodes/saya-comfy-couple-plus
```

Windows PowerShell:

```powershell
Remove-Item -Recurse -Force C:\ComfyUI\custom_nodes\saya-comfy-couple-plus
```

Then restart ComfyUI.

---

# Troubleshooting

## The node does not appear

Make sure the repository exists inside:

```text
ComfyUI/custom_nodes/
```

For example:

```text
ComfyUI/
└── custom_nodes/
    └── saya-comfy-couple-plus/
```

Then completely restart ComfyUI.

Also check the ComfyUI startup log for Python import errors.

---

## I updated the node but the workflow still shows the old ports

Fully restart ComfyUI.

If necessary, delete the old node from the workflow and create it again so ComfyUI rebuilds its input/output definition.

---

## Person 1 and Person 2 appear reversed

Check:

```text
orientation
center
```

Then verify your external routing:

```text
mask_positive_1
→ Person 1 IPAdapter

mask_positive_2
→ Person 2 IPAdapter
```

Also verify that the correct character prompts are connected to the correct person inputs.

---

## Character identity feels weak

Keep character-specific information inside:

```text
person_1_positive
```

or:

```text
person_2_positive
```

Remember that the regional contexts are constructed as:

```text
MAIN + PERSON 1
MAIN + PERSON 2
```

If all identity information is placed in `MAIN`, the separation between characters becomes much less meaningful.

---

## My detailer receives too much scene information

Try:

```text
duo_positive
→ detailer positive
```

instead of:

```text
full_positive
→ detailer positive
```

`duo_positive` focuses on character information without including the complete main scene context.

---

## Different prompt lengths cause problems

Saya Comfy Couple+ includes padding logic for different encoded conditioning lengths.

If you still encounter a tensor-size error, report the error together with:

```text
ComfyUI version
checkpoint/model
resolution
workflow
full traceback
```

This project is still WIP, so reproducible bug reports are particularly useful.

---

# Current project status

> [!CAUTION]
> **Saya Comfy Couple+ is not considered finished yet.**

The project is currently being actively tested and refactored.

The current implementation already supports real workflows, but development is still focused on improving:

* reliability
* character separation
* regional routing
* automatic workflow behavior
* compatibility
* maintainability
* documentation
* edge-case handling

Some behavior may therefore change between versions.

Do not assume that every internal API, node port or experimental behavior is permanently frozen yet.

---

# What this project is not

Saya Comfy Couple+ is not intended to:

* replace every regional prompting solution
* guarantee perfect identity separation in every image
* magically fix badly structured prompts
* replace IPAdapter
* replace detailers
* replace the sampler

Instead, it acts as a **routing and regional conditioning layer** that helps these components work together in a cleaner two-character workflow.

---

# Who is this for?

Saya Comfy Couple+ is mainly aimed at users building more advanced ComfyUI workflows involving:

* anime or illustration generation
* one or two characters
* separate character identities
* regional prompting
* IPAdapter references
* detailers
* automatic generation pipelines
* reusable prompt blocks

Simple workflows may not need this level of separation.

For larger workflows, however, keeping scene logic and character identity logic separate can make the graph significantly easier to maintain.

---

# Development

This project is currently WIP.

Bug reports, reproducible examples and technical feedback are useful while the architecture is still being refined.

When reporting a problem, please include as much of the following as possible:

```text
ComfyUI version
model/checkpoint
resolution
relevant custom nodes
error traceback
workflow or minimal reproduction
expected behavior
actual behavior
```

---

# Credits

Saya Comfy Couple+ is based on the Comfy Couple concept and expands it for more structured solo/duo prompt routing and larger automatic ComfyUI workflows.

---

# License

See the repository license for the applicable terms.
