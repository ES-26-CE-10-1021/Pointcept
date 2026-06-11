# 3.y Architectural variants

> **Companion to** the §3.x baseline section (the four-block decomposition of
> 3DETR: **pre-encoder → encoder → decoder → prediction heads**). This file
> is the design-chapter context for the architectural variants used in the
> thesis. Dataset specifics (ScanNet vs. AGCO, sensor crops, batch sizes,
> box parameterization) are intentionally **not** discussed here — they
> belong in the implementation chapter.

The previous section established 3DETR as four sequential blocks:
**pre-encoder → encoder → decoder → prediction heads**. This section
introduces the architectural variants used to answer the three research
questions of §3.5. Each subsection is organized around one research
question and identifies which block (or interface between blocks) is
modified to test it. Across all variants, the encoder, decoder, and
prediction-head blocks of the §3.x baseline are reused unchanged unless
otherwise noted, so that any quality delta between two variants can be
attributed to a small and well-defined set of architectural changes.

## 3.y.1 Pre-encoder variants — VFM initialization (RQ1)

RQ1 asks whether initializing the encoder with weights from a 3D vision
foundation model (VFM) — Utonia — improves detection and segmentation
performance compared to training the *same architecture* from scratch. The
research question is therefore explicitly a question about **weight
initialization** rather than about architectural choice. We isolate it by
constructing three variants that all use a PTv3-style sparse-voxel
transformer as the pre-encoder, and differ only in which parameters are
loaded from the Utonia checkpoint and which are trainable. A fourth
variant — the PointNet++ SA pre-encoder of §3.x — is retained as a
**field-standard reference point** so that the entire RQ1 axis can be
related back to the published 3DETR baseline; it is not part of the
strict from-scratch-vs-VFM comparison.

**(a) PointNet++ SA — the §3.x reference baseline.** Described in §3.x.1.
Retained here only for comparison against the published 3DETR architecture.
All parameters are trainable; no pretraining is used.

**(b) PTv3 from scratch — RQ1's from-scratch arm.** The PointNet++ SA stack
is replaced by PointTransformerV3 (Wu et al., 2023), a sparse-voxel point
transformer with a 5-stage hierarchical encoder attending over locally
serialized neighborhoods at progressively coarser voxel resolutions. Its
output is a *variable-length* set of voxel-resolution tokens, one per
occupied voxel at the deepest stage. To restore the fixed-length token
contract that the downstream encoder block expects, a
farthest-point-sampling (FPS) step selects a fixed number of tokens from
this variable-length set. The full pre-encoder block is thus
`PTv3 encoder → FPS`. All parameters are randomly initialized and trained
end-to-end. This variant is the **architectural target** of the RQ1
comparison: it fixes the architecture so that variants (c) and (d) below
differ from it only by weight initialization and freeze policy.

**(c) Utonia, partially fine-tuned ("Utonia ft") — RQ1's partial-transfer
arm.** The PTv3 encoder of variant (b) is initialized with weights from the
**Utonia** point-cloud foundation model — specifically the publicly
released PT-v3m3 checkpoint, trained self-supervised on a large multi-domain
point-cloud corpus that does *not* include agricultural outdoor LiDAR.
Most of the encoder is then frozen: the embedding layer and all but the
last encoder stage are held in evaluation mode and excluded from gradient
updates. The **last encoder stage** remains trainable, allowing the deepest
features to adapt to the downstream task while the bulk of the pretrained
representation is preserved. As in (b), an FPS step is appended to obtain
a fixed-length token sequence. This variant tests whether VFM pretraining
transfers useful features when only minimal task-specific adaptation is
permitted.

**(d) Utonia, fully frozen — RQ1's strong-transfer arm.** Identical to (c)
except that the **entire encoder is frozen**, including the last stage.
No gradients flow into the backbone; the only trainable parameters are the
downstream encoder, decoder, and prediction heads. This variant tests the
strongest form of VFM transfer — whether the Utonia representation is, on
its own, a sufficient feature extractor for downstream tasks on
agricultural LiDAR without any task-specific updates to the backbone.

**Why this design isolates the research question.** Variants (b), (c), and
(d) share the same PTv3 architecture and differ only along two axes: (i)
which weights initialize the encoder (random vs. Utonia), and (ii) how
much of the encoder is trainable (all vs. last stage only vs. none).
Variant (b) acts as the "no VFM" reference, and variants (c) and (d)
together explore how much task-specific adaptation is necessary to leverage
the VFM. Any difference in downstream performance between (b) and (c)/(d)
is attributable to the pretraining; any difference between (c) and (d) is
attributable to the freeze policy.

**Cross-cutting use.** This same set of three PTv3-architecture variants is
reused in RQ3 (§3.y.3) as the backbone axis of the multi-task experiments,
so the RQ1 comparison and the RQ3 comparison are run on architecturally
identical backbones.

## 3.y.2 Image-feature variants — color input and 2D-VFM injection (RQ2)

RQ2 asks whether the inclusion of image-derived features — either as raw
per-point color or as frozen 2D foundation model embeddings — improves
model performance beyond what a geometry-only point cloud encoder
provides. The research question is divided into two hypotheses that test
progressively richer image signals:

- **H1 (per-point color).** Including per-point RGB color as an input
  feature to the model improves performance over the geometry-only
  baseline. Color provides a direct visual cue that complements the purely
  geometric features of LiDAR returns, particularly for objects whose
  shapes are similar but whose appearances are not.
- **H2 (DINOv3 feature injection).** Injecting frozen DINOv3 image features
  into the 3DETR pipeline improves performance further, beyond what raw
  color alone provides, by supplying learned visual semantics that are not
  easily obtained from color values.

The two hypotheses correspond to two distinct architectural changes — one
that widens the input contract of the pre-encoder, and one that inserts a
new block into the architecture — described in turn below. Both
modifications operate on the same interface (the pre-encoder), so they can
be combined or studied in isolation without affecting any of the
downstream blocks of §3.x.

### Variant 1: per-point RGB color (H1)

The simpler of the two image-feature variants threads raw RGB color through
the existing 3DETR pipeline by changing only the **input channel contract**
of the pre-encoder. No new architectural block is introduced. The
implementation differs slightly per backbone:

- **PointNet++ SA** and **PTv3 from scratch** accept an arbitrary input
  feature dimensionality. The pre-encoder simply widens its first
  projection layer to accept three additional input channels when color is
  provided.
- **Utonia** was pretrained on a fixed 9-channel input contract
  `[xyz, rgb, normal]`. The pretrained backbone therefore always expects
  nine input channels; modalities not present in the dataset (color,
  surface normals, or both) are **zero-filled** at the input boundary. The
  color-off variant zero-fills the RGB slot of the nine-channel input; the
  color-on variant fills it with measured RGB values. The backbone
  weights, architecture, and trainable surface area are bit-identical
  between the two — only the input tensor changes. This zero-fill
  construction is what allows the Utonia color ablation to be a clean
  A/B test.

This variant is structurally minimal: each RQ1 backbone has a color-on
counterpart, the rest of the architecture is unchanged, and the
trainable-parameter count grows by at most a few hundred weights (the
widened first projection).

### Variant 2: DINOv3 feature fusion (H2)

The richer of the two image-feature variants introduces a **new
architectural block** that is not present in the §3.x baseline: a fusion
module that combines the 3D tokens emitted by the pre-encoder with 2D
image features from a frozen DINOv3 backbone. The DINOv3 backbone itself
is held fixed throughout training — only the fusion module and the
downstream blocks receive gradients. The fusion block sits at the
interface between the pre-encoder and the transformer encoder, so the
encoder, decoder, and head blocks remain unchanged. This placement is
deliberate: the pre-encoder output is the **common interface** across
every RQ1 backbone variant, so a fusion mechanism attached at that point
can be applied uniformly to any of the RQ1 pre-encoders without further
architectural change.

Concretely, the fused architecture expands to five blocks:

```
   raw point  ──▶  pre-encoder  ──▶  fusion  ──▶  encoder  ──▶  decoder  ──▶  heads
   cloud + image                       ▲
                                       │
                          DINOv3 ──────┘
                          (frozen, 2D)
```

The fusion module receives 3D tokens from the pre-encoder, each with an
associated 3D anchor coordinate, and 2D patch features from the DINOv3
backbone applied to one or more co-captured RGB images. It associates each
3D token with the visual evidence relevant to its spatial location (the
specific association mechanism — geometric projection of the token's
coordinate into the image plane, cross-attention between 3D tokens and 2D
patches, or a learned mixture of the two — is specified in the
implementation chapter). The output is a sequence of fused tokens of the
same length and effective dimensionality as the pre-encoder output, ready
to be consumed by the §3.x transformer encoder.

The DINOv3 fusion strictly extends the color variant: it provides the same
visual modality (RGB images) but processed through a 2D foundation model
that has been pretrained on internet-scale imagery. The hypothesis is
therefore not about *whether* the model has access to visual information —
the color variant already provides that — but about whether learned 2D
semantic structure is meaningfully richer than raw color values.

### Experimental matrix

H1 and H2 are tested against the same point-cloud-only baselines drawn
from §3.y.1. Three image-feature settings are instantiated for each
relevant RQ1 backbone:

- **Geometry only** — no color, no DINO. Coincides with the §3.y.1
  pre-encoder variants.
- **+RGB color** — per-point color appended at the pre-encoder input.
  Tests H1.
- **+DINO fusion** — frozen DINOv3 patch features fused into the
  pre-encoder output. Tests H2.

Comparing geometry-only to +RGB-color directly tests H1; comparing
+RGB-color to +DINO-fusion directly tests H2; comparing geometry-only to
+DINO-fusion gives the total effect of image-derived features in either
form. We run this image-feature axis across more than one RQ1 backbone so
that any backbone-specific interactions (for example, a pretrained
backbone making weaker use of additional input modalities than a
from-scratch backbone) are visible rather than hidden behind a single
backbone choice.

### Why the fusion placement matters

Placing the DINO fusion at the pre-encoder output, rather than (for
example) at the decoder's cross-attention or at the input to the
pre-encoder, has two design advantages. First, the pre-encoder output is
the only point in the architecture where every RQ1 backbone variant
produces tokens with the same contract, so the fusion module can be
reused without modification across all backbones. Second, the fusion
happens *before* the transformer encoder, so the encoder block — which is
the component that mixes information globally across the scene — has full
access to the fused signal when refining tokens, rather than seeing the
3D and 2D modalities only at the decoder stage.

## 3.y.3 Prediction-head variants — multi-task learning (RQ3)

RQ3 asks whether attaching multiple task-specific prediction heads to a
single 3DETR backbone preserves the per-task performance achieved by
independently trained single-task models. The concern is **negative
transfer**: when two tasks share parameters, their gradients may conflict,
and the shared representation may degrade to a compromise that serves
neither task as well as a single-task model would. The opposing
possibility is that multi-task supervision regularizes the shared
representation and yields per-task improvements.

This axis modifies the **prediction-head block** of the §3.x architecture
and, in the multi-task case, also changes which point in the network the
prediction blocks tap into. Three task variants are defined, and each is
combined with three of the §3.y.1 pre-encoders (PTv3 from scratch,
Utonia ft, Utonia frozen) to give a 3×3 matrix of pre-encoder × task
combinations. This factorization lets us examine the multi-task question
independently of the choice of backbone — and conversely, it lets us see
whether VFM pretraining changes the answer to RQ3.

### Detection only

The §3.x detection-only architecture from §3.y.1: pre-encoder, transformer
encoder, DETR decoder, and the parallel MLP heads for class, center, size,
and angle. The detection variants under RQ3 are exactly the RQ1 variants
(b)–(d) of §3.y.1, repurposed here as the *detection-only* row of the 3×3
matrix.

### Segmentation only

The detection-specific blocks of §3.x — the transformer encoder, the DETR
decoder, and the box-parameter MLP heads — are removed. The pre-encoder
block from §3.y.1 is extended with its **decoder symmetric**: where the
detection-only variant uses the pre-encoder's deepest features as tokens
for the DETR decoder, the segmentation-only variant runs the full PTv3
U-Net (encoder + skip-connected decoder) to produce per-point features at
the original input resolution. A single linear classification head over
each point's feature vector emits a per-class logit. The loss is a sum of
cross-entropy and Lovász-softmax (Berman et al., 2018), each weighted
equally; this is a standard segmentation loss combination in the
point-cloud literature.

For frozen-backbone variants (Utonia frozen), the pretrained encoder is
held fixed exactly as in §3.y.1(d), and the U-Net decoder is added
**fresh** on top — randomly initialized and trained from scratch. This is
necessary because the Utonia checkpoint distributes only the encoder
weights; no pretrained decoder is available. For the partially-fine-tuned
slot (Utonia ft), the same fresh decoder is used, and the last encoder
stage remains trainable.

### Multi-task: detection and segmentation jointly

The multi-task variant combines the detection-only and segmentation-only
models into a single network with a **shared backbone**. The PTv3 encoder
runs once; its output is consumed by two parallel prediction stacks:

- The **segmentation branch** runs the U-Net decoder over the full encoder
  output, producing per-point features that are passed to the segmentation
  head exactly as in the segmentation-only variant.
- The **detection branch** taps the **encoder bottleneck** (the deepest,
  lowest-resolution encoder features) and feeds them into the §3.x
  transformer encoder, DETR decoder, and box-parameter heads, exactly as in
  the detection-only variant of §3.y.1.

Critically, the two branches share only the *backbone encoder*: each
branch has its own decoder block and its own prediction heads. The
segmentation branch's U-Net decoder does not feed the detection branch,
and vice versa. The branches diverge at the encoder bottleneck and operate
independently from that point onward.

This wiring matches the structural argument behind the RQ3 hypothesis: the
3DETR per-query features inside the detection branch are abstract and
task-agnostic, and they remain so regardless of whether a segmentation
head is also being trained on the same backbone. The shared backbone
encoder is what carries information between the two tasks. Whether that
sharing helps or hurts is precisely the empirical question RQ3 asks.

**Loss combination.** The total loss is a weighted sum of the segmentation
loss and the detection set-prediction loss. We adopt **uncertainty
weighting** (Kendall et al., 2018): rather than fix the relative weight of
the two losses as a hyperparameter, we introduce a single learnable scalar
per task — interpreted as the log-variance of a task-specific likelihood —
and let the optimizer trade the two off automatically. The composite loss
has the form

$$\mathcal{L}_{\text{total}} = \frac{1}{c_{\text{seg}} \sigma_{\text{seg}}^2} \mathcal{L}_{\text{seg}} + \frac{1}{c_{\text{det}} \sigma_{\text{det}}^2} \mathcal{L}_{\text{det}} + \log \sigma_{\text{seg}} + \log \sigma_{\text{det}}$$

where $\sigma_{\text{seg}}, \sigma_{\text{det}}$ are the learnable
uncertainty scalars and $c_{\text{seg}} = c_{\text{det}} = 2$ are constants
of the Cipolla et al. (2018) form of the loss. The two log-$\sigma$
parameters are excluded from weight decay; otherwise they would be biased
toward zero and the loss would collapse to the high-uncertainty regime.

Using a learned weighting is particularly important for RQ3 because we
want to control for the confounding effect of an unfavorable manual loss
balance: any per-task degradation observed in the multi-task model should
be attributable to negative transfer in the shared representation, not to
the wrong loss weights.

## 3.y.4 Synthesis: how the three research questions intersect the four blocks

```
                    pre-encoder         fusion          encoder        decoder         heads
                    (input + arch)
   §3.x baseline    PointNet++ SA       —               Vanilla 3L     8L DETR         box params
   ─────────────────────────────────────────────────────────────────────────────────────────────
   RQ1              ◆ varied (4) ◆      —               fixed          fixed           fixed
                    (init / freeze)
   RQ2 / H1         ◆ +RGB channel ◆    —               fixed          fixed           fixed
                    (input widened)
   RQ2 / H2         fixed (within          ◆ added ◆    fixed          fixed           fixed
                    each H2 row)        (DINOv3, frozen)
   RQ3              ◆ varied (3) ◆      —               fixed          varied:         ◆ varied ◆
                    (init / freeze)                                    det / seg /     (det / seg /
                                                                       both            both)
```

RQ1 isolates the **pre-encoder's weights** (initialization and freeze
policy). RQ2 isolates the **image-feature signal** entering the model,
splitting it into two architecturally distinct changes: H1 widens the
input channel contract of the pre-encoder to accept per-point RGB color,
and H2 inserts a new fusion block between the pre-encoder and the
transformer encoder that injects frozen 2D-VFM features. RQ3 isolates the
**prediction heads** (and, for the multi-task variant, the wiring from the
backbone to those heads). The encoder block is held fixed across all three
research questions; the decoder block varies in RQ3 only because the
segmentation task uses a U-Net decoder rather than the DETR decoder. This
factorization is what makes the experiment matrix interpretable: any
quality delta between two cells in the matrix can be attributed to a small
and well-defined set of architectural changes.

---

## Framing notes (for the thesis author, not for the chapter)

- The **specific 2D→3D fusion mechanism** for the H2 DINO variant in
  §3.y.2 is described abstractly ("the fusion module associates each 3D
  token with the visual evidence relevant to its spatial location"). The
  implementation chapter will need to commit to a concrete mechanism —
  geometric projection + patch-feature sampling, cross-attention between
  3D tokens and 2D patches, or a learned combination of the two. The
  design section as written leaves room to choose; once the choice is
  made it can be lifted into §3.y.2 (variant 2) as a short paragraph
  without disturbing the surrounding structure.
- The H1 / H2 framing is **mechanically asymmetric**: H1 only widens the
  pre-encoder's input contract (a few extra parameters), whereas H2
  introduces a new architectural block (a frozen 2D backbone plus a
  trainable fusion module — many more parameters and a meaningfully more
  expensive forward pass). This asymmetry is real and worth being
  explicit about in the chapter — the two hypotheses test progressively
  more invasive interventions, not two siblings of equal architectural
  weight.
- A fifth pre-encoder variant — Utonia frozen with the §3.x transformer
  encoder block replaced by an identity mapping — was originally trained
  as a "linear-probing-like" reference point but does not actually
  function as linear probing (the 8-layer DETR decoder remains trainable).
  It is kept out of this design section and retained as an ablation for
  the discussion chapter.
- The uncertainty-weighting equation is the **Cipolla form**. If the
  simpler 2018-paper form without the $c_i = 2$ constants is preferred,
  the math reduces accordingly and the constants disappear.
- The "Synthesis" ASCII block in §3.y.4 is a placeholder for what could
  become Table 3.y in the thesis — drop or convert to a real LaTeX/TikZ
  table as needed.
- Dataset specifics (AGCO/ScanNet/ouster, oriented vs. axis-aligned boxes,
  number of queries, FPS budget, batch sizes) are intentionally absent —
  they belong in the implementation chapter alongside the per-config field
  tables in [`3detr_config_overview.md`](3detr_config_overview.md) and the
  experiment matrix in [`thesis-experiment-context.md`](thesis-experiment-context.md).
