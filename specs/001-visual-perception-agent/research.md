# Research: Self-Improving Visual Perception Agent

**Date**: 2025-12-17
**Purpose**: Technology research and best practices for implementing the visual perception agent

## Technology Decisions

### 1. Segmentation Model: SAM2 (Segment Anything Model 2)

**Decision**: Use SAM2 for instance segmentation with uncertainty detection

**Rationale**:
- **Foundation model**: Pre-trained on massive datasets, excellent zero-shot performance
- **Instance segmentation**: Native support for identifying individual object instances
- **Uncertainty estimation**: Access to logit scores enables confidence measurement
- **Fine-tuning capable**: Supports incremental learning with new data
- **Open-source**: Apache 2.0 license from Meta AI
- **Production-ready**: Well-documented, actively maintained, strong community

**Alternatives Considered**:
- **Mask R-CNN**: Classic approach, requires more labeled data, slower inference
- **YOLOv8-seg**: Faster but less accurate for complex scenes, weaker uncertainty metrics
- **Detectron2**: Excellent framework but SAM2 offers better foundation model advantages

**Implementation Approach**:
```python
# Use SAM2 via official repository
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load model
predictor = SAM2ImagePredictor(build_sam2("sam2_hiera_large.yaml", checkpoint))

# Uncertainty from logits
masks, scores, logits = predictor.predict(...)
uncertainty = 1 - scores  # or entropy from logits
```

**Key Resources**:
- Official repo: https://github.com/facebookresearch/segment-anything-2
- Paper: "SAM 2: Segment Anything in Images and Videos"
- Model zoo: Multiple checkpoint sizes (tiny, small, base, large)

**Best Practices**:
- Use `sam2_hiera_large` for best quality, `sam2_hiera_tiny` for speed
- Batch process frames for efficiency
- Cache embeddings to avoid recomputation
- Use automatic mask generation for discovery mode
- Fine-tune with LoRA adapters for memory efficiency

---

### 2. Vision-Language Model: GPT-5.2

**Decision**: Use GPT-5.2 via OpenAI API for uncertain region labeling

**Rationale**:
- **State-of-the-art**: Latest OpenAI model with superior vision capabilities
- **Reasoning**: Provides explanations alongside labels
- **API accessibility**: Well-documented REST API
- **Reliability**: Production-grade service with high uptime
- **Context understanding**: Handles complex scenes and edge cases

**Alternatives Considered**:
- **Claude Opus 4.5**: Excellent but user preferred GPT-5.2
- **LLaVA v1.6**: Open-source local option, lower accuracy than GPT-5.2
- **CogVLM**: Good performance but less refined API

**Implementation Approach**:
```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What object is in this image? Provide a specific label and reasoning."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    }],
    max_tokens=300
)
```

**Key Resources**:
- API docs: https://platform.openai.com/docs
- Vision guide: https://platform.openai.com/docs/guides/vision
- Rate limits: Monitor via headers, implement exponential backoff

**Best Practices**:
- Batch queries where possible to reduce latency
- Implement retry logic with exponential backoff
- Cache VLM responses for identical crops
- Use structured prompts: "Provide: 1) Object label 2) Confidence 3) Reasoning"
- Set reasonable `max_tokens` (200-300) to control costs
- Monitor token usage and implement budget alerts

---

### 3. Image Generation: Z-Image

**Decision**: Use Z-Image (Alibaba Tongyi-MAI) for synthetic training data

**Rationale**:
- **Open-source**: Apache 2.0 license, self-hostable
- **High quality**: Ranked #1 among open-source models
- **Efficient**: Fits in 16GB VRAM, sub-second inference with Turbo variant
- **Bilingual**: Strong English and Chinese text rendering
- **Multiple variants**: Base (fine-tuning), Turbo (speed), Edit (modifications)

**Alternatives Considered**:
- **Stable Diffusion XL**: Mature ecosystem but Z-Image offers better prompt adherence
- **Flux**: Initially considered, but Z-Image has better leaderboard rankings
- **DALL-E 3**: Highest quality but API-only with per-image costs

**Implementation Approach**:
```python
from diffusers import DiffusionPipeline
import torch

# Load Z-Image-Turbo for speed
pipeline = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipeline.to("cuda")

# Generate with prompt
images = pipeline(
    prompt=f"A photorealistic {label} in various lighting and backgrounds",
    num_inference_steps=8,  # Turbo uses fewer steps
    num_images_per_prompt=20
).images
```

**Key Resources**:
- GitHub: https://github.com/Tongyi-MAI/Z-Image
- Model Hub: Available on Hugging Face and ModelScope
- Paper: "Z-Image: Scalable Single-Stream DiT Architecture"

**Best Practices**:
- Use Z-Image-Turbo for speed (8 steps vs 50+ for base)
- Prompt engineering: Include "photorealistic", context, lighting variations
- Generate in batches to amortize model loading overhead
- Quality filtering: Use CLIP similarity scores to filter poor generations
- Vary prompts: "daytime/nighttime", "indoor/outdoor", "close-up/wide angle"

---

### 4. GUI Framework: Streamlit

**Decision**: Use Streamlit for web-based interactive GUI

**Rationale**:
- **Rapid development**: Build UIs with pure Python, no HTML/CSS/JS
- **ML-focused**: Built-in widgets for images, videos, plots
- **Real-time updates**: Native support for session state and reactive updates
- **Deployment**: Easy local or cloud deployment (Streamlit Cloud, Docker)
- **Community**: Large ecosystem of components and examples

**Alternatives Considered**:
- **Gradio**: Similar but less flexible for custom layouts
- **PyQt5/6**: More control but steeper learning curve, desktop-only
- **Tkinter**: Too limited for rich ML visualizations

**Implementation Approach**:
```python
import streamlit as st

st.set_page_config(layout="wide", page_title="Visual Perception Agent")

# Video upload
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# Multi-column layout for visualizations
col1, col2, col3 = st.columns(3)
with col1:
    st.image(frame, caption="Original Frame")
with col2:
    st.image(segmentation_overlay, caption="Segmentation")
with col3:
    st.image(uncertainty_map, caption="Uncertainty")

# Progress tracking
progress_bar = st.progress(0)
status_text = st.empty()
```

**Key Resources**:
- Docs: https://docs.streamlit.io
- Component gallery: https://streamlit.io/components
- Examples: https://streamlit.io/gallery

**Best Practices**:
- Use `st.session_state` for persistent data across interactions
- Implement caching with `@st.cache_data` for expensive operations
- Use `st.empty()` containers for dynamic updates
- Organize with tabs (`st.tabs`) for different pipeline stages
- Implement `st.spinner()` for long-running operations
- Use `st.columns()` for side-by-side visualizations

---

### 5. Uncertainty Detection Method

**Decision**: Use logit-based entropy for uncertainty quantification

**Rationale**:
- **Direct access**: SAM2 provides logits alongside predictions
- **Principled**: Entropy measures prediction uncertainty
- **Threshold-based**: Easy to configure and interpret
- **Fast**: No additional model required

**Implementation Approach**:
```python
import torch.nn.functional as F

def compute_uncertainty(logits):
    """Compute prediction uncertainty from logits"""
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    # Normalize entropy to [0, 1]
    max_entropy = torch.log(torch.tensor(probs.shape[-1]))
    normalized_uncertainty = entropy / max_entropy
    return normalized_uncertainty

# Flag uncertain regions
uncertain_mask = normalized_uncertainty > threshold  # default 0.25
```

**Best Practices**:
- Use entropy (full distribution) over confidence (max probability) for better uncertainty
- Calibrate threshold on validation set for target precision
- Visualize uncertainty heatmaps for interpretability
- Consider spatial smoothing to reduce noise
- Track uncertainty distribution across frames for anomaly detection

---

### 6. Incremental Training Strategy

**Decision**: Use LoRA (Low-Rank Adaptation) for efficient SAM2 fine-tuning

**Rationale**:
- **Memory efficient**: Only train small adapter weights, not full model
- **Fast training**: Fewer parameters to optimize
- **Preserves base knowledge**: Foundation model capabilities retained
- **Modular**: Can train/swap multiple LoRA adapters for different domains

**Implementation Approach**:
```python
from peft import LoraConfig, get_peft_model

# Configure LoRA
lora_config = LoraConfig(
    r=8,  # rank
    lora_alpha=32,
    target_modules=["qkv"],  # attention layers
    lora_dropout=0.1,
)

# Wrap SAM2 with LoRA
model = get_peft_model(sam2_model, lora_config)

# Train only LoRA weights
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Training loop with combined original + synthetic data
for batch in dataloader:
    masks_pred = model(batch["images"])
    loss = loss_fn(masks_pred, batch["masks"])
    loss.backward()
    optimizer.step()
```

**Best Practices**:
- Start with small LoRA rank (r=4 or 8), increase if underfitting
- Use learning rate warmup for stability
- Mix synthetic and original data in each batch (50/50 ratio)
- Validate on held-out original data to detect overfitting to synthetic
- Track metrics: IoU, precision, recall, uncertainty reduction
- Save LoRA checkpoints separately for easy versioning

---

### 7. Data Storage Strategy

**Decision**: File-based hierarchical storage with JSON metadata

**Rationale**:
- **Simplicity**: No database setup required
- **Transparency**: Easy to inspect and debug
- **Scalability**: Sufficient for target scale (100s of sessions)
- **Portability**: Easy to backup and transfer

**Structure**:
```text
data/
├── sessions/
│   └── {session_id}/
│       ├── metadata.json          # Video info, timestamps
│       ├── video.mp4                # Original video
│       ├── frames/                  # Extracted frames
│       │   ├── frame_0000.jpg
│       │   └── ...
│       ├── segmentations/           # Mask arrays
│       │   ├── frame_0000.npz
│       │   └── ...
│       ├── uncertainties/           # Uncertainty maps
│       │   └── ...
│       ├── vlm_queries/             # VLM results
│       │   └── region_*.json
│       └── synthetic/               # Generated images
│           └── ...
├── models/
│   ├── base/                        # Base SAM2 checkpoint
│   └── versions/                    # Fine-tuned versions
│       ├── v1/
│       │   ├── lora_weights.pt
│       │   └── metadata.json
│       └── ...
└── datasets/
    └── training_data.jsonl          # Training set manifest
```

**Best Practices**:
- Use UUIDs for session IDs to avoid collisions
- Compress masks with `.npz` format
- Store metadata as JSON for easy querying
- Implement cleanup policies for old sessions
- Use symlinks for synthetic images to avoid duplication
- Implement atomic writes (temp file + rename) for crash safety

---

### 8. Testing Strategy

**Decision**: Layered testing with pytest (unit → integration → end-to-end)

**Test Layers**:

1. **Unit Tests**: Test individual components in isolation
   - Mock external APIs (GPT-5.2, model inference)
   - Test pure functions (uncertainty calculation, mask processing)
   - Fast execution (< 1s per test)

2. **Integration Tests**: Test service interactions
   - Use small test videos and images
   - Test pipeline stages with real (small) models
   - Moderate execution (seconds per test)

3. **Contract Tests**: Verify API assumptions
   - Test GPT-5.2 API response structure
   - Validate model input/output shapes
   - Critical for external dependencies

4. **End-to-End Tests**: Full pipeline validation
   - Process sample video through complete cycle
   - Verify output quality metrics
   - Slow execution (minutes), run less frequently

**Best Practices**:
- Use `pytest-mock` for mocking external services
- Use `pytest-cov` for coverage tracking (target 80%+)
- Fixtures for test data (sample videos, images, mock responses)
- Parametrize tests for different scenarios
- Separate fast and slow tests with markers: `@pytest.mark.slow`
- Run fast tests in pre-commit hook, slow tests in CI

---

## Integration Patterns

### SAM2 + Uncertainty Detection Pipeline

```python
class SegmentationService:
    def __init__(self, model_path, confidence_threshold=0.75):
        self.predictor = SAM2ImagePredictor(...)
        self.threshold = confidence_threshold

    def segment_frame(self, frame):
        # Inference
        masks, scores, logits = self.predictor.predict(frame)

        # Uncertainty detection
        uncertainty = self.compute_uncertainty(logits)
        uncertain_regions = self.extract_uncertain_regions(
            masks, uncertainty, self.threshold
        )

        return {
            "masks": masks,
            "scores": scores,
            "uncertain_regions": uncertain_regions
        }
```

### VLM Query with Retry Logic

```python
class VLMService:
    def __init__(self, api_key, max_retries=3):
        self.client = OpenAI(api_key=api_key)
        self.max_retries = max_retries

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def label_region(self, image_crop):
        response = self.client.chat.completions.create(...)
        return self.parse_response(response)
```

### Synthetic Data Generation Pipeline

```python
class GenerationService:
    def generate_training_data(self, label, num_images=20):
        # Generate variations
        prompts = self.create_prompt_variations(label)
        images = self.pipeline(prompts, num_images_per_prompt=num_images)

        # Re-segment for masks
        segmented = [self.segmentation_service.segment(img) for img in images]

        # Quality filter
        filtered = self.quality_filter(images, segmented)

        return filtered
```

---

## Performance Optimization

### GPU Memory Management
- Use mixed precision (fp16) for inference
- Clear CUDA cache between stages
- Process frames in batches (4-8 frames)
- Offload inactive models to CPU

### Latency Reduction
- Preload models at startup
- Cache video frames in memory
- Use async/await for I/O operations
- Prefetch next frame during processing

### API Cost Management
- Batch VLM queries where possible
- Cache identical crops (use perceptual hash)
- Implement query budgets and alerts
- Use cheaper models for low-priority queries

---

## Security Considerations

1. **API Key Management**: Use environment variables, never commit keys
2. **Input Validation**: Check video format, size, frame count before processing
3. **Resource Limits**: Cap video length, frame count, storage per session
4. **Error Handling**: Sanitize error messages, avoid leaking system info
5. **Rate Limiting**: Implement per-user limits for API calls

---

## Monitoring and Observability

**Key Metrics**:
- Processing latency per frame
- Uncertainty region count per video
- VLM query success rate
- Synthetic image quality scores
- Model improvement over iterations (IoU, precision, recall)
- GPU memory usage
- API costs

**Implementation**:
```python
import logging
from prometheus_client import Counter, Histogram

# Metrics
frame_processing_time = Histogram('frame_processing_seconds', 'Frame processing time')
uncertain_regions_count = Counter('uncertain_regions_total', 'Total uncertain regions detected')
vlm_queries = Counter('vlm_queries_total', 'Total VLM queries', ['status'])

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@frame_processing_time.time()
def process_frame(frame):
    logger.info(f"Processing frame {frame_id}")
    # ... processing
    uncertain_regions_count.inc(len(uncertain_regions))
```

---

## Deployment Considerations

**Local Development**:
```bash
# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download_models.py  # SAM2, Z-Image weights

# Set API keys
export OPENAI_API_KEY="sk-..."

# Run Streamlit app
streamlit run src/gui/app.py
```

**Docker Deployment**:
```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu22.04
RUN pip install torch torchvision sam2 z-image streamlit openai
COPY src/ /app/src/
CMD ["streamlit", "run", "/app/src/gui/app.py"]
```

**Resource Requirements**:
- GPU: NVIDIA with 16GB+ VRAM (RTX 4080, A100, etc.)
- CPU: 8+ cores recommended
- RAM: 32GB+ recommended
- Storage: 100GB+ for models and data

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| API rate limits | Implement exponential backoff, caching, budget alerts |
| GPU OOM | Use fp16, batch size tuning, clear cache between stages |
| Poor synthetic quality | Quality filtering with CLIP scores, manual review UI |
| Model degradation | Validate on held-out data, rollback capability |
| Long processing time | Progress indicators, pause/resume, async processing |

---

## Next Steps

1. ✅ Research complete - All technology choices validated
2. ➡️ Phase 1: Define data models (`data-model.md`)
3. ➡️ Phase 1: Create API contracts (`contracts/`)
4. ➡️ Phase 1: Write quickstart guide (`quickstart.md`)
5. ➡️ Phase 2: Generate task breakdown (`tasks.md`)
