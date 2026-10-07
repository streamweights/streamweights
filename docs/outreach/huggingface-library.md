# Hugging Face registry pull request (draft, not posted)

Nothing here has been submitted. This is the draft of a pull request to https://github.com/huggingface/huggingface.js that shows streamweights on model pages. It should wait for the verification batch in [plan.md](../plan.md) (the CUDA gates and the proof run) and for the PyPI release, so the install line in the snippet can be `pip install streamweights`.

## Which registry, and why

huggingface.js has two registries that show a tool on a model page, both in `packages/tasks/src/`:

| file | what it does | fits streamweights? |
|---|---|---|
| `model-libraries.ts` (with snippets in `model-libraries-snippets.ts`) | a library whose models carry `library_name: <key>` on the Hub, shown as a tag and a "Use this model" button | no: spill publishes no models with a `library_name` of its own, and it runs models that other libraries own |
| `local-apps.ts` (`LOCAL_APPS`) | an app that appears in the "Use this model" dropdown on any compatible model page, with a copy-paste snippet | **yes**: `spill run org/name file.jsonl` takes any Hugging Face repo id of a supported architecture |

So the change is one entry in `LOCAL_APPS` and one snippet function in `packages/tasks/src/local-apps.ts`. Nothing else changes.

## Compatibility rule

streamweights streams bf16 safetensors of the dense decoder families it has verified: `llama` (also Mistral), `qwen2`, `qwen3`, `phi3` and `gemma2` ([models](../models.md)). It does not run quantized repos, GGUF-only repos or mixture-of-experts. The button shows on a model page when the model is a text-generation model with safetensors weights, a verified `model_type`, and no `quantization_config`.

## The change

In `packages/tasks/src/local-apps.ts`, next to the other helpers:

```ts
const STREAMWEIGHTS_MODEL_TYPES = ["llama", "mistral", "qwen2", "qwen3", "phi3", "gemma2"];

function isStreamweightsModel(model: ModelData): boolean {
	return (
		model.pipeline_tag === "text-generation" &&
		model.tags.includes("safetensors") &&
		STREAMWEIGHTS_MODEL_TYPES.includes(model.config?.model_type ?? "") &&
		!model.config?.quantization_config
	);
}

const snippetStreamweights = (model: ModelData): LocalAppSnippet[] => [
	{
		title: "Install",
		setup: "pip install streamweights",
		content: "spill doctor",
	},
	{
		title: "Run a file of prompts through the full-precision model (streamed from disk when it is bigger than RAM)",
		content: `spill run ${model.id} prompts.jsonl --out answers.jsonl`,
	},
	{
		title: "Fine-tune a LoRA adapter on it and grade it",
		content: [
			`spill tune ${model.id} train.jsonl --name mine`,
			`spill eval evals.jsonl ${model.id} ${model.id}+mine`,
		],
	},
];
```

and in the `LOCAL_APPS` object, after `"mlx-lm"`:

```ts
	streamweights: {
		prettyLabel: "streamweights",
		docsUrl: "https://streamweights.github.io/streamweights/",
		mainTask: "text-generation",
		displayOnModelPage: isStreamweightsModel,
		snippet: snippetStreamweights,
	},
```

`packages/tasks/src/local-apps.spec.ts` gets one test next to the existing ones: a Qwen2 safetensors model shows the app, a model with a `quantization_config` or a GGUF-only repo does not.

## Pull request text

**Title:** Add streamweights to local apps

**Body:**

> streamweights (`spill`) fine-tunes, distills and evaluates LLMs locally on a Mac (MLX) or Linux (PyTorch), and streams full-precision safetensors from disk so a model bigger than RAM still runs. This adds it to the "Use this model" dropdown for text-generation safetensors models of the families it supports (llama, mistral, qwen2, qwen3, phi3, gemma2), with install, run and tune snippets.
>
> - Docs: https://streamweights.github.io/streamweights/
> - Repo (Apache-2.0): https://github.com/streamweights/streamweights
> - Measured: Llama 3.3 70B, 141 GB unquantized, run on a 48 GB MacBook Pro: https://streamweights.github.io/streamweights/reports/002-phase1/
>
> The compatibility check excludes quantized and GGUF repos, which streamweights does not run. Adds a test for both cases.

## Before opening it

1. Run the verification batch and the PyPI release, then change the install line and re-read the three commands against `spill --help`.
2. Run `pnpm install && pnpm --filter @huggingface/tasks test` in a fork of huggingface.js.
3. Open the pull request from the maintainer's own account. This draft was not posted.
