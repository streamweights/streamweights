Resume fixture: qwen2.5:0.5b, toy task, rank 4 on q_proj and v_proj, lr 3e-05, 100-step
cosine schedule, stopped at step 50 on MLX (bf16 base weights, Apple GPU).
state/ is the portable checkpoint; job.json is the invocation; expected_losses.json is the
uninterrupted MLX run.
  cd <this directory> && spill tune --config job.json --engine torch-cpu
continues it from step 51 on any machine.
