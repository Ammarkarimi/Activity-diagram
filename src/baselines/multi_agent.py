from src.pipeline.orchestrator import MultiAgentPipeline


def run(requirement_text: str, max_iterations: int = 3, model: str | None = None):
    return MultiAgentPipeline(model=model).run(
        sample_id="baseline_multi_agent",
        requirement_text=requirement_text,
        max_iterations=max_iterations,
    )
