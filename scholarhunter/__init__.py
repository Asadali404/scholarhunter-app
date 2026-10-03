"""scholarhunter — scholarship matching pipeline.

Public surface (see blueprint §3):
  * schemas.UserProfile / Scholarship / ScholarshipMatch
  * tools.*                 deterministic search / pre-filter primitives
  * llm_client.LLMClient    stdlib-only Groq/OpenAI chat client
  * crew.run_pipeline       end-to-end pipeline (Researcher → Searcher
                            → PreFilter → Matchmaker), with offline demo path

Everything degrades gracefully when GROQ_API_KEY is unset — the "cached
demo mode" of blueprint §2.
"""
from . import schemas, tools, llm_client, crew  # noqa: F401
from .crew import run_pipeline, PipelineResult

__all__ = ["schemas", "tools", "llm_client", "crew", "run_pipeline", "PipelineResult"]
__version__ = "0.1.0"
