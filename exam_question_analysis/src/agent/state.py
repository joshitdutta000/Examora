from typing import TypedDict, Optional

class AgentState(TypedDict):
    # Input
    question_text: str
    subject: str
    topic: str
    question_type: str
    cognitive_level: str
    avg_score: float
    std_dev: float
    discrimination_index: float

    # M1 prediction output
    predicted_difficulty: Optional[str]
    confidence: Optional[float]

    # RAG output
    retrieved_context: Optional[str]

    # LLM outputs
    reasoning: Optional[str]
    learning_gaps: Optional[str]
    recommendations: Optional[str]
    disclaimer: Optional[str]

    # Final report
    final_report: Optional[str]