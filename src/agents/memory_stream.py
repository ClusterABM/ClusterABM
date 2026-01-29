"""
Complete memory stream implementation for generative agents.
Based on "Generative Agents" (Park et al., 2023).
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from openai import OpenAI
import os
import json


@dataclass
class MemoryNode:
    """A single memory node in the memory stream."""
    
    description: str
    timestamp: datetime
    node_id: int
    importance: float = 5.0
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    embedding: Optional[np.ndarray] = None
    memory_type: str = "observation"  # observation, reflection, plan
    related_nodes: List[int] = field(default_factory=list)
    
    def recency_score(self, current_time: datetime) -> float:
        """Exponential decay based on time."""
        hours_passed = (current_time - self.timestamp).total_seconds() / 3600
        return 0.995 ** hours_passed
    
    def importance_score(self) -> float:
        """Normalized importance."""
        return self.importance / 10.0
    
    def relevance_score(self, query_embedding: np.ndarray) -> float:
        """Cosine similarity."""
        if self.embedding is None:
            return 0.0
        dot = np.dot(self.embedding, query_embedding)
        norm = np.linalg.norm(self.embedding) * np.linalg.norm(query_embedding)
        return dot / norm if norm > 0 else 0.0
    
    def retrieval_score(
        self, 
        query_embedding: np.ndarray, 
        current_time: datetime,
        alpha: float = 1.0, 
        beta: float = 1.0, 
        gamma: float = 1.0
    ) -> float:
        """Combined retrieval score."""
        recency = self.recency_score(current_time)
        importance = self.importance_score()
        relevance = self.relevance_score(query_embedding)
        
        self.access_count += 1
        self.last_accessed = current_time
        
        return alpha * recency + beta * importance + gamma * relevance


class MemoryStream:
    """Memory stream with retrieval and reflection."""
    
    def __init__(self, agent_id: int, agent_name: str, llm_client: OpenAI = None):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.llm_client = llm_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.memory_nodes: List[MemoryNode] = []
        self.next_node_id = 0
        self.importance_accumulator = 0.0
        self.reflection_threshold = 150.0
    
    def add(self, description: str, importance: Optional[float] = None, memory_type: str = "observation") -> MemoryNode:
        """Add memory with auto-importance."""
        if importance is None:
            importance = self._compute_importance(description)
        
        embedding = self._embed(description)
        
        node = MemoryNode(
            description=description,
            timestamp=datetime.now(),
            node_id=self.next_node_id,
            importance=importance,
            embedding=embedding,
            memory_type=memory_type
        )
        
        self.memory_nodes.append(node)
        self.next_node_id += 1
        self.importance_accumulator += importance
        
        if self.importance_accumulator >= self.reflection_threshold:
            self._reflect()
        
        return node
    
    def retrieve(self, query: str, n: int = 5) -> List[MemoryNode]:
        """Retrieve top-n relevant memories."""
        if not self.memory_nodes:
            return []
        
        query_emb = self._embed(query)
        current_time = datetime.now()
        
        scored = [(node.retrieval_score(query_emb, current_time), node) for node in self.memory_nodes]
        scored.sort(reverse=True, key=lambda x: x[0])
        
        return [node for _, node in scored[:n]]
    
    def get_recent_context(self, n: int = 10) -> str:
        """Get recent memories as text."""
        recent = sorted(self.memory_nodes, key=lambda x: x.timestamp, reverse=True)[:n]
        if not recent:
            return "No recent memories."
        
        return "\n".join([f"- {node.description}" for node in recent])
    
    def _reflect(self):
        """Generate reflections from memories."""
        recent = sorted(
            [n for n in self.memory_nodes if n.memory_type == "observation"],
            key=lambda x: x.importance,
            reverse=True
        )[:10]
        
        if len(recent) < 3:
            return
        
        memories_text = "\n".join([f"- {n.description}" for n in recent])
        
        prompt = f"""Based on these observations about {self.agent_name}, generate 2-3 high-level insights.

OBSERVATIONS:
{memories_text}

Generate insights that identify patterns, draw conclusions, or inform future decisions.
Output 2-3 insights, one per line:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Generate insightful reflections."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            insights = [line.strip("- ").strip() for line in response.choices[0].message.content.split("\n") if line.strip()]
            
            for insight in insights:
                self.add(insight, importance=8.0, memory_type="reflection")
            
            self.importance_accumulator = 0.0
        
        except Exception as e:
            print(f"Reflection failed: {e}")
    
    def _compute_importance(self, description: str) -> float:
        """Rate importance 1-10."""
        prompt = f"""Rate the importance (1-10) of this observation for decision-making:
"{description}"

Consider health impact, goal relevance, and novelty.
Output only a number 1-10:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "Rate importance."}, {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=10
            )
            
            import re
            match = re.search(r'\d+', response.choices[0].message.content)
            return min(max(float(match.group()), 1.0), 10.0) if match else 5.0
        except:
            return 5.0
    
    def _embed(self, text: str) -> np.ndarray:
        """Create embedding."""
        try:
            response = self.llm_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding)
        except:
            return np.random.randn(1536)
