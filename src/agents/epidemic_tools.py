"""
Domain-specific tools for epidemic simulation agents.
Updated to use real COVID-19 knowledge base with persistent storage.
"""

from pathlib import Path
from typing import Optional, Dict, List
import numpy as np


class EpidemicTool:
    """Base class for epidemic-specific tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, agent, **kwargs) -> str:
        """Execute tool with agent context."""
        raise NotImplementedError


class QueryTransmissionRulesTool(EpidemicTool):
    """Query RAG database for transmission dynamics."""
    
    def __init__(self, kb_path: Optional[Path] = None):
        super().__init__(
            name="query_transmission_rules",
            description="Query epidemic knowledge base for transmission dynamics, risk factors, and intervention effectiveness"
        )
        
        # Set default KB path
        if kb_path is None:
            kb_path = Path("data/knowledge_base")
        
        self.kb_path = kb_path
        self.kb = None
        
        # Try to load knowledge base
        try:
            # Check if KB exists
            if kb_path.exists() and (kb_path / "chroma.sqlite3").exists():
                print(f"  ✓ Loading RAG knowledge base from {kb_path}")
                from src.knowledge.epidemic_kb import EpidemicKnowledgeBase
                self.kb = EpidemicKnowledgeBase(
                    use_real_data=False,  # Don't re-fetch, use existing
                    persist_directory=str(kb_path)
                )
                print(f"  ✓ Knowledge base loaded: {self.kb.collection.count()} documents")
            else:
                print(f"  ⚠ Knowledge base not found at {kb_path}")
                print(f"    Creating knowledge base with fallback data...")
                # Create KB directory
                kb_path.mkdir(parents=True, exist_ok=True)
                # Initialize with fallback
                from src.agents.epidemic_knowledge import EpidemicKnowledgeBase
                self.kb = EpidemicKnowledgeBase(
                    use_real_data=False,  # Use fallback
                    persist_directory=str(kb_path)
                )
                print(f"  ✓ Fallback knowledge base created")
        
        except Exception as e:
            print(f"  ✗ Error loading knowledge base: {e}")
            print(f"  Using inline fallback data")
            self.kb = None
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize minimal fallback knowledge (in-memory only)."""
        import chromadb
        from chromadb.config import Settings
        
        self.client = chromadb.Client(Settings(anonymized_telemetry=False, allow_reset=True))
        self.collection = self.client.get_or_create_collection("epidemic_fallback")
        
        if self.collection.count() > 0:
            return
        
        fallback_items = [
            {
                "id": "fallback_household",
                "text": """HOUSEHOLD TRANSMISSION:
- Secondary Attack Rate: 30-40%
- Daily probability while infectious: 15-25%
- Isolation reduces risk by 60-80%
- Shared bedroom: 40-50% SAR
- Separate rooms: 25-35% SAR"""
            },
            {
                "id": "fallback_vaccination",
                "text": """VACCINATION EFFECTIVENESS:
- Fully vaccinated: 70-85% infection reduction
- Partially vaccinated: 40-60% reduction
- Protection against severe disease: 90-95%
- Waning: drops to 50-60% after 6 months"""
            },
            {
                "id": "fallback_age_risk",
                "text": """AGE-BASED RISK:
- Children (0-12): Lower susceptibility (~60% of adults)
- Teens (13-17): Similar to young adults
- Adults (18-64): Baseline risk
- Elderly (65+): 2-3x more susceptible, higher severity"""
            }
        ]
        
        for item in fallback_items:
            self.collection.add(
                documents=[item["text"]],
                ids=[item["id"]]
            )
    
    def execute(self, agent, query: str, n_results: int = 2) -> str:
        """Query knowledge base."""
        try:
            # Use real KB if available
            if self.kb and hasattr(self.kb, 'query'):
                results = self.kb.query(query, n_results=n_results)
                
                if not results:
                    return "No relevant transmission rules found in knowledge base."
                
                output = "=== EPIDEMIC KNOWLEDGE BASE ===\n"
                for i, result in enumerate(results, 1):
                    source = result.get('metadata', {}).get('source', 'Unknown')
                    content = result.get('content', '')
                    # Truncate for readability
                    content_preview = content[:400] + "..." if len(content) > 400 else content
                    output += f"\n[Source {i}: {source}]\n{content_preview}\n"
                
                return output
            
            # Fallback to in-memory collection
            elif hasattr(self, 'collection'):
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
                
                if not results['documents'][0]:
                    return "No relevant transmission rules found."
                
                output = "=== EPIDEMIC KNOWLEDGE (Fallback) ===\n"
                for i, doc in enumerate(results['documents'][0], 1):
                    doc_preview = doc[:400] + "..." if len(doc) > 400 else doc
                    output += f"\n[Source {i}]\n{doc_preview}\n"
                
                return output
            
            else:
                return "Knowledge base unavailable. Use general COVID-19 precautions."
        
        except Exception as e:
            return f"Error querying knowledge base: {str(e)[:100]}"


class CheckNeighborStatesTool(EpidemicTool):
    """Check infection states of neighboring agents."""
    
    def __init__(self):
        super().__init__(
            name="check_neighbor_states",
            description="Check the infection states (S/E/I/R/D) of agents you're connected to"
        )
    
    def execute(self, agent, **kwargs) -> str:
        """Get neighbor states."""
        if not hasattr(agent, 'neighbors') or not agent.neighbors:
            return "No neighbors found."
        
        neighbor_info = []
        state_counts = {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0}
        household_infected = 0
        total_neighbors = len(agent.neighbors)
        
        for neighbor in agent.neighbors:
            state = neighbor.state
            state_counts[state] = state_counts.get(state, 0) + 1
            
            is_household = (neighbor.profile.get('household_id') == 
                          agent.profile.get('household_id'))
            
            relationship = "HOUSEHOLD" if is_household else self._get_edge_type(agent, neighbor)
            
            neighbor_info.append(
                f"- {neighbor.profile['name']}: {state} ({relationship})"
            )
            
            if state in ['E', 'I'] and is_household:
                household_infected += 1
        
        output = f"=== YOUR CONTACTS ({total_neighbors} total) ===\n\n"
        output += f"State distribution: S={state_counts['S']}, E={state_counts['E']}, I={state_counts['I']}, R={state_counts['R']}, D={state_counts['D']}\n"
        output += f"HOUSEHOLD INFECTED: {household_infected}\n\n"
        output += "\n".join(neighbor_info[:10])  # Show first 10
        
        if len(neighbor_info) > 10:
            output += f"\n... and {len(neighbor_info) - 10} more contacts"
        
        return output
    
    def _get_edge_type(self, agent1, agent2) -> str:
        """Determine relationship type."""
        if agent1.profile.get('household_id') == agent2.profile.get('household_id'):
            return "household"
        
        occ1 = agent1.profile.get('occupation', '')
        occ2 = agent2.profile.get('occupation', '')
        
        if occ1 == occ2 and occ1 in ['healthcare_worker', 'office_worker', 'retail_worker', 'teacher']:
            return "work"
        
        if occ1 == 'student' and occ2 == 'student':
            age_diff = abs(agent1.profile.get('age', 0) - agent2.profile.get('age', 0))
            if age_diff <= 3:
                return "school"
        
        if occ1 in ['student', 'teacher'] and occ2 in ['student', 'teacher']:
            return "school"
        
        return "social"


class CalculateExposureRiskTool(EpidemicTool):
    """Calculate infection risk based on contacts."""
    
    def __init__(self):
        super().__init__(
            name="calculate_exposure_risk",
            description="Calculate your infection risk based on infected contacts and protection measures"
        )
    
    def execute(self, agent, **kwargs) -> str:
        """Calculate risk."""
        if agent.state not in ['S', 'R']:
            return f"You are currently {agent.state}. This tool is for susceptible/recovered agents."
        
        infected_neighbors = [n for n in agent.neighbors if n.state in ['E', 'I']]
        
        if not infected_neighbors:
            return "No infected contacts detected. Your exposure risk is minimal (baseline community transmission only)."
        
        # Calculate risk per contact
        risk_breakdown = []
        total_daily_risk = 0.0
        
        for neighbor in infected_neighbors:
            edge_type = self._get_edge_type(agent, neighbor)
            
            # Base rate
            beta = 0.05
            
            # Contact type multiplier
            multipliers = {
                'household': 3.0,
                'work': 1.8 if agent.profile.get('occupation') == 'healthcare_worker' else 1.5,
                'school': 1.8,
                'social': 0.8
            }
            beta *= multipliers.get(edge_type, 1.0)
            
            # Vaccination protection
            vacc_protection = {0: 0.0, 1: 0.5, 2: 0.8}
            beta *= (1 - vacc_protection.get(agent.profile.get('vaccination_status', 0), 0))
            
            # Mask compliance
            beta *= (1 - agent.profile.get('compliance_score', 0.7) * 0.5)
            
            # Age susceptibility
            age = agent.profile.get('age', 40)
            if age < 18:
                beta *= 0.6
            elif age > 60:
                beta *= 1.3
            
            # Comorbidities
            beta *= (1 + agent.profile.get('comorbidity_count', 0) * 0.15)
            
            contact_risk = beta * 100  # Convert to percentage
            total_daily_risk += beta
            
            risk_breakdown.append(
                f"- {neighbor.profile['name']} ({edge_type}): {contact_risk:.1f}% per day"
            )
        
        # Combined probability
        combined_prob = (1 - (1 - total_daily_risk)) * 100
        
        output = f"=== EXPOSURE RISK CALCULATION ===\n\n"
        output += f"Infected contacts: {len(infected_neighbors)}\n\n"
        output += "Risk per contact:\n" + "\n".join(risk_breakdown[:5])  # Show first 5
        output += f"\n\nCOMBINED DAILY PROBABILITY: {combined_prob:.1f}%\n"
        
        return output
    
    def _get_edge_type(self, agent1, agent2) -> str:
        """Determine relationship type."""
        if agent1.profile.get('household_id') == agent2.profile.get('household_id'):
            return "household"
        
        occ1 = agent1.profile.get('occupation', '')
        occ2 = agent2.profile.get('occupation', '')
        
        if occ1 == occ2 and occ1 in ['healthcare_worker', 'office_worker', 'retail_worker', 'teacher']:
            return "work"
        
        if occ1 == 'student' and occ2 == 'student':
            return "school"
        
        if occ1 in ['student', 'teacher'] and occ2 in ['student', 'teacher']:
            return "school"
        
        return "social"


class CheckDiseaseProgressionTool(EpidemicTool):
    """Check disease progression and recovery timeline."""
    
    def __init__(self):
        super().__init__(
            name="check_disease_progression",
            description="Check how long you've been in current state and estimate recovery timeline"
        )
    
    def execute(self, agent, **kwargs) -> str:
        """Check progression."""
        days_in_state = getattr(agent, 'days_in_state', 0)
        
        if agent.state == 'S':
            return f"You are SUSCEPTIBLE. You have not been infected."
        
        elif agent.state == 'E':
            output = f"=== EXPOSURE STATUS ===\n\n"
            output += f"Days since exposure: {days_in_state}\n\n"
            output += "You are EXPOSED but not yet infectious.\n"
            output += "Typical incubation: 2-5 days\n"
            
            if days_in_state < 3:
                output += "Stage: Early incubation\n"
                output += "- Not yet infectious\n"
                output += "- Progression to infectious likely in 1-3 days\n"
            else:
                output += "Stage: Late incubation\n"
                output += "- May become infectious soon\n"
                output += "- Progression to infectious highly likely\n"
            
            return output
        
        elif agent.state == 'I':
            output = f"=== INFECTION STATUS ===\n\n"
            output += f"Days infected: {days_in_state}\n\n"
            
            if days_in_state < 5:
                output += "Stage: EARLY INFECTION\n"
                output += "- Still in early infectious period\n"
                output += "- Symptoms may be developing or mild\n"
                output += "- Recovery unlikely yet (typical: 7-14 days total)\n"
            elif days_in_state < 10:
                output += "Stage: ACTIVE INFECTION\n"
                output += "- Peak infectious period\n"
                output += "- Symptoms typically most severe now\n"
                output += "- Recovery becoming more likely\n"
            else:
                output += "Stage: PROLONGED INFECTION\n"
                output += "- Overdue for recovery (typical: 7-14 days)\n"
                output += "- Recovery highly likely soon\n"
            
            # Estimate recovery probability
            base_recovery = 0.10
            if days_in_state < 5:
                recovery_rate = base_recovery * 0.3
            elif days_in_state < 10:
                recovery_rate = base_recovery * 1.0
            else:
                recovery_rate = base_recovery * 2.0
            
            # Adjust for factors
            vacc = agent.profile.get('vaccination_status', 0)
            if vacc == 2:
                recovery_rate *= 1.3
            elif vacc == 1:
                recovery_rate *= 1.15
            
            age = agent.profile.get('age', 40)
            if age < 18:
                recovery_rate *= 1.3
            elif age > 60:
                recovery_rate *= 0.7
            
            recovery_rate *= (0.85 ** agent.profile.get('comorbidity_count', 0))
            
            output += f"\nEstimated recovery probability today: {recovery_rate*100:.1f}%\n"
            
            return output
        
        elif agent.state == 'R':
            output = f"=== RECOVERY STATUS ===\n\n"
            output += f"Days since recovery: {days_in_state}\n\n"
            
            if days_in_state < 90:
                output += "Immunity: STRONG (95%+ protection)\n"
                output += "- Recent infection provides robust protection\n"
                output += "- Reinfection risk very low\n"
            elif days_in_state < 180:
                output += "Immunity: WANING (80% protection)\n"
                output += "- Protection declining but still substantial\n"
                output += "- Reinfection risk moderate\n"
            else:
                output += "Immunity: WEAK (50% protection)\n"
                output += "- Significant waning after 6+ months\n"
                output += "- Reinfection risk elevated\n"
            
            return output
        
        elif agent.state == 'D':
            return "You are deceased. No disease progression."
        
        return "Unknown state."


class GetActivityScheduleTool(EpidemicTool):
    """Get planned activities for current day."""
    
    def __init__(self):
        super().__init__(
            name="get_activity_schedule",
            description="Get your planned activities for today based on your occupation and schedule"
        )
        
        self.activity_templates = self._init_templates()
    
    def _init_templates(self) -> Dict:
        """Initialize activity templates by occupation."""
        return {
            'healthcare_worker': [
                {'type': 'work', 'location': 'hospital', 'duration_hours': 10, 'contact_level': 'HIGH', 'required': True},
                {'type': 'errands', 'location': 'grocery store', 'duration_hours': 1, 'contact_level': 'MODERATE', 'required': False}
            ],
            'office_worker': [
                {'type': 'work', 'location': 'office', 'duration_hours': 8, 'contact_level': 'MODERATE', 'required': True, 'remote_possible': True},
                {'type': 'errands', 'location': 'retail', 'duration_hours': 1, 'contact_level': 'MODERATE', 'required': False}
            ],
            'retail_worker': [
                {'type': 'work', 'location': 'retail store', 'duration_hours': 8, 'contact_level': 'HIGH', 'required': True},
                {'type': 'social', 'location': 'friend/family', 'duration_hours': 2, 'contact_level': 'MODERATE', 'required': False}
            ],
            'teacher': [
                {'type': 'work', 'location': 'school', 'duration_hours': 8, 'contact_level': 'HIGH', 'required': True},
                {'type': 'errands', 'location': 'retail', 'duration_hours': 1, 'contact_level': 'MODERATE', 'required': False}
            ],
            'student': [
                {'type': 'school', 'location': 'school', 'duration_hours': 7, 'contact_level': 'HIGH', 'required': True},
                {'type': 'social', 'location': 'friends', 'duration_hours': 3, 'contact_level': 'HIGH', 'required': False}
            ],
            'retired': [
                {'type': 'errands', 'location': 'retail', 'duration_hours': 2, 'contact_level': 'LOW', 'required': False},
                {'type': 'social', 'location': 'family visit', 'duration_hours': 3, 'contact_level': 'MODERATE', 'required': False}
            ]
        }
    
    def execute(self, agent, day_of_week: int = 0, **kwargs) -> str:
        """Get today's schedule."""
        occupation = agent.profile.get('occupation', 'office_worker')
        activities = self.activity_templates.get(occupation, [])
        
        # Filter weekday/weekend
        filtered_activities = []
        for activity in activities:
            if activity['type'] in ['work', 'school'] and day_of_week >= 5:
                continue  # Weekend, skip work/school
            filtered_activities.append(activity)
        
        if not filtered_activities:
            return "No scheduled activities today (weekend/rest day)."
        
        output = f"=== TODAY'S SCHEDULE (Day {day_of_week % 7}) ===\n\n"
        for i, activity in enumerate(filtered_activities, 1):
            output += f"{i}. {activity['type'].upper()}\n"
            output += f"   Location: {activity['location']}\n"
            output += f"   Duration: {activity['duration_hours']} hours\n"
            output += f"   Contact level: {activity['contact_level']}\n"
            if activity.get('remote_possible'):
                output += f"   Remote option: Available\n"
            output += "\n"
        
        return output


class EpidemicToolRegistry:
    """Registry of epidemic-specific tools with knowledge base integration."""
    
    def __init__(self, kb_path: Optional[Path] = None):
        """
        Initialize tool registry.
        
        Args:
            kb_path: Path to knowledge base directory (default: data/knowledge_base)
                    If KB exists at this path, it will be loaded
                    If not, fallback data will be used
        """
        self.kb_path = kb_path if kb_path else Path("data/knowledge_base")
        self.tools = {}
        self._register_tools()
    
    def _register_tools(self):
        """Register all epidemic tools."""
        tools = [
            QueryTransmissionRulesTool(kb_path=self.kb_path),  # ← Pass KB path
            CheckNeighborStatesTool(),
            CalculateExposureRiskTool(),
            CheckDiseaseProgressionTool(),
            GetActivityScheduleTool()
        ]
        
        for tool in tools:
            self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[EpidemicTool]:
        """Get tool by name."""
        return self.tools.get(name)
    
    def get_tool_descriptions(self) -> str:
        """Get all tool descriptions."""
        desc = "=== AVAILABLE TOOLS ===\n\n"
        for name, tool in self.tools.items():
            desc += f"• {name}\n  {tool.description}\n\n"
        return desc
    
    def execute_tool(self, tool_name: str, agent, **kwargs) -> str:
        """Execute a tool."""
        tool = self.get_tool(tool_name)
        if tool:
            return tool.execute(agent, **kwargs)
        return f"ERROR: Tool '{tool_name}' not found"
