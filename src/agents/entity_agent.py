"""
LLM-based entity agent with memory, tools, planning, reasoning.


DUAL MODE SUPPORT:
1. trace_collection: Lightweight for pre-simulation clustering
2. full: Complete hybrid reasoning for main simulation

SEIRD MODEL with Death:
- S→E: Exposure
- E→I: Disease progression
- I→R: Recovery
- I→D: Death (age-stratified IFR, cluster + individual)
- R→S: Reinfection

FIXED ISSUES (Dec 2025):
- JSON serialization (numpy bool → Python bool)
- Initial stagnation (days 0-4) → proper temporal variation
- Vaccination effectiveness (was 0%, now 50-70%)
- Attack rate calibration (was 100%, now 40-60%)
- Zero temporal variance → added stochasticity
- Competing risk normalization  (no normalization, raw hazards)

"""

from typing import List, Dict, Optional, Tuple
from openai import OpenAI
import os
from datetime import datetime
import json
import re
import numpy as np
import time
from collections import deque

from src.agents.memory_stream import MemoryStream
from src.agents.epidemic_tools import EpidemicToolRegistry
from src.config.epidemic_params import EpidemicParams

def _compress(text: str, n: int = 180) -> str:
    return text.replace("\n", " ")[:n]

class SimpleRateLimiter:
    """
    Global rate limiter to prevent hitting OpenAI rate limits.
    
    Limits API calls to max_calls_per_minute to stay under OpenAI's limits.
    """
    
    def __init__(self, max_calls_per_minute=450):
        """
        Args:
            max_calls_per_minute: Maximum API calls allowed per minute
                                  (set to 450 to stay under OpenAI's 500/min limit)
        """
        self.max_calls = max_calls_per_minute
        self.calls = deque()  # Timestamps of recent calls
    
    def wait_if_needed(self):
        """Wait if we're approaching the rate limit."""
        now = time.time()
        
        # Remove calls older than 60 seconds
        while self.calls and now - self.calls[0] > 60:
            self.calls.popleft()
        
        # If we're at the limit, wait until the oldest call is >60s old
        if len(self.calls) >= self.max_calls:
            sleep_time = 60 - (now - self.calls[0]) + 1
            if sleep_time > 0:
                print(f"        [Rate limit] Sleeping {sleep_time:.0f}s to avoid quota...", flush=True)
                time.sleep(sleep_time)
                
                # Clean up old calls after sleeping
                now = time.time()
                while self.calls and now - self.calls[0] > 60:
                    self.calls.popleft()
        
        # Record this call
        self.calls.append(now)

# Create ONE global rate limiter instance
_llm_rate_limiter = SimpleRateLimiter(max_calls_per_minute=450)

class EntityAgent:
    """
    Full LLM-based entity agent representing a person in epidemic simulation.
    
    Capabilities:
    - Memory: Stores and retrieves experiences
    - Tools: Uses domain-specific epidemic tools
    - Planning: Makes daily plans
    - Reasoning: Reasons about risk and decisions
    - Agency: Makes actual decisions (not just probabilities)
    - Death: Can die from infection (age-stratified IFR + cluster context)
    
    Modes:
    - trace_collection: Lightweight reasoning for pattern capture
    - full: Complete hybrid reasoning for production simulation
    """
    
    def __init__(
        self,
        agent_id: int,
        profile: Dict,
        tool_registry: EpidemicToolRegistry,
        llm_client: Optional[OpenAI] = None,
        logger = None,
        mode: str = 'full'  # 'trace_collection' or 'full'
    ):
        """
        Initialize entity agent.
        
        Args:
            agent_id: Unique agent identifier
            profile: Agent profile (demographics, health, etc.)
            tool_registry: Registry of available tools
            llm_client: OpenAI client
            logger: Optional conversation logger
            mode: 'trace_collection' for pre-simulation, 'full' for main simulation
        """
        self.agent_id = agent_id
        self.profile = profile
        self.name = profile['name']
        self.mode = mode

        # Singapore-specific profile fields
        self.is_imported = profile.get('is_imported', False)
        self.cluster_name = profile.get('cluster', '')
        self.nationality = profile.get('nationality', 'unknown')
        self.gender = profile.get('gender', 'unknown')
        self.quarantine_status = profile.get('quarantine_status', False)
        
        # LLM client
        self.llm_client = llm_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Logger (optional)
        self.logger = logger
        
        # Memory system (only in full mode)
        if mode == 'full':
            self.memory = MemoryStream(agent_id, self.name, self.llm_client)
        else:
            self.memory = None
        
        # Tools
        self.tools = tool_registry
        
        # State tracking
        self.state = profile['initial_state']
        self.days_in_state = 0
        self.symptoms = []
        self.is_isolated = False
        
        # Death tracking
        self.is_dead = (self.state == 'D')
        self.days_since_death = 0
        self.death_date = None
        
        # Neighbors (will be set externally)
        self.neighbors: List['EntityAgent'] = []
        
        # Current plan
        self.daily_plan = []
        self.current_activity = None
        
        # State history tracking
        self.state_history = [self.state]
        
        # Trace collection logging
        self.conversation_history = []
        self.tool_usage_log = []
        self.last_conversation = None
        self.last_tools_used = []
        self.last_reasoning_text = ""
        
        # Initialize with background (full mode only)
        if mode == 'full' and not self.is_dead:
            self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize agent with background knowledge."""
        cluster_info = f" (Cluster: {self.cluster_name})" if self.cluster_name else ""
        imported_info = " [Imported case]" if self.is_imported else ""
        quarantine_info = " [Under Quarantine Order]" if self.quarantine_status else ""
        
        background = f"""I am {self.name}, a {self.profile['age']}-year-old {self.profile['occupation']}{imported_info}.
                        I am a {self.nationality} national living in Singapore.
                        I live in household {self.profile['household_id']}{cluster_info}{quarantine_info}.
                        My vaccination status: {['Unvaccinated', 'Partially vaccinated', 'Fully vaccinated'][self.profile['vaccination_status']]}.
                        I have {self.profile['comorbidity_count']} health conditions.
                        My current infection status: {self.state}."""
        
        if self.memory:
            self.memory.add(background, importance=8.0, memory_type="reflection")
    
    def add_neighbor(self, neighbor: 'EntityAgent'):
        """Add a neighboring agent."""
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)

    def _ifr_to_daily_hazard(self, ifr: float, expected_days: int = 10) -> float:
        """
        Convert cumulative IFR to daily death hazard.
        
        Formula: daily_hazard = 1 - (1 - IFR)^(1/days)
        
        Example: IFR=7% over 10 days → 0.73% daily hazard
        """
        if ifr <= 0:
            return 0.0
        ifr = min(ifr, 0.99)  # Cap at 99%
        return 1.0 - (1.0 - ifr) ** (1.0 / expected_days)
    
    def is_active(self) -> bool:
        """Check if agent is active (not dead)."""
        return not self.is_dead
    
    def should_be_removed_from_network(self, current_timestep: int) -> bool:
        """
        Check if dead agent should be removed from network.
        
        Dead agents remain in network for 7-10 days (funeral/mourning period).
        After that, they're removed from active simulation.
        
        Args:
            current_timestep: Current simulation timestep
            
        Returns:
            True if agent should be removed
        """
        if not self.is_dead:
            return False
        
        # Remove after 10 days
        return self.days_since_death > 10
    
    def _sanitize_for_json(self, obj):
        """
        Convert numpy types to Python types for JSON serialization.
        
        Compatible with NumPy 2.0+ (np.float_ was removed).
        """
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def perceive_environment(self, timestep: int) -> str:
        """
        Perceive current state of environment.
        
        Args:
            timestep: Current simulation timestep
            
        Returns:
            Perception summary
        """
        if self.mode != 'full' or self.is_dead:
            return ""  # Dead agents don't perceive
        
        perceptions = []
        
        # Observe own state
        if self.state == 'I':
            perceptions.append(f"I am currently infected (day {self.days_in_state} of infection).")
            if self.symptoms:
                perceptions.append(f"I have symptoms: {', '.join(self.symptoms)}.")
        elif self.state == 'R':
            perceptions.append(f"I have recovered from COVID-19 ({self.days_in_state} days ago).")
        else:
            perceptions.append("I am currently healthy and susceptible.")
        
        # Observe neighbors (including dead)
        infected_neighbors = [n for n in self.neighbors if n.state == 'I']
        dead_neighbors = [n for n in self.neighbors if n.is_dead and n.days_since_death <= 10]
        
        if infected_neighbors:
            perceptions.append(
                f"I notice {len(infected_neighbors)} of my contacts are currently infected: "
                f"{', '.join([n.name for n in infected_neighbors])}."
            )
        
        if dead_neighbors:
            perceptions.append(
                f"I am mourning {len(dead_neighbors)} contacts who recently died from COVID-19: "
                f"{', '.join([n.name for n in dead_neighbors])}."
            )
        
        # Store perceptions in memory
        perception_text = " ".join(perceptions)
        if self.memory:
            self.memory.add(_compress(perception_text), importance=6.0)
        
        return perception_text
    
    def reason_and_sample_transition(
        self,
        cluster_probabilities: Dict,
        neighbor_context: str,
        timestep: int
    ) -> Dict:
        """
        Agent determines their state transition through hybrid reasoning.
        
        Dead agents don't reason - they increment days_since_death.
        
        Routes to appropriate mode:
        - trace_collection: Lightweight, logging-focused (NO DEATH)
        - full: Complete hybrid reasoning (WITH DEATH)
        
        Args:
            cluster_probabilities: Dict with all cluster-level probabilities (including I→D)
            neighbor_context: Context about neighboring clusters
            timestep: Current timestep
            
        Returns:
            Dict with final state, probability, and reasoning (JSON-serializable)
        """
        # Dead agents don't transition
        if self.is_dead:
            self.days_since_death += 1
            result = {
                'old_state': 'D',
                'new_state': 'D',
                'transitioned': False,
                'probability': 0.0,
                'reasoning': f"Deceased (day {self.days_since_death} since death)"
            }
            return self._sanitize_for_json(result)
        
        if self.mode == 'trace_collection':
            result = self._trace_collection_reasoning(cluster_probabilities, timestep)
        else:
            result = self._full_hybrid_reasoning(cluster_probabilities, neighbor_context, timestep)
        
        # Sanitize before returning
        return self._sanitize_for_json(result)
    
    def _trace_collection_reasoning(
        self,
        cluster_probabilities: Dict,
        timestep: int
    ) -> Dict:
        """
        Lightweight reasoning for trace collection mode.
        
        Focus: Capture behavioral patterns, tool usage, decision-making style.
        NOT production-quality - just for clustering data.
        
        NOTE: Death is NOT modeled in trace collection mode (too rare, not useful for clustering).
        """
        old_state = self.state
        
        # Determine relevant transition (NO DEATH in trace collection)
        transition_map = {'S': 'S->E', 'E': 'E->I', 'I': 'I->R', 'R': 'R->S', 'D': None}
        relevant_transition = transition_map.get(self.state)
        
        if not relevant_transition:
            return {
                'old_state': old_state,
                'new_state': old_state,
                'transitioned': False,
                'probability': 0.0,
                'reasoning': "No valid transition"
            }
        
        cluster_prob = cluster_probabilities.get(relevant_transition, {}).get('fused', 0.05)
        
        # Simple baseline
        baseline_multiplier = self._calculate_risk_multiplier(relevant_transition)
        baseline_prob = cluster_prob * baseline_multiplier
        baseline_prob = min(max(baseline_prob, 0.001), 0.80)
        
        # Lightweight tool usage (for logging)
        tools_used = []
        tool_outputs = {}
        
        # Tool 1: Check neighbors (always for logging)
        neighbor_info = self._check_neighbors_simple()
        tools_used.append('check_neighbors')
        tool_outputs['neighbors'] = neighbor_info
        
        # Tool 2: Query knowledge (transition-specific)
        if self.state == 'S':
            query = f"COVID-19 transmission risk {self.profile['occupation']}"
        elif self.state == 'E':
            query = f"COVID-19 incubation period disease progression"
        elif self.state == 'I':
            query = f"COVID-19 recovery timeline day {self.days_in_state} vaccination"
        else:
            query = "COVID-19 reinfection risk immunity duration"
        
        knowledge = self.tools.execute_tool("query_transmission_rules", self, query=query)
        tools_used.append('query_knowledge')
        tool_outputs['knowledge'] = knowledge
        
        # Tool 3: Get activities (optional)
        if np.random.random() < 0.5:  # 50% chance
            day_of_week = timestep % 7
            activities = self.tools.execute_tool("get_activity_schedule", self, day_of_week=day_of_week)
            tools_used.append('get_activities')
            tool_outputs['activities'] = activities
        
        # Track tool usage
        self.last_tools_used = tools_used
        self.tool_usage_log.append({
            'timestep': timestep,
            'tools': tools_used
        })
        
        # BUILD TRANSITION-SPECIFIC PROMPT
        prompt = self._build_transition_specific_prompt(
            relevant_transition=relevant_transition,
            cluster_prob=cluster_prob,
            baseline_prob=baseline_prob,
            baseline_multiplier=baseline_multiplier,
            neighbor_info=neighbor_info,
            knowledge=knowledge,
            activities=tool_outputs.get('activities', 'N/A'),
            timestep=timestep
        )
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=300
            )
            
            content = response.choices[0].message.content
            
            # Log conversation
            self.last_conversation = content
            self.conversation_history.append({
                'timestep': timestep,
                'state': self.state,
                'transition': relevant_transition,
                'prompt': prompt,
                'response': content,
                'tools_used': tools_used
            })
            
            # Parse
            adjustment, final_prob, reasoning = self._parse_llm_response_simple(content, baseline_prob)
            
            self.last_reasoning_text = reasoning
        
        except Exception as e:
            print(f"      Warning: LLM failed for agent {self.agent_id}: {e}")
            adjustment = 1.0
            final_prob = baseline_prob
            reasoning = f"Baseline {baseline_prob:.1%}"
        
        # Sample transition
        u = float(np.random.random())
        transitioned = bool(u < final_prob)
        
        new_state = old_state
        if transitioned:
            transition_results = {'S': 'E', 'E': 'I', 'I': 'R', 'R': 'S'}
            new_state = transition_results.get(old_state, old_state)
            self.state = new_state
            self.days_in_state = 0
            self.state_history.append(new_state)
        else:
            self.days_in_state += 1
        
        return {
            'old_state': old_state,
            'new_state': new_state,
            'transitioned': transitioned,
            'probability': float(final_prob),
            'reasoning': reasoning,
            'tools_used': tools_used,
            'baseline_prob': float(baseline_prob),
            'adjustment': float(adjustment),
            'sampled_value': u
        }
    
    def _build_transition_specific_prompt(
        self,
        relevant_transition: str,
        cluster_prob: float,
        baseline_prob: float,
        baseline_multiplier: float,
        neighbor_info: str,
        knowledge: str,
        activities: str,
        timestep: int
    ) -> str:
        """
        Build transition-specific prompt with COMPLETE education about SEIRD model.
        
        CRITICAL: LLM needs to understand what each state means and what affects each transition!
        """
        
        # SEIRD MODEL EXPLANATION (always included)
        seird_explanation = """=== UNDERSTANDING THE SEIRD EPIDEMIC MODEL ===

STATES (where you can be):
- S (Susceptible): Healthy, never infected, CAN catch COVID-19 from infected contacts
- E (Exposed): Infected but no symptoms yet, incubating the virus (2-5 days typically)
- I (Infected): Actively sick with symptoms, infectious to others
- R (Recovered): Was infected, now recovered with immunity (can wane over time)
- D (Dead): Deceased from infection (NOT modeled in trace collection)

TRANSITIONS (how you move between states):
1. S→E (Exposure): Catch virus from infected contacts → become exposed
   - Affected by: contact with infected people, vaccination, protective behaviors
   
2. E→I (Progression): Incubation ends → symptoms develop
   - Affected by: time since exposure, immune system strength, age, vaccination
   - NOT affected by: current contact exposure (already infected!)
   
3. I→R (Recovery): Immune system clears infection → recover
   - Affected by: time infected, age, vaccination, health conditions
   - NOT affected by: contact exposure (can't get "more infected"!)
   
4. R→S (Reinfection): Immunity wanes → become susceptible again
   - Affected by: time since recovery, vaccination status, new exposure 

CRITICAL LOGIC RULES:
✓ If you're in S: contact exposure matters (can catch it)
✓ If you're in E or I: contact exposure doesn't affect YOUR progression/recovery
✓ If you're in R: contact exposure matters again (reinfection possible if immunity wanes)

"""
        
        # Common header
        prompt = seird_explanation + f"""
=== YOUR CURRENT SITUATION ===

You are {self.name}, age {self.profile['age']}, {self.profile['occupation']}.

CURRENT STATUS (Day {timestep}):
- Your state: {self._get_state_description(self.state)}
- Days in this state: {self.days_in_state}
- Assessing transition: {relevant_transition}

PROBABILITY CALCULATION:
- Cluster baseline: {cluster_prob * 100:.1f}% (average for people like you)
- Your personal multiplier: {baseline_multiplier:.2f}x (based on age, health, contacts)
- Your baseline probability: {baseline_prob * 100:.1f}%

"""
        
        # TRANSITION-SPECIFIC CONTEXT
        if relevant_transition == 'S->E':
            singapore_context = ""

            if self.is_imported:
                singapore_context += "\n🛫 IMPORTED CASE: You traveled from abroad recently."

            if self.quarantine_status:
                singapore_context += "\n🔒 QUARANTINE ORDER: You are under strict home quarantine (reduces exposure 80%)."

            if self._is_in_dormitory_cluster():
                singapore_context += "\n🏢 DORMITORY ENVIRONMENT: High-density shared living (2.5x transmission risk)."

            singapore_section = ""
            if singapore_context:
                singapore_section = f"""
            === SINGAPORE-SPECIFIC CONTEXT ===
            {singapore_context}"""

            prompt += f"""=== S→E TRANSITION: DAILY EXPOSURE HAZARD ===

            WHAT THIS MEANS:
            You are currently SUSCEPTIBLE (not infected).
            This transition represents a DAILY PROBABILISTIC EXPOSURE PROCESS.
            You are NOT deciding whether infection happens.
            You are ONLY adjusting the EXPOSURE HAZARD based on contact intensity.

            FACTORS THAT MATTER (CONTACT CONTEXT):
            {neighbor_info}

            NOTE ON BIOLOGY (IMPORTANT):
            - Biological susceptibility, immunity, and incubation are handled OUTSIDE this reasoning.
            - Vaccination reduces disease severity and susceptibility, NOT exposure occurrence.
            - Household exposure implies sustained close contact.

            PROFILE INFORMATION (DO NOT OVERRIDE BIOLOGY):
            - Vaccination status: {['Unvaccinated', 'Partially vaccinated', 'Fully vaccinated'][self.profile['vaccination_status']]}
            - Precaution compliance: {self.profile.get('compliance_score', 0.7):.0%}
            - Age: {self.profile['age']}

            DOMAIN KNOWLEDGE (BACKGROUND ONLY):
            {knowledge[:300] if knowledge else 'No specific knowledge available'}

            TODAY'S ACTIVITIES (CONTACT INTENSITY ONLY):
            {activities[:200]}

            YOUR TASK:
            Adjust the DAILY EXPOSURE HAZARD relative to baseline.
            You MUST NOT reason about avoiding infection.
            You MUST NOT introduce new biological assumptions.

            VALID REASONS TO INCREASE HAZARD:
            ✓ More infected contacts
            ✓ Household infected contacts
            ✓ Crowded or high-contact activities

            VALID REASONS TO DECREASE HAZARD:
            ✓ Reduced social activity
            ✓ Isolation or remote work

            INVALID REASONS (DO NOT USE):
            ✗ "Being careful"
            ✗ "Avoiding infection"
            ✗ Vaccination preventing exposure
            ✗ Health status or immunity
            ✗ Moral or emotional reasoning

            BOUNDARY CONDITIONS (STRICT):
            - If infected household contacts > 0 → adjustment MUST be ≥ 1.0
            - Adjustment range: [0.9, 1.2]

            OUTPUT FORMAT (STRICT):
            ADJUSTMENT_FACTOR: <single number between 0.9 and 1.2>
            REASONING: <2–3 factual sentences based only on contact context>
            """

        
        elif relevant_transition == 'E->I':
            prompt += f"""=== E→I TRANSITION: DISEASE PROGRESSION ===

WHAT THIS MEANS:
You are currently EXPOSED (virus is incubating, no symptoms yet).
This assesses: Will you develop symptoms today?

WHAT MATTERS FOR E→I:
✓ Time since exposure (longer = more likely to progress)
✓ Your immune system (age, vaccination, health)
✗ Current infected contacts (you're ALREADY infected, can't get more infected!)

YOUR PROGRESSION FACTORS:
- Days exposed: {self.days_in_state} ({'early incubation' if self.days_in_state < 3 else 'late incubation, symptoms likely soon'})
- Vaccination: {['Unvaccinated (may develop severe symptoms)', 'Partially vaccinated (milder symptoms likely)', 'Fully vaccinated (very mild or no symptoms)'][self.profile['vaccination_status']]}
- Age: {self.profile['age']} ({'young = strong immune response' if self.profile['age'] < 40 else 'older = slower immune response'})
- Health conditions: {self.profile['comorbidity_count']} ({'healthy' if self.profile['comorbidity_count'] == 0 else 'comorbidities present'})

DOMAIN KNOWLEDGE:
{knowledge[:400] if knowledge else 'No specific knowledge available'}

YOUR TASK:
Assess your E→I probability (will you develop symptoms?).

Key considerations:
1. How long have you been exposed? (2-5 days is typical incubation)
2. Does your immune system fight it off? (young + vaccinated = may not progress)
3. Are you healthy or vulnerable? (comorbidities = more likely to progress)

IGNORE: How many people around you are infected (irrelevant - you're already infected!)

"""
        
        elif relevant_transition == 'I->R':
            prompt += f"""=== I→R TRANSITION: RECOVERY ===

WHAT THIS MEANS:
You are currently INFECTED (actively sick with symptoms).
This assesses: Will you recover today?

WHAT MATTERS FOR I→R:
✓ Time infected (longer = closer to recovery)
✓ Your immune response (age, vaccination, health)
✗ Current infected contacts (COMPLETELY IRRELEVANT - you're already sick!)

YOUR RECOVERY FACTORS:
- Days infected: {self.days_in_state} ({'early infection' if self.days_in_state < 4 else 'mid-late infection' if self.days_in_state < 7 else 'should recover soon'})
- Vaccination: {['Unvaccinated (slower recovery)', 'Partially vaccinated (faster recovery)', 'Fully vaccinated (much faster recovery, 1.3x)'][self.profile['vaccination_status']]}
- Age: {self.profile['age']} ({'young = fast recovery' if self.profile['age'] < 40 else 'older = slower recovery'})
- Health conditions: {self.profile['comorbidity_count']} ({'healthy = fast recovery' if self.profile['comorbidity_count'] == 0 else 'comorbidities = slower recovery'})

DOMAIN KNOWLEDGE:
{knowledge[:400] if knowledge else 'No specific knowledge available'}

YOUR TASK:
Assess your I→R probability (will you recover today?).

Key considerations:
1. How long have you been sick? (5-10 days is typical recovery time)
2. Does your vaccination help? (vaccinated recover 30% faster)
3. Are you young and healthy? (young + healthy = fast recovery)
4. Are you elderly or have health issues? (older + comorbid = slower recovery)

CRITICAL: Infected contacts around you DO NOT affect your recovery!
You cannot get "more infected" - you're already infected.
Recovery depends on YOUR immune system, not exposure.

"""
        
        elif relevant_transition == 'R->S':
            prompt += f"""=== R→S TRANSITION: REINFECTION RISK ===

WHAT THIS MEANS:
You are currently RECOVERED (had COVID, now immune).
This assesses: Will you lose immunity and become susceptible again?

FACTORS THAT MATTER:
- Days since recovery: {self.days_in_state} ({'strong immunity' if self.days_in_state < 90 else 'waning immunity' if self.days_in_state < 180 else 'weak immunity'})
- Vaccination: {['Unvaccinated (immunity wanes faster)', 'Partially vaccinated (some boost)', 'Fully vaccinated (strong lasting immunity)'][self.profile['vaccination_status']]}

CURRENT EXPOSURE:
{neighbor_info}

DOMAIN KNOWLEDGE:
{knowledge[:400] if knowledge else 'No specific knowledge available'}

YOUR TASK:
Assess your R→S probability (reinfection risk).

Key considerations:
1. How long ago did you recover? (< 90 days = very low risk, > 180 days = higher risk)
2. Are you vaccinated? (vaccination extends immunity duration)
3. Are you exposed to infected contacts? (exposure + waning immunity = reinfection possible)

"""
        
        else:
            # Fallback (should not happen)
            prompt += f"""CONTEXT:
{neighbor_info}

KNOWLEDGE:
{knowledge[:300] if knowledge else 'No knowledge available'}

TASK: Assess your {relevant_transition} probability.
"""
        
        # Common ending
        prompt += """
=== YOUR RESPONSE ===

Provide your assessment in this EXACT format:

REASONING: <2-3 sentences explaining your logic based on the factors above>
ADJUSTMENT: <number between 0.7 and 1.3>
PROBABILITY: <final probability as decimal, e.g., 0.15>

IMPORTANT RULES:
- Use decimal format (0.15) NOT percentage (15%)
- ADJUSTMENT should be relative to your baseline
- PROBABILITY should be baseline × adjustment
- Focus only on factors that actually affect THIS transition

Example:
REASONING: I've been infected for 5 days with full vaccination. Vaccinated people recover 30% faster. My age (32) also helps recovery.
ADJUSTMENT: 1.2
PROBABILITY: 0.14
"""
        
        return prompt
    
    def _get_state_description(self, state: str) -> str:
        """Get human-readable state description."""
        descriptions = {
            'S': 'S (Susceptible - healthy, can catch COVID)',
            'E': 'E (Exposed - infected but no symptoms yet)',
            'I': 'I (Infected - actively sick with symptoms)',
            'R': 'R (Recovered - immune, had COVID)',
            'D': 'D (Deceased)'
        }
        return descriptions.get(state, state)

    def _full_hybrid_reasoning(
        self,
        cluster_probabilities: Dict,
        neighbor_context: str,
        timestep: int
    ) -> Dict:
        """
        FULL hybrid reasoning for production simulation.
        Complete implementation with all features INCLUDING DEATH (I→D).

        FIXED: No competing risk normalization - use raw hazards directly.
        """
        old_state = self.state

        # ============================================================
        # HARD BIOLOGICAL IMMUNITY GATE (NON-NEGOTIABLE)
        # ============================================================
        if self.state == 'R' and self.days_in_state < 180:
            self.days_in_state += 1
            return {
                'old_state': 'R',
                'new_state': 'R',
                'transitioned': False,
                'probability': 0.0,
                'reasoning': (
                    f"Recovered {self.days_in_state} days ago — immunity intact. "
                    f"Reinfection biologically impossible before 180 days."
                )
            }

        # ============================================================
        # SPECIAL CASE: INFECTED AGENTS — THREE POSSIBLE OUTCOMES
        # ============================================================
        if self.state == 'I':
            print(f"          [Step 0] Infected agent: death vs recovery vs stay infected...", flush=True)

            # -------- Death hazard (from IFR) --------
            cluster_ifr = cluster_probabilities.get('I->D', {}).get('fused', 0.005)
            ifr_multiplier = self._calculate_risk_multiplier('I->D')
            individual_ifr = min(cluster_ifr * ifr_multiplier, 0.40)

            death_prob = self._ifr_to_daily_hazard(
                individual_ifr,
                expected_days=10
            )

            # -------- Recovery hazard --------
            cluster_recov = cluster_probabilities.get('I->R', {}).get('fused', 0.15)
            recov_multiplier = self._calculate_risk_multiplier('I->R')
            recovery_prob = min(cluster_recov * recov_multiplier, 0.90)

            # ============================================================
            # FIXED: NO NORMALIZATION - Use raw hazards directly
            # 
            # Three outcomes:
            # 1. Die (death_prob)
            # 2. Recover (recovery_prob)
            # 3. Stay infected (1 - death_prob - recovery_prob)
            # ============================================================
            
            u = float(np.random.random())

            print(
                f"            Hazards → death: {death_prob:.4f}, "
                f"recovery: {recovery_prob:.4f}, "
                f"stay infected: {1.0 - death_prob - recovery_prob:.4f}", flush=True
            )

            # -------- Sample outcome (no normalization) --------
            if u < death_prob:
                # DIE
                self.state = 'D'
                self.is_dead = True
                self.death_date = timestep
                self.days_since_death = 0
                self.state_history.append('D')

                if self.memory:
                    self.memory.add(
                        f"Day {timestep}: Died from COVID-19. "
                        f"Age {self.profile['age']}, "
                        f"day {self.days_in_state} infected.",
                        importance=10.0,
                        memory_type="reflection"
                    )

                print(f"            Sampled {u:.4f} < {death_prob:.4f} → DIED ✗", flush=True)

                return {
                    'old_state': 'I',
                    'new_state': 'D',
                    'transitioned': True,
                    'probability': float(death_prob),
                    'reasoning': (
                        f"Died from COVID-19 "
                        f"(IFR={individual_ifr:.2%}, "
                        f"day {self.days_in_state})"
                    ),
                    'cluster_prob': float(cluster_ifr),
                    'baseline_prob': float(death_prob),
                    'sampled_value': u,
                    'method': 'raw_competing_hazards'
                }

            elif u < death_prob + recovery_prob:
                # RECOVER
                self.state = 'R'
                self.days_in_state = 0
                self.symptoms = []
                self.is_isolated = False
                self.state_history.append('R')

                if self.memory:
                    self.memory.add(
                        f"Day {timestep}: Recovered from COVID-19.",
                        importance=9.0
                    )

                print(f"            Sampled {u:.4f} < {death_prob + recovery_prob:.4f} → RECOVERED ✓", flush=True)

                return {
                    'old_state': 'I',
                    'new_state': 'R',
                    'transitioned': True,
                    'probability': float(recovery_prob),
                    'reasoning': "Recovered from infection",
                    'cluster_prob': float(cluster_recov),
                    'baseline_prob': float(recovery_prob),
                    'sampled_value': u,
                    'method': 'raw_competing_hazards'
                }

            else:
                # STAY INFECTED
                self.days_in_state += 1
                print(f"            Sampled {u:.4f} ≥ {death_prob + recovery_prob:.4f} → REMAINS INFECTED", flush=True)

                return {
                    'old_state': 'I',
                    'new_state': 'I',
                    'transitioned': False,
                    'probability': float(1.0 - death_prob - recovery_prob),
                    'reasoning': "Remains infected",
                    'cluster_prob': float(cluster_recov),
                    'baseline_prob': float(recovery_prob),
                    'sampled_value': u,
                    'agent_id': self.agent_id
                }

        # ============================================================
        # NORMAL TRANSITIONS (S→E, E→I, R→S)
        # ============================================================
        transition_map = {'S': 'S->E', 'E': 'E->I', 'R': 'R->S'}
        relevant_transition = transition_map.get(self.state)

        if not relevant_transition:
            return {
                'old_state': old_state,
                'new_state': old_state,
                'transitioned': False,
                'probability': 0.0,
                'reasoning': "Unknown state"
            }

        cluster_prob_info = cluster_probabilities.get(relevant_transition, {})
        cluster_prob = float(cluster_prob_info.get('fused', 0.05))

        print(f"          [Step 1] Gathering information...", flush=True)
        tool_outputs = self._gather_personal_information(timestep)
        memory_context = self._retrieve_relevant_memories(relevant_transition)
        history_context = self._review_state_history()

        print(f"          [Step 2] Calculating baseline risk...", flush=True)
        baseline_multiplier = self._calculate_risk_multiplier(relevant_transition)
        baseline_prob = min(max(cluster_prob * baseline_multiplier, 0.001), 0.80)

        print(
            f"            Baseline: {cluster_prob:.3f} × "
            f"{baseline_multiplier:.2f} = {baseline_prob:.3f}",
            flush=True
        )

        print(f"          [Step 3] LLM analyzing context...", flush=True)
        reasoning_result = self._hybrid_reasoning_llm(
            transition_type=relevant_transition,
            cluster_prob=cluster_prob,
            baseline_prob=baseline_prob,
            baseline_multiplier=baseline_multiplier,
            tool_outputs=tool_outputs,
            memory_context=memory_context,
            history_context=history_context,
            cluster_probabilities=cluster_probabilities,
            timestep=timestep
        )

        personal_prob = float(reasoning_result['probability'])
        reasoning_text = reasoning_result['reasoning']
        llm_adjustment = float(reasoning_result.get('adjustment', 1.0))

        print(
            f"            LLM adjustment: {llm_adjustment:.2f}x "
            f"→ Personal prob: {personal_prob:.3f}",
            flush=True
        )

        u = float(np.random.random())
        transitioned = bool(u < personal_prob)

        new_state = old_state
        if transitioned:
            new_state = {'S': 'E', 'E': 'I', 'R': 'S'}[old_state]
            self.state = new_state
            self.days_in_state = 0
            self.state_history.append(new_state)
        else:
            self.days_in_state += 1

        return {
            'old_state': old_state,
            'new_state': new_state,
            'transitioned': transitioned,
            'probability': personal_prob,
            'reasoning': reasoning_text,
            'cluster_prob': cluster_prob,
            'baseline_prob': float(baseline_prob),
            'sampled_value': u,
            'agent_id': self.agent_id
        }

    
    def _check_neighbors_simple(self) -> str:
        """Simple neighbor check for trace collection, Singapore-aware."""
        infected = [n for n in self.neighbors if n.state in ['E', 'I']]
        dead = [n for n in self.neighbors if n.is_dead and n.days_since_death <= 10]
        household_infected = [n for n in infected 
                            if n.profile.get('household_id') == self.profile.get('household_id')]
        household_dead = [n for n in dead
                        if n.profile.get('household_id') == self.profile.get('household_id')]
        
        # Singapore-specific: Check dormitory contacts
        dorm_infected = [n for n in infected 
                        if 'dorm' in n.profile.get('cluster', '').lower()]
        
        total = len(self.neighbors)
        n_infected = len(infected)
        n_household = len(household_infected)
        n_dead = len(dead)
        n_household_dead = len(household_dead)
        n_dorm = len(dorm_infected)
        
        info_parts = []
        
        if n_household > 0:
            info_parts.append(f"HOUSEHOLD: {n_household} infected household members!")
        if n_household_dead > 0:
            info_parts.append(f"HOUSEHOLD DEATHS: {n_household_dead} household members died!")
        
        # Singapore dormitory context
        if n_dorm > 0 and self._is_in_dormitory_cluster():
            info_parts.append(f"DORMITORY: {n_dorm} infected dormitory contacts (high-density)!")
        
        if n_infected > 0:
            info_parts.append(f"CONTACTS: {n_infected}/{total} infected.")
        if n_dead > 0 and n_household_dead == 0:
            info_parts.append(f"DEATHS: {n_dead} contacts died from COVID.")
        
        if not info_parts:
            info_parts.append(f"CONTACTS: All {total} contacts are healthy.")
        
        return " ".join(info_parts)
    
    def _parse_llm_response_simple(self, content: str, baseline: float) -> Tuple[float, float, str]:
        """Parse LLM response for trace collection mode."""
        adjustment = 1.0
        probability = baseline
        reasoning = ""
        
        try:
            # Extract ADJUSTMENT
            if "ADJUSTMENT:" in content:
                adj_line = content.split("ADJUSTMENT:")[1].split("\n")[0].strip()
                adj_text = adj_line.split()[0] if adj_line.split() else "1.0"
                adj_text = adj_text.replace('%', '').replace(',', '').strip()
                try:
                    adjustment = float(adj_text)
                    adjustment = float(np.clip(adjustment, 0.6, 1.5))
                except ValueError:
                    adjustment = 1.0
            
            # Extract PROBABILITY
            if "PROBABILITY:" in content:
                prob_line = content.split("PROBABILITY:")[1].split("\n")[0].strip()
                prob_text = prob_line.split()[0] if prob_line.split() else str(baseline)
                prob_text = prob_text.replace(',', '').strip()
                
                try:
                    if '%' in prob_text:
                        prob_text = prob_text.replace('%', '')
                        probability = float(prob_text) / 100.0
                    else:
                        probability = float(prob_text)
                    
                    probability = float(np.clip(probability, 0.001, 0.80))
                except ValueError:
                    probability = baseline * adjustment
            else:
                probability = baseline * adjustment
            
            # Extract REASONING
            if "REASONING:" in content:
                reasoning = content.split("REASONING:")[1].split("ADJUSTMENT:")[0].strip()
                reasoning = reasoning[:200]
            else:
                reasoning = content[:100]
        
        except Exception:
            adjustment = 1.0
            probability = baseline
            reasoning = content[:100] if content else "No reasoning provided"
        
        return float(adjustment), float(probability), reasoning
    
    def _gather_personal_information(self, timestep: int) -> Dict:
        """Gather comprehensive information using tools (FULL MODE)."""
        outputs = {}
        
        # Tool 1: Check neighbors (including dead)
        outputs['neighbors'] = self.tools.execute_tool("check_neighbor_states", self)
        
        # Tool 2: Calculate exposure (if susceptible)
        if self.state in ['S', 'R']:
            outputs['exposure'] = self.tools.execute_tool("calculate_exposure_risk", self)
        
        # Tool 3: Check disease progression (if exposed/infected)
        if self.state in ['E', 'I']:
            outputs['progression'] = self.tools.execute_tool("check_disease_progression", self)
        
        # Tool 4: Get activity schedule
        day_of_week = timestep % 7
        outputs['activities'] = self.tools.execute_tool("get_activity_schedule", self, day_of_week=day_of_week)
        
        # Tool 5: Query RAG
        if self.state == 'S':
            query = f"daily infection risk for {self.profile['occupation']} age {self.profile['age']}"
        elif self.state in ['E', 'I']:
            query = f"recovery timeline day {self.days_in_state} infected vaccination"
        else:
            query = f"reinfection risk {self.days_in_state} days recovered"
        
        outputs['knowledge'] = self.tools.execute_tool("query_transmission_rules", self, query=query)
        
        return outputs

    def _retrieve_relevant_memories(self, transition_type: str) -> str:
        """Retrieve relevant memories (FULL MODE)."""
        if not self.memory:
            return "No memory system available."
        
        query_map = {
            'S->E': "infection exposure risk contacts symptoms protection",
            'E->I': "symptoms progression disease development",
            'I->R': "recovery symptoms health progression timeline",
            'R->S': "immunity reinfection protection waning"
        }
        
        query = query_map.get(transition_type, "health status")
        memories = self.memory.retrieve(query, n=5)
        
        if not memories:
            return "No relevant past memories."
        
        return "\n".join([f"- {m.description[:150]}" for m in memories])

    def _review_state_history(self) -> str:
        """Review personal state history."""
        if len(self.state_history) < 2:
            return f"Current state: {self.state} (day {self.days_in_state})"
        
        recent = self.state_history[-7:]
        return f"Recent history (last {len(recent)} days): {' → '.join(recent)}. Current: {self.state} (day {self.days_in_state})"

    def _calculate_risk_multiplier(self, transition_type: str) -> float:
        """
        Calculate deterministic baseline risk multiplier for SEIRD transitions.
        
        UPDATED: Uses EpidemicParams for all transmission parameters.
        """
        multiplier = 1.0
        
        if transition_type == 'S->E':  # Exposure
            # STEP 1: Vaccination FIRST (reduces base susceptibility)
            vacc_efficacy = EpidemicParams.VACCINE_EFFICACY[self.profile['vaccination_status']]
            multiplier *= vacc_efficacy['exposure']
            
            # STEP 2: Contact exposure using calibrated β
            infected = [n for n in self.neighbors if n.state in ['E', 'I']]
            household_infected = [
                n for n in infected 
                if n.profile.get('household_id') == self.profile.get('household_id')
            ]

            # ⭐ SINGAPORE-SPECIFIC: Quarantine enforcement
            if self.quarantine_status:
                multiplier *= 0.20  # 80% reduction under Quarantine Order
            
            # ⭐ SINGAPORE-SPECIFIC: Dormitory cluster detection
            is_dormitory = self._is_in_dormitory_cluster()
            if is_dormitory:
                # Dormitory outbreaks: 2.5x community transmission
                multiplier *= 2.5
            
            if len(household_infected) > 0:
                # ⭐ FIXED: Use β directly, don't add 1.0 (was double-counting!)
                beta = EpidemicParams.BETA_HOUSEHOLD
                # Direct β usage (no exponential, no doubling)
                household_risk = beta * len(household_infected)
                multiplier *= (1.0 + household_risk)  # ⭐ 1 + (0.08 × 1) = 1.08x (not 1.70x!)
                
            elif len(infected) > 0:
                # ⭐ FIXED: Direct β usage
                beta_contact = EpidemicParams.BETA_CONTACT
                contact_risk = beta_contact * len(infected)
                multiplier *= (1.0 + contact_risk)  # ⭐ 1 + (0.005 × 10) = 1.05x
                
            else:
                # ⭐ Community baseline
                multiplier *= (1.0 + EpidemicParams.BETA_COMMUNITY)
            
            # STEP 3: Age (unchanged)
            age = self.profile['age']
            if age < 18:
                multiplier *= 0.7
            elif age > 60:
                multiplier *= 1.3
            
            # STEP 4: Comorbidities (unchanged)
            multiplier *= (1.0 + self.profile['comorbidity_count'] * 0.15)
            
            # STEP 5: Occupation (unchanged)
            occupation = self.profile['occupation']
            occ_multiplier = EpidemicParams.OCCUPATION_RISK.get(occupation, 1.0)
            multiplier *= occ_multiplier
            
            # STEP 6: Compliance (unchanged)
            multiplier *= (2.0 - self.profile.get('compliance_score', 0.7))
        
        elif transition_type in ['E->I', 'I->R']:  # Progression/Recovery
            # ⭐ STRONGER TIME GATING
            days = self.days_in_state
            
            if transition_type == 'E->I':
                # Incubation: NO symptoms before day 2
                if days < 2:
                    return 0.0  # ⭐ HARD GATE
                elif days < 4:
                    multiplier *= 0.5  # ⭐ Slow ramp
                else:
                    multiplier *= min(1.0 + 0.3 * (days - 4), 2.0)
            
            else:  # I->R
                # Recovery: NO recovery before day 7
                if days < 7:
                    return 0.0  # ⭐ HARD GATE
                elif days < 10:
                    multiplier *= 0.4  # ⭐ Slow recovery
                else:
                    multiplier *= min(1.0 + 0.2 * (days - 10), 2.5)

            # ⭐ Vaccination using calibrated efficacy
            vacc_status = self.profile['vaccination_status']
            vacc_efficacy = EpidemicParams.VACCINE_EFFICACY[vacc_status]
            
            if transition_type == 'I->R':
                multiplier *= vacc_efficacy['recovery']
            else:  # E->I
                multiplier *= vacc_efficacy['progression']
            
            # Age (unchanged)
            age = self.profile['age']
            if transition_type == 'I->R':
                if age < 18:
                    multiplier *= 1.3
                elif age > 60:
                    multiplier *= 0.7
            else:  # E->I
                if age < 18:
                    multiplier *= 0.9
                elif age > 60:
                    multiplier *= 1.3
            
            # Comorbidities (unchanged)
            if transition_type == 'I->R':
                multiplier *= (0.85 ** self.profile['comorbidity_count'])
            else:
                multiplier *= (1.0 + self.profile['comorbidity_count'] * 0.15)

        
        elif transition_type == 'I->D':  # DEATH - SINGAPORE
            age = self.profile['age']
            
            # ⭐ Use age-stratified IFR from EpidemicParams
            base_ifr = EpidemicParams.get_ifr(age)
            
            # ⭐ SINGAPORE HEALTHCARE FACTOR (0.3x global IFR)
            singapore_healthcare_mult = 0.3
            base_ifr *= singapore_healthcare_mult
            
            # Relative to 50-70 age group (IFR = 0.005 * 0.3 = 0.0015)
            multiplier = base_ifr / 0.0015  # Normalize to Singapore baseline
            
            # DAYS INFECTED (peak risk days 7-14)
            days = self.days_in_state
            if days < 7:
                multiplier *= 0.05  # Very low early
            elif 7 <= days <= 14:
                multiplier *= 1.5  # Peak (lower than global 2.0)
            else:
                multiplier *= 0.2  # Late risk
            
            # COMORBIDITIES
            multiplier *= (1.0 + self.profile.get('comorbidity_count', 0) * 0.5)
            
            # ⭐ VACCINATION PROTECTION
            vacc_status = self.profile.get('vaccination_status', 0)
            vacc_efficacy = EpidemicParams.VACCINE_EFFICACY[vacc_status]
            multiplier *= vacc_efficacy['death']
        
        elif transition_type == 'R->S':  # Reinfection
            # ⭐ Use immunity duration from EpidemicParams
            if self.days_in_state < EpidemicParams.IMMUNITY_DURATION:
                return 0.0
        
        return multiplier

    def _is_in_dormitory_cluster(self) -> bool:
        """
        Detect if agent is in dormitory cluster (Singapore-specific).
        
        Returns:
            True if in dormitory environment
        """
        # Check cluster name
        if 'dorm' in self.cluster_name.lower():
            return True
        
        # Check occupation
        if self.profile.get('occupation') == 'manual_worker':
            return True
        
        # Check if most neighbors are manual workers (dormitory indicator)
        if self.neighbors:
            manual_workers = sum(1 for n in self.neighbors 
                                if n.profile.get('occupation') == 'manual_worker')
            if manual_workers > len(self.neighbors) * 0.6:
                return True
        
        return False

    def _hybrid_reasoning_llm(
        self,
        transition_type: str,
        cluster_prob: float,
        baseline_prob: float,
        baseline_multiplier: float,
        tool_outputs: Dict,
        memory_context: str,
        history_context: str,
        cluster_probabilities: Dict,
        timestep: int
    ) -> Dict:
        """
        Hybrid reasoning with CAUSAL GATING and STOCHASTICITY.

        LLM is ONLY used for:
        - S→E (behavioral exposure)

        All biological transitions (E→I, I→R) return deterministically WITH STOCHASTICITY.
        """

        # ============================================================
        # 🚫 BIOLOGICAL TRANSITIONS — NO LLM, ADD STOCHASTICITY
        # ============================================================
        if transition_type in {"E->I", "I->R"}:
            days = self.days_in_state
            if transition_type == "E->I":
                # NO symptoms before day 2
                if days < 3:
                    return {
                        'probability': 0.0,
                        'adjustment': 0.0,
                        'reasoning': f"Incubation too early (day {days}). No symptoms before day 2."
                    }

            elif transition_type == "I->R":
                # NO recovery before day 7
                if days < 7:
                    return {
                        'probability': 0.0,
                        'adjustment': 0.0,
                        'reasoning': f"Recovery too early (day {days}). COVID recovery takes 7-14 days minimum."
                    }

            # ⭐ Add ±15% random variation (was ±20%, now tighter)
            stochastic_factor = float(np.random.uniform(0.85, 1.15))
            final_prob = baseline_prob * stochastic_factor
            final_prob = float(np.clip(final_prob, 0.001, 0.40))  # ⭐ Cap at 40% (was 50%)
            
            return {
                'probability': final_prob,
                'adjustment': stochastic_factor,
                'reasoning': (
                    f"Biological transition with individual variation "
                    f"(base={baseline_prob:.3f}, factor={stochastic_factor:.2f}). "
                    f"Days in state: {days}"
                )
            }

        # I→D SHOULD NEVER HIT THIS FUNCTION (already handled upstream)
        if transition_type == "I->D":
            return {
                'probability': float(baseline_prob),
                'adjustment': 1.0,
                'reasoning': (
                    f"IFR-based death hazard. "
                    f"Age-stratified risk with peak window adjustment "
                    f"(day {self.days_in_state})."
                )
            }

        # ============================================================
        # ✅ BEHAVIORAL TRANSITIONS — USE LLM
        # ============================================================
        if transition_type != "S->E":
            # Safety fallback
            return {
                'probability': float(baseline_prob),
                'adjustment': 1.0,
                'reasoning': "Unknown transition type. Using baseline."
            }

        # ------------------------------------------------------------
        # Behavioral context
        # ------------------------------------------------------------
        infected_contacts = len([n for n in self.neighbors if n.state in ['E', 'I']])
        household_infected = len([
            n for n in self.neighbors
            if n.state in ['E', 'I']
            and n.profile.get('household_id') == self.profile.get('household_id')
        ])

        prompt = f"""You are {self.name}, assessing {transition_type} risk for day {timestep}.

BASELINE (already computed):
- Cluster probability: {cluster_prob:.1%}
- Personal multiplier: {baseline_multiplier:.2f}x
- Baseline probability: {baseline_prob:.1%}

PROFILE:
- Age: {self.profile['age']}
- Occupation: {self.profile['occupation']}
- State: {self.state} (day {self.days_in_state})
- Vaccination: {['None', 'Partial', 'Full'][self.profile['vaccination_status']]}

LOCAL CONTEXT:
- Infected contacts: {infected_contacts}
- Household infected: {household_infected}

NEIGHBOR STATES:
{tool_outputs.get('neighbors', '')[:300]}

ACTIVITIES:
{tool_outputs.get('activities', '')[:300]}

MEMORY:
{memory_context[:300]}

TASK:
Refine the baseline probability based on BEHAVIORAL factors only.
Do NOT reason about biology or incubation.

Provide:
ADJUSTMENT_FACTOR: <0.7–1.3>
FINAL_PROBABILITY: <decimal>
REASONING: <2–3 sentences>
"""

        try:
            _llm_rate_limiter.wait_if_needed()
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You adjust exposure or reinfection risk using behavioral context only. "
                            "Do not reason about disease progression or immunity biology."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=350,
                timeout=45
            )

            content = response.choices[0].message.content
            content_clean = content.replace("**", "").replace("*", "")

            adjustment = 1.0
            final_prob = baseline_prob

            # Parse adjustment
            if "ADJUSTMENT" in content_clean:
                nums = re.findall(r'[0-9]*\.?[0-9]+', content_clean)
                if nums:
                    adjustment = float(nums[0])
                    adjustment = float(np.clip(adjustment, 0.7, 1.3))

            if household_infected > 0:
                adjustment = max(adjustment, 1.0)

            # Parse probability
            prob_match = re.search(r'FINAL_PROBABILITY[:\s]+([0-9]*\.?[0-9]+)%?', content_clean)
            if prob_match:
                final_prob = float(prob_match.group(1))
                # If it looks like a percentage (>1.0), convert
                if final_prob > 1.0:
                    final_prob /= 100.0
                elif '%' in content_clean:
                    # If percentage symbol present, treat as percentage
                    final_prob /= 100.0
            else:
                # Fallback: use baseline × adjustment
                final_prob = baseline_prob * adjustment
            
            # Clamp to reasonable range
            final_prob = float(np.clip(final_prob, 0.0, 0.80))

            # Clamp to reasonable range BASED ON TRANSITION TYPE
            if transition_type == 'S->E':
                final_prob = float(np.clip(final_prob, 0.0, 0.15))  # ⭐ Max 15% individual exposure
            elif transition_type == 'E->I':
                final_prob = float(np.clip(final_prob, 0.0, 0.30))  # ⭐ Max 30%
            elif transition_type == 'I->R':
                final_prob = float(np.clip(final_prob, 0.0, 0.40))  # ⭐ Max 40%
            else:
                final_prob = float(np.clip(final_prob, 0.0, 0.80))  # ⭐ General cap

            return {
                'probability': final_prob,
                'adjustment': adjustment,
                'reasoning': content_clean[-350:]
            }

        except Exception as e:
            return {
                'probability': float(baseline_prob),
                'adjustment': 1.0,
                'reasoning': f"LLM unavailable ({str(e)[:40]}). Used baseline."
            }
    
    def get_trace_data(self) -> Dict:
        """Get collected trace data (for trace collection mode)."""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.name,
            'conversations': self.conversation_history,
            'tool_usage_log': self.tool_usage_log,
            'state_history': self.state_history
        }
    
    def __repr__(self):
        if self.is_dead:
            return f"EntityAgent({self.agent_id}: {self.name}, DECEASED day {self.days_since_death})"
        return f"EntityAgent({self.agent_id}: {self.name}, {self.state})"
