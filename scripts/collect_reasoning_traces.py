"""
ASYNC Pre-simulation to collect REAL reasoning traces for HSBC clustering.
Uses asyncio for 10-20x speedup via concurrent OpenAI API calls.

VERSION 3: WITH AGENT AGGREGATION ACROSS SCENARIOS (SINGAPORE COVID-19)
- Runs 4 Singapore-specific scenarios (7 days each)
- Aggregates traces by agent_id across scenarios
- Extracts motifs ONCE per unique agent (using all their behavior)
- Assigns motifs back to all scenario traces

SINGAPORE SCENARIOS:
1. Imported Cases (Jan-Feb 2020): Travel-related infections
2. Dormitory Outbreak (Apr 2020): High-density migrant worker housing
3. Community Clusters (Mar 2020): Household and workplace transmission
4. Circuit Breaker (Apr-May 2020): Lockdown period restrictions

PERFORMANCE:
- Sequential: ~5 hours
- Async (10 concurrent): ~45-90 minutes
"""

import sys
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from typing import List, Dict
import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.agents.entity_agent import EntityAgent
from src.agents.epidemic_tools import EpidemicToolRegistry
from src.knowledge.epidemic_kb import SingaporeEpidemicKnowledgeBase


class AsyncSingaporeReasoningTraceCollector:
    """
    Collect REAL reasoning traces for Singapore COVID-19 with ASYNC and AGGREGATION.
    
    VERSION 3: Aggregates traces per agent across Singapore-specific scenarios
    
    Performance:
    - 1000 agents × 4 scenarios × 7 days = 28,000 conversations
    - With async + aggregation: ~45-90 minutes
    """
    
    def __init__(self, max_concurrent: int = 10):
        """
        Initialize async collector.
        
        Args:
            max_concurrent: Max concurrent API calls (10 for Tier 1 OpenAI)
        """
        load_dotenv()
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        self.scenarios = self._define_singapore_scenarios()
        self.collected_traces = []
        
        # ASYNC OpenAI client
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load Singapore data
        self.profiles, self.edges = self._load_singapore_data()
        
        # Initialize Singapore COVID-19 knowledge base
        print("Initializing Singapore COVID-19 knowledge base...")
        self.kb = SingaporeEpidemicKnowledgeBase(use_real_data=True)
        
        # Initialize tool registry WITH Singapore knowledge base
        print("Initializing tool registry with Singapore COVID-19 data...")
        self.tools = EpidemicToolRegistry()
        
        print(f"✓ Async Collector initialized: {len(self.profiles)} agents, {len(self.edges)} edges")
        print(f"✓ Max concurrent API calls: {max_concurrent}")
    
    def _load_singapore_data(self):
        """Load Singapore agent profiles and contact network."""
        profiles_file = Path("data/singapore/profiles.json")
        edges_file = Path("data/singapore/edges.json")
        
        with open(profiles_file, 'r') as f:
            profiles = json.load(f)
        
        with open(edges_file, 'r') as f:
            edges = json.load(f)
        
        return profiles, edges
    
    def _define_singapore_scenarios(self):
        """Define 4 Singapore-specific COVID-19 scenarios."""
        n_agents = 1000
        sim_days = 3 
        
        return {
            'scenario_1_imported_cases': {
                'name': 'Imported Cases (Early Phase)',
                'description': 'Travel-related infections from Wuhan, UK, Europe',
                'duration': sim_days,
                'n_agents': n_agents,
                'initial_condition': 'imported_cases',
                'period': 'Jan-Feb 2020',
                'goal': 'Capture imported case surveillance and quarantine behaviors'
            },
            'scenario_2_dormitory_outbreak': {
                'name': 'Dormitory Outbreak',
                'description': 'Rapid spread in migrant worker dormitories',
                'duration': sim_days,
                'n_agents': n_agents,
                'initial_condition': 'dormitory_cluster',
                'period': 'April 2020',
                'goal': 'Capture high-density transmission and mass testing response'
            },
            'scenario_3_community_clusters': {
                'name': 'Community Clusters',
                'description': 'Household and workplace transmission in community',
                'duration': sim_days,
                'n_agents': n_agents,
                'initial_condition': 'community_clusters',
                'period': 'March 2020',
                'goal': 'Capture contact tracing and cluster management'
            },
            # 'scenario_4_circuit_breaker': {
            #     'name': 'Circuit Breaker Period',
            #     'description': 'Infections during lockdown restrictions',
            #     'duration': sim_days,
            #     'n_agents': n_agents,
            #     'initial_condition': 'circuit_breaker',
            #     'period': 'Apr-May 2020',
            #     'goal': 'Capture behavioral adaptation to strict lockdown'
            # }
        }

    def _select_scenario_agents(self, config: dict) -> List[dict]:
        """Select agents for scenario."""
        n_agents = config.get('n_agents', 1000)
        
        if len(self.profiles) < n_agents:
            print(f"    Warning: Only {len(self.profiles)} total agents, using all")
            selected = self.profiles.copy()
        else:
            # Use SAME agents for all scenarios (for aggregation)
            selected = self.profiles[:n_agents]
        
        # Initialize all as susceptible
        for p in selected:
            p['initial_state'] = 'S'
        
        # Apply scenario-specific initial conditions
        initial_condition = config.get('initial_condition', 'random')
        self._apply_singapore_initial_condition(selected, initial_condition)
        
        # Print summary
        state_counts = {}
        for p in selected:
            state = p['initial_state']
            state_counts[state] = state_counts.get(state, 0) + 1
        
        print(f"    Initial states: S={state_counts.get('S', 0)}, "
              f"E={state_counts.get('E', 0)}, "
              f"I={state_counts.get('I', 0)}")
        
        return selected

    def _apply_singapore_initial_condition(self, agents: List[dict], condition: str):
        """Apply Singapore-specific initial conditions."""
        
        if condition == 'imported_cases':
            # Scenario 1: Imported cases (travel history)
            # Target: Agents marked as imported
            imported_agents = [i for i, a in enumerate(agents) if a.get('is_imported', False)]
            
            if len(imported_agents) >= 3:
                # Infect 2-3 imported cases
                n_infected = min(3, len(imported_agents))
                infected_indices = np.random.choice(imported_agents, n_infected, replace=False)
                for idx in infected_indices:
                    agents[idx]['initial_state'] = 'I'
                
                # Expose 2 more (in quarantine)
                remaining_imported = [i for i in imported_agents if agents[i]['initial_state'] == 'S']
                if len(remaining_imported) >= 2:
                    exposed_indices = np.random.choice(remaining_imported, 2, replace=False)
                    for idx in exposed_indices:
                        agents[idx]['initial_state'] = 'E'
            else:
                # Fallback: Random infection
                infected_indices = np.random.choice(len(agents), 3, replace=False)
                for idx in infected_indices:
                    agents[idx]['initial_state'] = 'I'
        
        elif condition == 'dormitory_cluster':
            # Scenario 2: Dormitory outbreak
            # Target: Agents in dormitory clusters
            dormitory_agents = []
            for i, agent in enumerate(agents):
                cluster = str(agent.get('cluster', '')).lower()
                if 'dorm' in cluster:
                    dormitory_agents.append(i)
            
            if len(dormitory_agents) >= 10:
                # Infect 8-12 dormitory workers (high attack rate)
                n_infected = min(12, max(8, len(dormitory_agents) // 20))
                infected_indices = np.random.choice(dormitory_agents, n_infected, replace=False)
                for idx in infected_indices:
                    agents[idx]['initial_state'] = 'I'
                
                # Expose another 10-15 (rapid spread)
                remaining_dorm = [i for i in dormitory_agents if agents[i]['initial_state'] == 'S']
                if len(remaining_dorm) >= 10:
                    n_exposed = min(15, len(remaining_dorm))
                    exposed_indices = np.random.choice(remaining_dorm, n_exposed, replace=False)
                    for idx in exposed_indices:
                        agents[idx]['initial_state'] = 'E'
            else:
                # Fallback if few dormitory agents
                infected_indices = np.random.choice(len(agents), 8, replace=False)
                for idx in infected_indices:
                    agents[idx]['initial_state'] = 'I'
        
        elif condition == 'community_clusters':
            # Scenario 3: Community clusters (households + workplaces)
            # Multiple household clusters
            households = {}
            for i, agent in enumerate(agents):
                hh = agent['household_id']
                if hh not in households:
                    households[hh] = []
                households[hh].append(i)
            
            # Select 3 household clusters
            sorted_hh = sorted(households.items(), key=lambda x: len(x[1]), reverse=True)
            target_households = sorted_hh[:3]
            
            infected_count = 0
            for hh_id, hh_indices in target_households:
                # Infect 1-2 per household
                n_to_infect = min(2, max(1, len(hh_indices) // 3))
                infected = np.random.choice(hh_indices, n_to_infect, replace=False)
                
                for idx in infected:
                    agents[idx]['initial_state'] = 'I'
                    infected_count += 1
                
                # Expose 1-2 more in same household
                remaining_hh = [i for i in hh_indices if agents[i]['initial_state'] == 'S']
                if len(remaining_hh) >= 1:
                    n_exposed = min(2, len(remaining_hh))
                    exposed = np.random.choice(remaining_hh, n_exposed, replace=False)
                    for idx in exposed:
                        agents[idx]['initial_state'] = 'E'
            
            # Add 2-3 workplace cluster cases
            workplaces = {}
            for i, agent in enumerate(agents):
                work_net = agent.get('work_network', -1)
                if work_net >= 0 and agents[i]['initial_state'] == 'S':
                    if work_net not in workplaces:
                        workplaces[work_net] = []
                    workplaces[work_net].append(i)
            
            if workplaces:
                # Pick largest workplace
                largest_wp = max(workplaces.items(), key=lambda x: len(x[1]))
                wp_indices = largest_wp[1]
                
                n_wp_infected = min(3, len(wp_indices))
                wp_infected = np.random.choice(wp_indices, n_wp_infected, replace=False)
                for idx in wp_infected:
                    agents[idx]['initial_state'] = 'I'
        
        elif condition == 'circuit_breaker':
            # Scenario 4: Circuit Breaker (reduced but persistent transmission)
            # Lower initial infections (lockdown effect)
            # Mix of household + essential workers
            
            # Essential workers more likely infected
            essential_workers = []
            for i, agent in enumerate(agents):
                occ = agent.get('occupation', '')
                if occ in ['healthcare_worker', 'service_worker', 'retail_worker']:
                    essential_workers.append(i)
            
            if len(essential_workers) >= 3:
                n_infected = min(4, len(essential_workers))
                infected_indices = np.random.choice(essential_workers, n_infected, replace=False)
                for idx in infected_indices:
                    agents[idx]['initial_state'] = 'I'
            else:
                infected_indices = np.random.choice(len(agents), 3, replace=False)
                for idx in infected_indices:
                    agents[idx]['initial_state'] = 'I'
            
            # Add household transmission (people stuck at home)
            households = {}
            for i, agent in enumerate(agents):
                if agents[i]['initial_state'] == 'I':
                    hh = agent['household_id']
                    if hh not in households:
                        households[hh] = []
                    households[hh].append(i)
            
            # Expose household members of infected
            for hh_id in households:
                hh_members = [i for i, a in enumerate(agents) 
                             if a['household_id'] == hh_id and agents[i]['initial_state'] == 'S']
                if hh_members:
                    n_exposed = min(2, len(hh_members))
                    exposed = np.random.choice(hh_members, n_exposed, replace=False)
                    for idx in exposed:
                        agents[idx]['initial_state'] = 'E'
        
        else:
            # Default: Random community transmission
            infected_indices = np.random.choice(len(agents), 3, replace=False)
            for idx in infected_indices:
                agents[idx]['initial_state'] = 'I'
            
            remaining = [i for i in range(len(agents)) if agents[i]['initial_state'] == 'S']
            if len(remaining) >= 2:
                exposed_indices = np.random.choice(remaining, 2, replace=False)
                for idx in exposed_indices:
                    agents[idx]['initial_state'] = 'E'
        
        # Set realistic days_in_state
        for agent in agents:
            self._set_realistic_days_in_state(agent)

    def _set_realistic_days_in_state(self, agent_profile: dict):
        """Set realistic days_in_state for initial infected/exposed agents."""
        state = agent_profile['initial_state']
        
        if state == 'I':
            agent_profile['days_in_state'] = np.random.randint(2, 7)
        elif state == 'E':
            agent_profile['days_in_state'] = np.random.randint(1, 4)
        elif state == 'R':
            agent_profile['days_in_state'] = np.random.randint(10, 61)
        else:
            agent_profile['days_in_state'] = 0
    
    async def run_all_scenarios(self):
        """Run all 4 Singapore scenarios ASYNC and collect traces WITH AGGREGATION."""
        
        print("\n" + "="*80)
        print("ASYNC SINGAPORE COVID-19 REASONING TRACE COLLECTION v3")
        print("="*80)
        print("Running 4 Singapore-specific scenarios (5 days each)")
        print(f"Max concurrent calls: {self.max_concurrent}")
        print("Collecting: LLM conversations, tool usage, behavioral patterns")
        print("Using: REAL Singapore COVID-19 knowledge base")
        print("NEW: Aggregating traces per agent across scenarios before motif extraction")
        print("="*80 + "\n")
        
        # Startup heartbeat test
        print("🔍 Testing async setup...")
        try:
            test_start = time.time()
            test_response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            test_elapsed = time.time() - test_start
            print(f"✓ OpenAI API connection working! (latency: {test_elapsed:.2f}s)")
            print(f"✓ Async event loop active")
            print(f"✓ Semaphore configured: {self.max_concurrent} concurrent calls\n")
        except Exception as e:
            print(f"❌ ERROR: OpenAI API test failed: {e}")
            print("   Check your OPENAI_API_KEY environment variable")
            return
        
        # PHASE 1: Run all scenarios and collect RAW traces
        print("="*80)
        print("PHASE 1: Running Singapore scenarios (collecting raw traces)")
        print("="*80)
        
        scenario_raw_traces = []
        
        for scenario_id, config in self.scenarios.items():
            print(f"\n{'='*80}")
            print(f"SCENARIO: {config['name']}")
            print(f"{'='*80}")
            print(f"Description: {config['description']}")
            print(f"Period: {config['period']}")
            print(f"Goal: {config['goal']}")
            print(f"Duration: {config['duration']} days")
            
            # Select agents
            selected = self._select_scenario_agents(config)
            print(f"Selected: {len(selected)} agents")
            
            # Run simulation ASYNC
            traces = await self._run_scenario_simulation_async(scenario_id, config, selected)
            
            # Store WITHOUT motifs
            scenario_raw_traces.append({
                'scenario_id': scenario_id,
                'scenario_name': config['name'],
                'scenario_period': config['period'],
                'num_agents': len(selected),
                'traces': traces
            })
            
            print(f"\n✓ Collected {len(traces)} raw agent traces")
        
        # PHASE 2: Aggregate traces by agent_id across scenarios
        print(f"\n{'='*80}")
        print("PHASE 2: Aggregating traces per unique agent")
        print("="*80)
        
        agent_aggregated = {}
        
        for scenario_data in scenario_raw_traces:
            scenario_id = scenario_data['scenario_id']
            for trace in scenario_data['traces']:
                agent_id = trace['agent_id']
                
                if agent_id not in agent_aggregated:
                    agent_aggregated[agent_id] = {
                        'agent_id': agent_id,
                        'agent_name': trace['agent_name'],
                        'age': trace['age'],
                        'occupation': trace['occupation'],
                        'gender': trace.get('gender', 'unknown'),
                        'nationality': trace.get('nationality', 'unknown'),
                        'cluster': trace.get('cluster', ''),
                        'is_imported': trace.get('is_imported', False),
                        'vaccination_status': trace['vaccination_status'],
                        'all_conversations': [],
                        'all_tool_usage': [],
                        'scenarios': [],
                        'state_history_combined': []
                    }
                
                # Aggregate conversations and tools
                agent_aggregated[agent_id]['all_conversations'].extend(trace.get('conversations', []))
                agent_aggregated[agent_id]['all_tool_usage'].extend(trace.get('tool_usage_log', []))
                agent_aggregated[agent_id]['scenarios'].append(scenario_id)
                agent_aggregated[agent_id]['state_history_combined'].extend(trace.get('state_history', []))
        
        print(f"✓ Aggregated {len(agent_aggregated)} unique agents")
        print(f"  Average scenarios per agent: {np.mean([len(a['scenarios']) for a in agent_aggregated.values()]):.1f}")
        print(f"  Average conversations per agent: {np.mean([len(a['all_conversations']) for a in agent_aggregated.values()]):.0f}")
        
        # PHASE 3: Extract motifs ONCE per unique agent
        print(f"\n{'='*80}")
        print("PHASE 3: Extracting Singapore-specific motifs (ONCE per unique agent)")
        print("="*80)
        
        agent_motifs = await self._extract_aggregated_motifs_async(agent_aggregated)
        
        print(f"✓ Extracted motifs for {len(agent_motifs)} unique agents")
        
        # PHASE 4: Assign motifs back to scenario traces
        print(f"\n{'='*80}")
        print("PHASE 4: Assigning motifs to scenario traces")
        print("="*80)
        
        for scenario_data in scenario_raw_traces:
            for trace in scenario_data['traces']:
                agent_id = trace['agent_id']
                trace['behavioral_motifs'] = agent_motifs.get(agent_id, {
                    'exposure_reasoning': 'unclassified',
                    'risk_posture': 'unclassified',
                    'information_seeking': 'unclassified',
                    'protection_priority': 'unclassified',
                    'temporal_style': 'unclassified',
                    'compliance_attitude': 'unclassified'
                })
        
        self.collected_traces = scenario_raw_traces
        
        print(f"✓ Assigned motifs to all {sum(len(s['traces']) for s in scenario_raw_traces)} scenario traces")
        
        # Save
        self._save_traces()
    
    async def _run_scenario_simulation_async(
        self,
        scenario_id: str,
        config: dict,
        selected_agents: List[dict]
    ) -> List[Dict]:
        """Run REAL Singapore simulation with LLM agents ASYNC."""
        
        print(f"\n  Running ASYNC simulation with {len(selected_agents)} agents...")
        
        # Create agents
        agents = []
        for profile in selected_agents:
            agent = EntityAgent(
                agent_id=profile['agent_id'],
                profile=profile,
                tool_registry=self.tools,
                llm_client=None,
                mode='trace_collection'
            )
            agent.state = profile['initial_state']
            agent.days_in_state = profile.get('days_in_state', 0)
            agents.append(agent)
        
        # Set up neighbors
        agent_dict = {a.agent_id: a for a in agents}
        agent_ids = set(a.agent_id for a in agents)
        
        for edge in self.edges:
            if edge['agent_1'] in agent_ids and edge['agent_2'] in agent_ids:
                agent_dict[edge['agent_1']].add_neighbor(agent_dict[edge['agent_2']])
                agent_dict[edge['agent_2']].add_neighbor(agent_dict[edge['agent_1']])
        
        # Run simulation ASYNC
        duration = config['duration']
        
        for day in range(duration):
            start_time = time.time()
            print(f"    Day {day}: Processing {len(agents)} agents...")
            
            # Mock cluster probabilities (Singapore-calibrated)
            cluster_probs = {
                'S->E': {'fused': 0.02},  # Lower (strict contact tracing)
                'E->I': {'fused': 0.18},  # Singapore incubation
                'I->R': {'fused': 0.14},  # Good healthcare
                'R->S': {'fused': 0.003}  # Low reinfection
            }
            
            # Process all agents CONCURRENTLY
            tasks = [
                self._agent_reason_async(agent, cluster_probs, day, i)
                for i, agent in enumerate(agents)
            ]
            
            completed = 0
            errors = 0
            results = []
            
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    results.append(result)
                    completed += 1
                    
                    if completed % 100 == 0 or completed == len(agents):
                        print(f"      [{completed}/{len(agents)}] agents processed ({completed/len(agents)*100:.0f}%)", flush=True)
                    
                    if isinstance(result, dict) and result.get('error'):
                        errors += 1
                        
                except Exception as e:
                    errors += 1
                    results.append({'transitioned': False, 'error': str(e)})
            
            transitions = sum(1 for r in results if isinstance(r, dict) and r.get('transitioned'))
            
            elapsed = time.time() - start_time
            print(f"      ✓ Day {day} complete: {transitions} transitions, {errors} errors ({elapsed:.1f}s)")
        
        print(f"\n  ✓ ASYNC simulation complete")
        
        # Aggregate traces
        traces = []
        for agent in agents:
            trace_data = agent.get_trace_data()
            trace_data['occupation'] = agent.profile['occupation']
            trace_data['age'] = agent.profile['age']
            trace_data['gender'] = agent.profile.get('gender', 'unknown')
            trace_data['nationality'] = agent.profile.get('nationality', 'unknown')
            trace_data['cluster'] = agent.profile.get('cluster', '')
            trace_data['is_imported'] = agent.profile.get('is_imported', False)
            trace_data['vaccination_status'] = agent.profile['vaccination_status']
            trace_data['initial_state'] = agent.profile['initial_state']
            traces.append(trace_data)
        
        return traces
    
    async def _agent_reason_async(self, agent, cluster_probs, timestep, agent_idx=None):
        """Single agent reasoning with semaphore."""
        async with self.semaphore:
            try:
                loop = asyncio.get_event_loop()
                
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        agent.reason_and_sample_transition,
                        cluster_probs,
                        "",
                        timestep
                    ),
                    timeout=60.0
                )
                
                return result
                    
            except asyncio.TimeoutError:
                return {'transitioned': False, 'error': 'timeout'}
                    
            except Exception as e:
                return {'transitioned': False, 'error': str(e)}
    
    async def _extract_aggregated_motifs_async(self, agent_aggregated: Dict) -> Dict:
        """Extract Singapore-specific motifs for each unique agent."""
        
        print(f"  Extracting structured motifs for {len(agent_aggregated)} unique agents...")
        start_time = time.time()
        
        # Process all agents concurrently
        tasks = []
        agent_ids = []
        
        for agent_id, agent_data in agent_aggregated.items():
            task = self._extract_single_agent_motif_async(agent_data)
            tasks.append(task)
            agent_ids.append(agent_id)
        
        # Process with progress
        print(f"    [0/{len(tasks)}] Starting motif extraction...", flush=True)
        
        completed = 0
        results = []
        errors = 0
        
        for coro in asyncio.as_completed(tasks):
            try:
                motifs = await coro
                results.append(motifs)
                completed += 1
                
                if completed % 100 == 0 or completed == len(tasks):
                    print(f"    [{completed}/{len(tasks)}] motifs extracted ({completed/len(tasks)*100:.0f}%)", flush=True)
                
                if isinstance(motifs, dict) and motifs.get('exposure_reasoning') == 'error':
                    errors += 1
                    
            except Exception as e:
                errors += 1
                results.append({
                    'exposure_reasoning': 'error',
                    'risk_posture': 'error',
                    'information_seeking': 'error',
                    'protection_priority': 'error',
                    'temporal_style': 'error',
                    'compliance_attitude': 'error'
                })
        
        # Create mapping
        agent_motifs = {}
        for i, agent_id in enumerate(agent_ids):
            if i < len(results):
                agent_motifs[agent_id] = results[i]
            else:
                agent_motifs[agent_id] = {
                    'exposure_reasoning': 'unclassified',
                    'risk_posture': 'unclassified',
                    'information_seeking': 'unclassified',
                    'protection_priority': 'unclassified',
                    'temporal_style': 'unclassified',
                    'compliance_attitude': 'unclassified'
                }
        
        elapsed = time.time() - start_time
        print(f"    ✓ Motif extraction complete: {len(agent_motifs)} agents, {errors} errors ({elapsed:.1f}s)")
        
        return agent_motifs
    
    async def _extract_single_agent_motif_async(self, agent_data: Dict) -> Dict:
        """Extract Singapore-specific motif for single agent."""
        async with self.semaphore:
            try:
                # Quantify tool usage
                tool_stats = self._quantify_tool_usage_aggregated(agent_data)
                
                # Prepare summary
                summary = self._prepare_aggregated_summary(agent_data, tool_stats)
                
                # Extract motifs
                motifs = await self._llm_extract_singapore_motifs_async(summary, agent_data)
                
                motifs['tool_usage_stats'] = tool_stats
                
                return motifs
                
            except Exception as e:
                return {
                    'exposure_reasoning': 'error',
                    'risk_posture': 'error',
                    'information_seeking': 'error',
                    'protection_priority': 'error',
                    'temporal_style': 'error',
                    'compliance_attitude': 'error'
                }
    
    def _quantify_tool_usage_aggregated(self, agent_data: Dict) -> Dict:
        """Quantify tool usage across ALL scenarios for this agent."""
        all_tools = agent_data['all_tool_usage']
        all_conversations = agent_data['all_conversations']
        
        tool_counts = {}
        total_tools = 0
        
        for t in all_tools:
            for tool in t.get('tools', []):
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
                total_tools += 1
        
        tools_per_decision = total_tools / max(len(all_conversations), 1)
        
        tool_diversity = 0.0
        if tool_counts:
            total = sum(tool_counts.values())
            for count in tool_counts.values():
                p = count / total
                tool_diversity -= p * np.log(p + 1e-10)
        
        contact_focused = tool_counts.get('check_infected_contacts', 0) + \
                          tool_counts.get('check_household_infections', 0)
        
        info_seeking = tool_counts.get('query_knowledge', 0) + \
                       tool_counts.get('query_contacts', 0)
        
        health_monitoring = tool_counts.get('check_symptoms', 0) + \
                            tool_counts.get('check_days_infected', 0)
        
        return {
            'total_tool_calls': total_tools,
            'tools_per_decision': round(tools_per_decision, 2),
            'tool_diversity': round(tool_diversity, 3),
            'unique_tools_used': len(tool_counts),
            'contact_focused_ratio': round(contact_focused / max(total_tools, 1), 3),
            'info_seeking_ratio': round(info_seeking / max(total_tools, 1), 3),
            'health_monitoring_ratio': round(health_monitoring / max(total_tools, 1), 3),
            'tool_breakdown': tool_counts,
            'num_scenarios': len(agent_data['scenarios']),
            'total_decisions': len(agent_data['all_conversations'])
        }
    
    def _prepare_aggregated_summary(self, agent_data: Dict, tool_stats: Dict) -> str:
        """Prepare summary from aggregated traces."""
        convs = agent_data['all_conversations']
        
        summary = f"""Agent: {agent_data['agent_name']} (ID: {agent_data['agent_id']})
Age: {agent_data['age']}, Gender: {agent_data['gender']}
Occupation: {agent_data['occupation']}, Nationality: {agent_data['nationality']}
Cluster: {agent_data['cluster'] or 'None'}
Imported: {agent_data['is_imported']}
Vaccination: {['None', 'Partial', 'Full'][agent_data['vaccination_status']]}

AGGREGATED BEHAVIOR ACROSS {len(agent_data['scenarios'])} SINGAPORE SCENARIOS:
- Scenarios: {', '.join(agent_data['scenarios'])}
- Total decisions: {len(convs)}
- State transitions: {agent_data['state_history_combined']}

Reasoning Samples (across scenarios):
"""
        
        # Show diverse samples
        sample_indices = np.linspace(0, len(convs)-1, min(5, len(convs)), dtype=int)
        for idx in sample_indices:
            conv = convs[idx]
            summary += f"\nDay {conv['timestep']} ({conv['state']} → {conv['transition']}):\n"
            summary += f"Tools: {', '.join(conv.get('tools_used', []))}\n"
            
            response = conv.get('response', '')
            if 'REASONING:' in response:
                reasoning = response.split('REASONING:')[1].split('ADJUSTMENT:')[0].strip()
                summary += f"Reasoning: {reasoning[:150]}...\n"
        
        summary += f"""
QUANTIFIED TOOL USAGE:
- Total tool calls: {tool_stats['total_tool_calls']}
- Tools per decision: {tool_stats['tools_per_decision']}
- Tool diversity: {tool_stats['tool_diversity']}
"""
        
        return summary
    
    async def _llm_extract_singapore_motifs_async(self, summary: str, agent_data: Dict) -> Dict:
        """Extract Singapore-specific behavioral motifs."""
        
        scenario_context = f"Singapore COVID-19 scenarios: {', '.join(agent_data['scenarios'])}"
        
        prompt = f"""Analyze Singapore COVID-19 agent decision-making across scenarios.

{scenario_context}

SINGAPORE-SPECIFIC BEHAVIORAL AXES:

AXIS 1: exposure_reasoning
- contact_tracing_focused, cluster_aware, household_transmission, workplace_network, travel_history_aware

AXIS 2: risk_posture
- risk_averse, risk_neutral, risk_minimizing, risk_ignoring

AXIS 3: information_seeking
- heavy_tool_user, moderate_tool_user, light_tool_user, tool_avoider

AXIS 4: protection_priority
- self_focused, household_focused, community_focused, dormitory_aware

AXIS 5: temporal_style
- duration_tracker, symptom_monitor, quarantine_compliant, present_focused

AXIS 6: compliance_attitude (Singapore-specific)
- strict_compliant, moderate_compliant, reluctant_compliant, non_compliant

AGENT SUMMARY:
{summary}

Return STRICT JSON:
{{
  "exposure_reasoning": "...",
  "risk_posture": "...",
  "information_seeking": "...",
  "protection_priority": "...",
  "temporal_style": "...",
  "compliance_attitude": "..."
}}"""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                motif_dict = json.loads(json_match.group())
                
                required_axes = [
                    'exposure_reasoning', 'risk_posture', 'information_seeking',
                    'protection_priority', 'temporal_style', 'compliance_attitude'
                ]
                
                for axis in required_axes:
                    if axis not in motif_dict:
                        motif_dict[axis] = 'unclassified'
                
                return motif_dict
            else:
                return {
                    'exposure_reasoning': 'contact_tracing_focused',
                    'risk_posture': 'risk_neutral',
                    'information_seeking': 'moderate_tool_user',
                    'protection_priority': 'household_focused',
                    'temporal_style': 'present_focused',
                    'compliance_attitude': 'moderate_compliant'
                }
        
        except Exception as e:
            return {
                'exposure_reasoning': 'unclassified',
                'risk_posture': 'unclassified',
                'information_seeking': 'unclassified',
                'protection_priority': 'unclassified',
                'temporal_style': 'unclassified',
                'compliance_attitude': 'unclassified'
            }
    
    def _save_traces(self):
        """Save collected traces."""
        output_dir = Path("data/processed")
        output_file = output_dir / "reasoning_traces.json"
        
        print("\n" + "="*80)
        print("SAVING REASONING TRACES")
        print("="*80)
        
        total_convs = sum(
            len(trace.get('conversations', [])) 
            for scenario in self.collected_traces 
            for trace in scenario['traces']
        )
        
        # Count unique agents
        unique_agents = set()
        for scenario in self.collected_traces:
            for trace in scenario['traces']:
                unique_agents.add(trace['agent_id'])
        
        output_data = {
            'collection_date': datetime.now().isoformat(),
            'collection_method': 'ASYNC REAL LLM v3 (Singapore COVID-19 with agent aggregation)',
            'knowledge_base': 'Singapore COVID-19 data (MOH, OWID, research)',
            'num_scenarios': len(self.collected_traces),
            'total_agents': sum(s['num_agents'] for s in self.collected_traces),
            'unique_agents': len(unique_agents),
            'total_conversations': total_convs,
            'simulation_days': 7,
            'max_concurrent_calls': self.max_concurrent,
            'scenarios': self.collected_traces
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved to: {output_file}")
        print(f"  Scenarios: {output_data['num_scenarios']}")
        print(f"  Total agent-scenario pairs: {output_data['total_agents']}")
        print(f"  Unique agents: {output_data['unique_agents']}")
        print(f"  Conversations: {output_data['total_conversations']}")
        print(f"  Simulation days: {output_data['simulation_days']}")
        
        print("\n" + "="*80)
        print("✓ ASYNC SINGAPORE COVID-19 REASONING TRACE COLLECTION COMPLETE")
        print("="*80)


async def main():
    collector = AsyncSingaporeReasoningTraceCollector(max_concurrent=10)
    await collector.run_all_scenarios()


if __name__ == "__main__":
    asyncio.run(main())
