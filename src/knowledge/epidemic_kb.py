"""
Validation utilities for extracted Singapore COVID-19 data.

VALIDATES initial states from Kaggle day_0 column.
Initial states reflect actual epidemic status on January 23, 2020 (simulation start).
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict


class DataValidator:
    """Validate extracted agent and network data."""
    
    @staticmethod
    def validate_profiles(profiles: List[Dict]) -> bool:
        """
        Validate profile data structure and consistency.
        
        Initial states come from Kaggle day_0 column (ground truth).
        """
        print("\nValidating profiles...")
        
        required_fields = [
            'agent_id', 'name', 'age', 'occupation', 'household_id',
            'vaccination_status', 'comorbidity_count', 'disease_status',
            'case_number', 'gender', 'nationality', 'confirmation_date',
            'days_since_start', 'is_imported', 'cluster'
        ]
        
        # Check all profiles have required fields
        missing_count = 0
        for p in profiles:
            for field in required_fields:
                if field not in p:
                    print(f"  ⚠️  Missing field: {field} in agent {p.get('agent_id', '?')}")
                    missing_count += 1
        
        if missing_count > 0:
            assert False, f"Missing {missing_count} required fields"
        
        print(f"  ✅ All required fields present")
        
        # Check unique IDs
        ids = [p['agent_id'] for p in profiles]
        assert len(ids) == len(set(ids)), "Duplicate agent IDs detected"
        assert min(ids) == 0, "Agent IDs should start at 0"
        assert max(ids) == len(ids) - 1, "Agent IDs should be sequential"
        print(f"  ✅ Agent IDs are sequential: 0 to {len(ids)-1}")
        
        # Convert disease_status to state letters
        def status_to_state(status: int) -> str:
            return {0: 'S', 1: 'E', 2: 'I', 3: 'R', 4: 'D'}[status]
        
        # Check state distribution (from Kaggle day_0)
        states = Counter(status_to_state(p['disease_status']) for p in profiles)
        total = len(profiles)
        
        print(f"\n  Initial state distribution (from Kaggle day_0):")
        for state in ['S', 'E', 'I', 'R', 'D']:
            count = states.get(state, 0)
            pct = 100 * count / total
            print(f"    {state}: {count:4d} ({pct:5.1f}%)")
        
        # Extract counts
        s_count = states.get('S', 0)
        e_count = states.get('E', 0)
        i_count = states.get('I', 0)
        r_count = states.get('R', 0)
        d_count = states.get('D', 0)
        
        # Validation: Check states are valid
        invalid_states = total - (s_count + e_count + i_count + r_count + d_count)
        assert invalid_states == 0, f"Found {invalid_states} agents with invalid states"
        
        # Check infected seed (CRITICAL for epidemic spread)
        infected_count = e_count + i_count
        infected_pct = 100 * infected_count / total
        
        if infected_count == 0:
            print(f"\n  ❌ CRITICAL: No infected agents (E or I) at simulation start!")
            print(f"     Epidemic cannot spread with zero initial infections")
            print(f"     Check Kaggle day_0 column has proper state data")
            return False
        elif infected_count < 5:
            print(f"\n  ⚠️  WARNING: Very few infected agents: {infected_count} ({infected_pct:.1f}%)")
            print(f"     Epidemic may spread slowly")
        else:
            print(f"\n  ✅ Infected seed adequate: {infected_count} agents ({infected_pct:.1f}%)")
            print(f"     E: {e_count}, I: {i_count}")
        
        # Report distribution characteristics
        if s_count < 0.5 * total:
            print(f"  ℹ️  Less than 50% susceptible - epidemic well underway")
        
        if r_count > 0.1 * total:
            print(f"  ℹ️  More than 10% recovered - later stage epidemic")
        
        if d_count > 0:
            print(f"  ℹ️  {d_count} deaths recorded at simulation start")
        
        # Check days_since_start
        days = [p['days_since_start'] for p in profiles]
        print(f"\n  Days since start (confirmation date):")
        print(f"    Min: {min(days)}, Max: {max(days)}, Mean: {np.mean(days):.1f}, Median: {np.median(days):.1f}")
        
        # Check chronological ordering
        non_monotonic = sum(1 for i in range(1, len(days)) if days[i] < days[i-1])
        if non_monotonic > 0:
            print(f"  ⚠️  {non_monotonic} cases out of order (not perfectly chronological)")
        else:
            print(f"  ✅ Cases are chronologically ordered")
        
        # Demographics
        ages = [p['age'] for p in profiles]
        print(f"\n  ✓ Age: range={min(ages)}-{max(ages)}, mean={np.mean(ages):.1f}, median={np.median(ages):.1f}")
        
        genders = Counter(p['gender'] for p in profiles)
        print(f"  ✓ Gender distribution:")
        for gender, count in sorted(genders.items(), key=lambda x: -x[1]):
            print(f"    {gender}: {count} ({100*count/total:.1f}%)")
        
        nationalities = Counter(p['nationality'] for p in profiles)
        print(f"  ✓ Top 5 nationalities:")
        for nat, count in nationalities.most_common(5):
            print(f"    {nat}: {count} ({100*count/total:.1f}%)")
        
        imported_count = sum(1 for p in profiles if p['is_imported'])
        print(f"  ✓ Imported cases: {imported_count} ({100*imported_count/total:.1f}%)")
        
        # Clusters
        clusters = [p['cluster'] for p in profiles 
                   if p['cluster'] and str(p['cluster']).lower() not in ['', 'nan', 'none']]
        if clusters:
            unique_clusters = len(set(clusters))
            print(f"  ✓ Clusters: {unique_clusters} unique, {len(clusters)} agents ({100*len(clusters)/total:.1f}%)")
            
            cluster_dist = Counter(clusters)
            print(f"    Top 5 clusters:")
            for cluster, count in cluster_dist.most_common(5):
                cluster_str = str(cluster)[:50]
                print(f"      {cluster_str}: {count} agents")
        else:
            print(f"  ℹ️  No cluster assignments")
        
        # Households
        households = len(set(p['household_id'] for p in profiles))
        avg_household = total / households if households > 0 else 0
        print(f"  ✓ Households: {households} (avg size: {avg_household:.1f})")
        
        # Vaccination (should be 0 for early 2020)
        vacc_dist = Counter(p['vaccination_status'] for p in profiles)
        print(f"  ✓ Vaccination: {dict(sorted(vacc_dist.items()))}")
        if vacc_dist.get(0, 0) < total * 0.95:
            print(f"    ⚠️  Expected ~100% unvaccinated for early 2020")
        
        # Occupation
        occ_dist = Counter(p['occupation'] for p in profiles)
        print(f"  ✓ Top 5 occupations:")
        for occ, count in occ_dist.most_common(5):
            print(f"    {occ}: {count} ({100*count/total:.1f}%)")
        
        # Comorbidities
        comorbid_dist = Counter(p['comorbidity_count'] for p in profiles)
        avg_comorbid = np.mean([p['comorbidity_count'] for p in profiles])
        print(f"  ✓ Comorbidities: mean={avg_comorbid:.2f}, dist={dict(sorted(comorbid_dist.items()))}")
        
        # Scores
        mobility = [p['mobility_score'] for p in profiles]
        compliance = [p['compliance_score'] for p in profiles]
        print(f"  ✓ Mobility: mean={np.mean(mobility):.2f}, std={np.std(mobility):.2f}")
        print(f"  ✓ Compliance: mean={np.mean(compliance):.2f}, std={np.std(compliance):.2f}")
        
        return True
    
    @staticmethod
    def validate_network(edges: List[Dict], n_agents: int) -> bool:
        """Validate network structure."""
        print("\nValidating network...")
        
        if len(edges) == 0:
            print(f"  ❌ CRITICAL: No edges in network!")
            return False
        
        # Check edge structure
        invalid_edges = 0
        for i, e in enumerate(edges):
            if 'agent_1' not in e or 'agent_2' not in e:
                invalid_edges += 1
            elif 'edge_type' not in e:
                invalid_edges += 1
            elif 'contact_rate' not in e:
                invalid_edges += 1
            elif not (0 <= e['agent_1'] < n_agents):
                invalid_edges += 1
            elif not (0 <= e['agent_2'] < n_agents):
                invalid_edges += 1
            elif e['agent_1'] >= e['agent_2']:
                invalid_edges += 1
            elif not (0 <= e.get('contact_rate', -1) <= 1):
                invalid_edges += 1
        
        if invalid_edges > 0:
            assert False, f"{invalid_edges} edges have validation errors"
        
        print(f"  ✅ All {len(edges)} edges are structurally valid")
        
        # Degree distribution
        degree = defaultdict(int)
        for e in edges:
            degree[e['agent_1']] += 1
            degree[e['agent_2']] += 1
        
        # Check connectivity
        all_agents = set(range(n_agents))
        connected_agents = set(degree.keys())
        isolated = all_agents - connected_agents
        
        if isolated:
            print(f"  ❌ Found {len(isolated)} isolated agents:")
            print(f"     {list(isolated)[:20]}{'...' if len(isolated) > 20 else ''}")
            assert False, f"{len(isolated)} agents are not connected"
        else:
            print(f"  ✅ All {n_agents} agents are connected")
        
        # Statistics
        degrees = list(degree.values())
        print(f"\n  Network statistics:")
        print(f"    Edges: {len(edges)}")
        print(f"    Degree distribution:")
        print(f"      Min: {min(degrees)}")
        print(f"      Q1:  {np.percentile(degrees, 25):.1f}")
        print(f"      Median: {np.median(degrees):.1f}")
        print(f"      Mean: {np.mean(degrees):.1f}")
        print(f"      Q3:  {np.percentile(degrees, 75):.1f}")
        print(f"      Max: {max(degrees)}")
        print(f"      Std: {np.std(degrees):.1f}")
        
        assert min(degrees) >= 1, "Isolated agents detected"
        
        # Check density
        avg_degree = np.mean(degrees)
        if avg_degree > 30:
            print(f"  ⚠️  Very dense network (avg degree: {avg_degree:.1f})")
        elif avg_degree < 3:
            print(f"  ⚠️  Very sparse network (avg degree: {avg_degree:.1f})")
        else:
            print(f"  ✅ Network density appropriate (avg degree: {avg_degree:.1f})")
        
        # Edge types
        edge_types = Counter(e['edge_type'] for e in edges)
        print(f"\n  Edge type distribution:")
        for edge_type, count in sorted(edge_types.items(), key=lambda x: -x[1]):
            print(f"    {edge_type}: {count} ({100*count/len(edges):.1f}%)")
        
        expected_types = {'household', 'contact', 'cluster', 'work', 'school', 'community'}
        actual_types = set(edge_types.keys())
        unexpected = actual_types - expected_types
        if unexpected:
            print(f"  ⚠️  Unexpected edge types: {unexpected}")
        
        # Contact rates
        contact_rates = [e['contact_rate'] for e in edges]
        print(f"\n  Contact rate distribution:")
        print(f"    Min: {min(contact_rates):.2f}")
        print(f"    Median: {np.median(contact_rates):.2f}")
        print(f"    Mean: {np.mean(contact_rates):.2f}")
        print(f"    Max: {max(contact_rates):.2f}")
        
        # Check household edges
        household_edges = [e for e in edges if e['edge_type'] == 'household']
        if household_edges:
            household_rates = [e['contact_rate'] for e in household_edges]
            min_rate = min(household_rates)
            if min_rate < 0.7:
                print(f"  ⚠️  Some household edges have low contact rates (min: {min_rate:.2f})")
            else:
                print(f"  ✅ Household edges have appropriate rates (min: {min_rate:.2f})")
        
        # Check connectivity (BFS)
        print(f"\n  Checking full connectivity...")
        visited = set()
        queue = [0]
        visited.add(0)
        
        while queue:
            node = queue.pop(0)
            for e in edges:
                if e['agent_1'] == node and e['agent_2'] not in visited:
                    visited.add(e['agent_2'])
                    queue.append(e['agent_2'])
                elif e['agent_2'] == node and e['agent_1'] not in visited:
                    visited.add(e['agent_1'])
                    queue.append(e['agent_1'])
        
        if len(visited) < n_agents:
            print(f"  ⚠️  Network has isolated components: {n_agents - len(visited)} unreachable")
        else:
            print(f"  ✅ Network is fully connected")
        
        return True
    
    @staticmethod
    def validate_all(profiles: List[Dict], edges: List[Dict]) -> bool:
        """Run all validation checks."""
        print("="*80)
        print("VALIDATING SINGAPORE COVID-19 DATA")
        print("Initial states from Kaggle day_0 column (2020-01-23)")
        print("="*80)
        
        n_agents = len(profiles)
        print(f"\nDataset: {n_agents} agents, {len(edges)} edges")
        
        try:
            profiles_ok = DataValidator.validate_profiles(profiles)
            if not profiles_ok:
                return False
            
            network_ok = DataValidator.validate_network(edges, n_agents)
            if not network_ok:
                return False
            
            print("\n" + "="*80)
            print("✅ ALL VALIDATIONS PASSED")
            print("="*80)
            print("\nData ready for:")
            print("  1. Neural training (cluster_data_generator.py)")
            print("  2. Simulation (run_rolling_window_simulation.py)")
            return True
            
        except AssertionError as e:
            print(f"\n" + "="*80)
            print(f"❌ VALIDATION FAILED")
            print("="*80)
            print(f"Error: {e}")
            return False
            
        except Exception as e:
            print(f"\n" + "="*80)
            print(f"❌ UNEXPECTED ERROR")
            print("="*80)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def validate_singapore_data(profiles: List[Dict], edges: List[Dict]) -> bool:
    """
    Validate Singapore COVID-19 data.
    
    Args:
        profiles: Agent profiles (with disease_status from day_0)
        edges: Network edges
        
    Returns:
        True if valid, False otherwise
    """
    return DataValidator.validate_all(profiles, edges)


if __name__ == "__main__":
    import json
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python data_validator.py <profiles.json> <edges.json>")
        print("\nExample:")
        print("  python data_validator.py data/openabm/profiles.json data/openabm/edges.json")
        sys.exit(1)
    
    profiles_path = sys.argv[1]
    edges_path = sys.argv[2]
    
    print(f"Loading data from:")
    print(f"  Profiles: {profiles_path}")
    print(f"  Edges: {edges_path}")
    
    with open(profiles_path, 'r') as f:
        profiles = json.load(f)
    
    with open(edges_path, 'r') as f:
        edges = json.load(f)
    
    success = validate_singapore_data(profiles, edges)
    
    sys.exit(0 if success else 1)
