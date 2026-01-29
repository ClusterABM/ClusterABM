"""
Extract agent profiles and contact networks from Singapore COVID-19 data.
Converts real case data to StateAgentNet format with inferred contact networks.

Data sources:
1. Kaggle Singapore COVID-19 cases (~3,252 cases)
   - MUST have day_0 column with actual SEIRD states on 2020-01-23
2. MOH aggregate time-series data

UPDATED: Uses first 1000 agents CHRONOLOGICALLY + reads day_0 states from Kaggle
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import random
from typing import List, Dict, Tuple, Set
from datetime import datetime, timedelta
import re


class SingaporeDataExtractor:
    """
    Extract agent profiles and contact networks from Singapore COVID-19 data.
    
    KEY FEATURES:
    - Use first 1,000 agents CHRONOLOGICALLY (not random)
    - Read initial states from Kaggle day_0 column (actual ground truth)
    - Infer contact network from cluster assignments and link descriptions
    - Handle sparse, realistic contact networks
    """
    
    def __init__(
        self,
        case_file_path: str,
        n_agents: int = 1000,
        simulation_start_date: str = "2020-01-23",  # First case in Singapore
        random_seed: int = 42
    ):
        """
        Initialize Singapore data extractor.
        
        Args:
            case_file_path: Path to Kaggle Singapore COVID-19 CSV (with day_0 column)
            n_agents: Number of agents to use (default: 1000, taken chronologically)
            simulation_start_date: Start date for simulation (ISO format)
            random_seed: Random seed for reproducibility
        """
        self.case_file_path = case_file_path
        self.n_agents = n_agents
        self.simulation_start_date = pd.to_datetime(simulation_start_date)
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        print(f"Initializing Singapore COVID-19 extractor")
        print(f"  Target agents: {n_agents} (chronological)")
        print(f"  Simulation start: {simulation_start_date}")
        print(f"  Random seed: {random_seed}")
        
        # Load data
        self.df_cases = self._load_case_data()
        print(f"  Loaded {len(self.df_cases)} cases from Singapore")
    
    def _load_case_data(self) -> pd.DataFrame:
        """Load and preprocess Kaggle Singapore case data with day_0 column."""
        print("\nLoading case data...")
        
        try:
            df = pd.read_csv(self.case_file_path)
            print(f"  ✓ Loaded {len(df)} cases")
        except Exception as e:
            raise FileNotFoundError(f"Could not load case file: {e}")
        
        # Standardize column names (case-insensitive)
        df.columns = df.columns.str.lower().str.strip()
        
        # Expected columns (based on Kaggle dataset)
        required_cols = ['case', 'date', 'age', 'gender', 'nationality']
        
        # CHECK FOR day_0 COLUMN (CRITICAL!)
        if 'day_0' not in df.columns:
            raise ValueError(
                "ERROR: Kaggle CSV must have 'day_0' column with initial states!\n"
                "Expected columns: case, date, age, gender, nationality, day_0, ...\n"
                f"Found columns: {list(df.columns)}"
            )
        
        print(f"  ✅ Found day_0 column for initial states")
        
        # Validate required columns
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        print(f"  Available columns: {list(df.columns)[:15]}...")
        
        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'date discharged' in df.columns:
            df['date_discharged'] = pd.to_datetime(df['date discharged'], errors='coerce')
        
        # Handle missing values
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['gender'] = df['gender'].fillna('unknown')
        df['nationality'] = df['nationality'].fillna('unknown')
        
        if 'imported' in df.columns:
            df['imported'] = df['imported'].fillna('')
        if 'cluster_local' in df.columns:
            df['cluster_local'] = df['cluster_local'].fillna('')
        if 'link' in df.columns:
            df['link'] = df['link'].fillna('')
        
        # Process day_0 column (initial states)
        df['day_0'] = df['day_0'].fillna('S')  # Default to S if missing
        df['day_0'] = df['day_0'].astype(str).str.upper().str.strip()
        
        # Validate states in day_0
        valid_states = {'S', 'E', 'I', 'R', 'D'}
        invalid_states = set(df['day_0'].unique()) - valid_states
        if invalid_states:
            print(f"  ⚠️  Found invalid states in day_0: {invalid_states}")
            print(f"      Replacing with 'S'")
            df.loc[~df['day_0'].isin(valid_states), 'day_0'] = 'S'
        
        # Print day_0 distribution
        day0_dist = df['day_0'].value_counts()
        print(f"\n  day_0 state distribution (full dataset):")
        for state in ['S', 'E', 'I', 'R', 'D']:
            count = day0_dist.get(state, 0)
            pct = 100 * count / len(df)
            print(f"    {state}: {count:4d} ({pct:5.1f}%)")
        
        # Drop cases with missing critical data
        df = df.dropna(subset=['case', 'date', 'age'])
        
        # Sort by date (CRITICAL for chronological selection)
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"\n  ✓ Cleaned data: {len(df)} valid cases")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Age range: {df['age'].min():.0f} to {df['age'].max():.0f}")
        
        return df
    
    def extract_all_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract profiles and network in one go.
        
        Returns:
            (profiles, edges) tuple
        """
        print("\n" + "="*80)
        print("EXTRACTING SINGAPORE COVID-19 AGENT DATA")
        print("Initial states from Kaggle day_0 column (ground truth)")
        print("="*80)
        
        # Step 1: Select agents (chronologically)
        sampled_cases = self._sample_agents()
        
        # Step 2: Create agent profiles
        profiles = self._create_agent_profiles(sampled_cases)
        
        # Step 3: Infer contact network
        edges = self._infer_contact_network(sampled_cases, profiles)
        
        # Step 4: Validate
        self._validate_data(profiles, edges)
        
        return profiles, edges
    
    def _sample_agents(self) -> pd.DataFrame:
        """
        Take first n_agents CHRONOLOGICALLY from dataset.
        
        CRITICAL: Uses earliest cases for proper epidemic progression.
        """
        print(f"\nSelecting first {self.n_agents} agents chronologically...")
        
        df = self.df_cases.copy()
        
        # Take first n_agents (already sorted by date)
        if len(df) >= self.n_agents:
            sampled = df.iloc[:self.n_agents].copy()
        else:
            print(f"  ⚠️ Dataset has only {len(df)} cases, using all")
            sampled = df.copy()
        
        # Print day_0 distribution for selected agents
        day0_dist = sampled['day_0'].value_counts()
        print(f"\n  day_0 state distribution (selected {len(sampled)} agents):")
        for state in ['S', 'E', 'I', 'R', 'D']:
            count = day0_dist.get(state, 0)
            pct = 100 * count / len(sampled)
            print(f"    {state}: {count:4d} ({pct:5.1f}%)")
        
        # Check infected seed
        infected = day0_dist.get('E', 0) + day0_dist.get('I', 0)
        if infected == 0:
            print(f"  ❌ CRITICAL: No infected agents in day_0!")
            print(f"     Check Kaggle data day_0 column")
        elif infected < 10:
            print(f"  ⚠️  WARNING: Only {infected} infected agents")
        else:
            print(f"  ✅ Infected seed: {infected} agents")
        
        # Calculate statistics
        sampled['age_group'] = pd.cut(
            sampled['age'],
            bins=[0, 18, 30, 40, 50, 60, 70, 120],
            labels=['0-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70+']
        )
        sampled['is_imported'] = sampled['imported'].apply(
            lambda x: 'imported' if x else 'local'
        )
        
        print(f"\n  ✓ Selected {len(sampled)} agents")
        print(f"  Date range: {sampled['date'].min()} to {sampled['date'].max()}")
        
        return sampled
    
    def _create_agent_profiles(self, sampled_cases: pd.DataFrame) -> List[Dict]:
        """
        Create agent profiles from sampled cases.
        
        Maps real case data to StateAgentNet format.
        Reads initial states from day_0 column.
        """
        print("\nCreating agent profiles...")
        
        profiles = []
        
        for idx, row in sampled_cases.iterrows():
            # Calculate days since start
            days_since_start = (row['date'] - self.simulation_start_date).days
            
            # READ INITIAL STATE FROM day_0 COLUMN (ground truth!)
            initial_state = str(row['day_0']).upper().strip()
            if initial_state not in ['S', 'E', 'I', 'R', 'D']:
                print(f"  ⚠️  Invalid state '{initial_state}' for agent {idx}, using 'S'")
                initial_state = 'S'
            
            # Occupation (inferred from age)
            occupation = self._infer_occupation(row['age'])
            
            # Household ID (inferred from clusters or age-based)
            household_id = self._infer_household(row, idx)
            
            # Work/school network (inferred from age and clusters)
            work_network, school_network = self._infer_work_school_network(row, idx)
            
            # Vaccination status (all 0 for early 2020)
            vaccination_status = 0  # No vaccines in early 2020
            
            # Comorbidities (inferred from age)
            comorbidity_count = self._determine_comorbidities(row['age'])
            
            profile = {
                'agent_id': idx,
                'case_number': int(row['case']),
                'name': self._generate_name(row, idx),
                'age': int(row['age']),
                'gender': row['gender'],
                'nationality': row['nationality'],
                'occupation': occupation,
                'household_id': household_id,
                'work_network': work_network,
                'school_network': school_network,
                'region': 0,
                'vaccination_status': vaccination_status,
                'comorbidity_count': comorbidity_count,
                'mobility_score': self._calculate_mobility(row),
                'compliance_score': np.random.uniform(0.4, 1.0),
                'risk_awareness': np.random.uniform(0.3, 0.9),
                'disease_status': self._state_to_status(initial_state),  # For simulation
                'confirmation_date': row['date'].strftime('%Y-%m-%d'),
                'days_since_start': days_since_start,
                'discharge_date': row.get('date_discharged', pd.NaT),
                'is_imported': bool(row.get('imported', '')),
                'cluster': row.get('cluster_local', ''),
                'quarantine_status': False,
                'test_result': 'positive',
                'hospitalized': True if pd.notna(row.get('hospital')) else False,
                'icu': False,
            }
            
            profiles.append(profile)
        
        print(f"✓ Created {len(profiles)} agent profiles")
        
        # Print state distribution
        state_dist = defaultdict(int)
        for p in profiles:
            state = self._status_to_state(p['disease_status'])
            state_dist[state] += 1
        
        print(f"\n  Initial state distribution (from day_0):")
        for state in ['S', 'E', 'I', 'R', 'D']:
            count = state_dist[state]
            pct = 100 * count / len(profiles)
            print(f"    {state}: {count:4d} ({pct:5.1f}%)")
        
        # Verify infected seed
        infected_count = state_dist['E'] + state_dist['I']
        print(f"\n  ✓ Total infected at start: {infected_count} ({100*infected_count/len(profiles):.1f}%)")
        
        if infected_count == 0:
            print(f"  ❌ CRITICAL: No infected agents - epidemic cannot spread!")
        elif infected_count < 10:
            print(f"  ⚠️  WARNING: Very few infected agents - epidemic may spread slowly")
        
        return profiles
    
    def _state_to_status(self, state: str) -> int:
        """Convert SEIRD state to disease_status number."""
        state_map = {'S': 0, 'E': 1, 'I': 2, 'R': 3, 'D': 4}
        return state_map.get(state, 0)
    
    def _status_to_state(self, status: int) -> str:
        """Convert disease_status number to SEIRD state."""
        status_map = {0: 'S', 1: 'E', 2: 'I', 3: 'R', 4: 'D'}
        return status_map.get(status, 'S')
    
    def _infer_occupation(self, age: float) -> str:
        """Infer occupation from age."""
        age = int(age)
        if age < 5:
            return 'preschool'
        elif age < 18:
            return 'student'
        elif age < 25:
            return np.random.choice(['student', 'service_worker', 'retail_worker'], 
                                   p=[0.5, 0.3, 0.2])
        elif age < 65:
            occupations = [
                'healthcare_worker', 'teacher', 'office_worker',
                'retail_worker', 'service_worker', 'manual_worker',
                'professional', 'unemployed'
            ]
            probs = [0.12, 0.08, 0.25, 0.15, 0.15, 0.12, 0.10, 0.03]
            return np.random.choice(occupations, p=probs)
        else:
            return 'retired'
    
    def _infer_household(self, row: pd.Series, agent_idx: int) -> int:
        """Infer household ID from cluster information."""
        cluster = str(row.get('cluster_local', '')).lower()
        
        # Family clusters
        if 'family' in cluster:
            match = re.search(r'family of (\d+)', cluster)
            if match:
                return 10000 + int(match.group(1))
        
        # Dormitory workers
        if 'dorm' in cluster:
            return 20000 + agent_idx
        
        # Default: households of size 2-5
        household_size = (agent_idx % 4) + 2
        household_id = agent_idx // household_size
        
        return household_id
    
    def _infer_work_school_network(self, row: pd.Series, agent_idx: int) -> Tuple[int, int]:
        """Infer work/school networks from age and cluster information."""
        age = row['age']
        cluster = str(row.get('cluster_local', '')).lower()
        
        work_network = -1
        school_network = -1
        
        # School
        if 5 <= age < 18:
            school_network = agent_idx // 30
        elif 18 <= age < 25:
            if np.random.random() < 0.6:
                school_network = agent_idx // 40
        
        # Work
        if 18 <= age < 65:
            if 'hospital' in cluster or 'healthcare' in cluster:
                work_network = 500 + hash(row.get('hospital', '')) % 50
            elif 'dorm' in cluster:
                work_network = 1000 + hash(cluster) % 100
            else:
                work_network = agent_idx // 20
        
        return work_network, school_network
    
    def _determine_comorbidities(self, age: float) -> int:
        """Infer comorbidity count from age."""
        age = int(age)
        if age < 30:
            return np.random.choice([0, 1], p=[0.90, 0.10])
        elif age < 50:
            return np.random.choice([0, 1, 2], p=[0.70, 0.22, 0.08])
        elif age < 70:
            return np.random.choice([0, 1, 2, 3], p=[0.40, 0.35, 0.18, 0.07])
        else:
            return np.random.choice([1, 2, 3, 4], p=[0.25, 0.40, 0.25, 0.10])
    
    def _calculate_mobility(self, row: pd.Series) -> float:
        """Calculate mobility score from age and occupation."""
        age = row['age']
        
        base = 0.7
        if age < 18:
            base = 0.6
        elif age > 65:
            base = 0.4
        
        if row.get('imported', ''):
            base += 0.1
        
        return float(np.clip(base + np.random.normal(0, 0.1), 0.1, 1.0))
    
    def _generate_name(self, row: pd.Series, idx: int) -> str:
        """Generate realistic name based on nationality and gender."""
        gender = row['gender']
        nationality = str(row.get('nationality', 'singapore')).lower()
        
        # Name pools by nationality
        if 'singapore' in nationality or 'china' in nationality:
            male_names = ['Wei', 'Jun', 'Ming', 'Hao', 'Kai', 'Jie', 'Yang', 'Xin']
            female_names = ['Mei', 'Li', 'Xia', 'Ying', 'Hui', 'Jia', 'Yan', 'Lin']
            surnames = ['Tan', 'Lim', 'Lee', 'Ng', 'Ong', 'Wong', 'Goh', 'Chua']
        elif 'india' in nationality:
            male_names = ['Raj', 'Kumar', 'Ajay', 'Vijay', 'Ravi', 'Arun', 'Deepak', 'Suresh']
            female_names = ['Priya', 'Kavya', 'Anjali', 'Divya', 'Lakshmi', 'Meera', 'Pooja', 'Rani']
            surnames = ['Kumar', 'Singh', 'Sharma', 'Patel', 'Reddy', 'Nair', 'Iyer', 'Pillai']
        elif 'malay' in nationality:
            male_names = ['Ahmad', 'Ali', 'Hassan', 'Ibrahim', 'Ismail', 'Omar', 'Yusof', 'Rahman']
            female_names = ['Nur', 'Siti', 'Fatimah', 'Aishah', 'Zainab', 'Hajar', 'Mariam', 'Sofiah']
            surnames = ['Abdullah', 'Rahman', 'Hassan', 'Ahmad', 'Ibrahim', 'Ismail', 'Ali', 'Omar']
        elif 'bangladesh' in nationality:
            male_names = ['Mohammed', 'Abdul', 'Rahim', 'Karim', 'Rashid', 'Jalal', 'Aziz', 'Hafiz']
            female_names = ['Fatima', 'Ayesha', 'Razia', 'Nasrin', 'Sultana', 'Begum', 'Amina', 'Zara']
            surnames = ['Rahman', 'Ahmed', 'Hossain', 'Ali', 'Khan', 'Islam', 'Miah', 'Uddin']
        else:
            male_names = ['John', 'James', 'Michael', 'David', 'Robert', 'William', 'Richard', 'Thomas']
            female_names = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Susan', 'Jessica', 'Sarah']
            surnames = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis', 'Wilson']
        
        np.random.seed(idx + 5000)
        if gender == 'f':
            first = np.random.choice(female_names)
        else:
            first = np.random.choice(male_names)
        
        last = np.random.choice(surnames)
        np.random.seed()
        
        return f"{first} {last}"
    
    def _infer_contact_network(
        self,
        sampled_cases: pd.DataFrame,
        profiles: List[Dict]
    ) -> List[Dict]:
        """Infer contact network from cluster assignments and links."""
        print("\nInferring contact network...")
        
        edges = []
        agent_ids = set(p['agent_id'] for p in profiles)
        agent_by_case = {p['case_number']: p['agent_id'] for p in profiles}
        
        # Layer 1: Household
        household_edges = self._generate_household_contacts(profiles)
        edges.extend(household_edges)
        print(f"  Layer 1 - Household: {len(household_edges)} edges")
        
        # Layer 2: Explicit links
        link_edges = self._extract_explicit_links(sampled_cases, agent_by_case, agent_ids)
        edges.extend(link_edges)
        print(f"  Layer 2 - Explicit links: {len(link_edges)} edges")
        
        # Layer 3: Clusters
        cluster_edges = self._generate_cluster_contacts(profiles, edges)
        edges.extend(cluster_edges)
        print(f"  Layer 3 - Clusters: {len(cluster_edges)} edges")
        
        # Layer 4: Work/school
        work_school_edges = self._generate_work_school_contacts(profiles, edges)
        edges.extend(work_school_edges)
        print(f"  Layer 4 - Work/School: {len(work_school_edges)} edges")
        
        # Ensure no isolated agents
        edges = self._ensure_no_isolated_agents(profiles, edges)
        
        print(f"\n✓ Total edges: {len(edges)}")
        print(f"  Average degree: {2 * len(edges) / len(profiles):.1f}")
        
        return edges
    
    def _generate_household_contacts(self, profiles: List[Dict]) -> List[Dict]:
        """Generate complete graphs within households."""
        edges = []
        
        households = defaultdict(list)
        for p in profiles:
            households[p['household_id']].append(p['agent_id'])
        
        for hh_id, members in households.items():
            if len(members) < 2:
                continue
            
            for i, agent_1 in enumerate(members):
                for agent_2 in members[i+1:]:
                    edges.append({
                        'agent_1': min(agent_1, agent_2),
                        'agent_2': max(agent_1, agent_2),
                        'edge_type': 'household',
                        'contact_rate': 0.95
                    })
        
        return edges
    
    def _extract_explicit_links(
        self,
        sampled_cases: pd.DataFrame,
        agent_by_case: Dict[int, int],
        agent_ids: Set[int]
    ) -> List[Dict]:
        """Extract explicit links from the 'link' field."""
        edges = []
        existing = set()
        
        for idx, row in sampled_cases.iterrows():
            agent_1 = idx
            link_text = str(row.get('link', '')).lower()
            
            if not link_text or link_text == 'nan':
                continue
            
            case_numbers = re.findall(r'\b(\d+)\b', link_text)
            
            for case_str in case_numbers:
                case_num = int(case_str)
                
                if case_num in agent_by_case:
                    agent_2 = agent_by_case[case_num]
                    
                    if agent_1 != agent_2:
                        edge_tuple = (min(agent_1, agent_2), max(agent_1, agent_2))
                        
                        if edge_tuple not in existing:
                            if 'family' in link_text or 'household' in link_text:
                                edge_type = 'household'
                                contact_rate = 0.95
                            elif 'work' in link_text:
                                edge_type = 'work'
                                contact_rate = 0.70
                            else:
                                edge_type = 'contact'
                                contact_rate = 0.60
                            
                            edges.append({
                                'agent_1': edge_tuple[0],
                                'agent_2': edge_tuple[1],
                                'edge_type': edge_type,
                                'contact_rate': contact_rate
                            })
                            existing.add(edge_tuple)
        
        return edges
    
    def _generate_cluster_contacts(
        self,
        profiles: List[Dict],
        existing_edges: List[Dict]
    ) -> List[Dict]:
        """Generate contacts within same cluster."""
        edges = []
        
        existing = set()
        for e in existing_edges:
            existing.add((e['agent_1'], e['agent_2']))
        
        clusters = defaultdict(list)
        for p in profiles:
            cluster = p.get('cluster', '')
            if cluster and cluster.lower() != 'nan':
                clusters[cluster].append(p['agent_id'])
        
        for cluster_name, members in clusters.items():
            if len(members) < 2:
                continue
            
            if len(members) > 20:
                # Sparse connections for large clusters
                for agent in members:
                    n_contacts = min(4, max(2, np.random.randint(2, 5)))
                    available = [m for m in members if m != agent]
                    
                    if len(available) >= n_contacts:
                        contacts = np.random.choice(available, size=n_contacts, replace=False)
                    else:
                        contacts = available
                    
                    for contact in contacts:
                        edge_tuple = (min(agent, contact), max(agent, contact))
                        if edge_tuple not in existing:
                            edges.append({
                                'agent_1': edge_tuple[0],
                                'agent_2': edge_tuple[1],
                                'edge_type': 'cluster',
                                'contact_rate': 0.50
                            })
                            existing.add(edge_tuple)
            else:
                # Higher connection probability for small clusters
                for i, agent_1 in enumerate(members):
                    for agent_2 in members[i+1:]:
                        edge_tuple = (min(agent_1, agent_2), max(agent_1, agent_2))
                        
                        if edge_tuple not in existing and np.random.random() < 0.4:
                            edges.append({
                                'agent_1': edge_tuple[0],
                                'agent_2': edge_tuple[1],
                                'edge_type': 'cluster',
                                'contact_rate': 0.60
                            })
                            existing.add(edge_tuple)
        
        return edges
    
    def _generate_work_school_contacts(
        self,
        profiles: List[Dict],
        existing_edges: List[Dict]
    ) -> List[Dict]:
        """Generate sparse work/school contacts."""
        edges = []
        
        existing = set()
        for e in existing_edges:
            existing.add((e['agent_1'], e['agent_2']))
        
        # Work networks
        work_networks = defaultdict(list)
        for p in profiles:
            if p['work_network'] >= 0:
                work_networks[p['work_network']].append(p['agent_id'])
        
        # School networks
        school_networks = defaultdict(list)
        for p in profiles:
            if p['school_network'] >= 0:
                school_networks[p['school_network']].append(p['agent_id'])
        
        # Generate contacts
        for network_dict, edge_type, contact_rate in [
            (work_networks, 'work', 0.70),
            (school_networks, 'school', 0.65)
        ]:
            for network_id, members in network_dict.items():
                if len(members) < 2:
                    continue
                
                for agent in members:
                    n_contacts = min(4, max(2, np.random.randint(2, 5)))
                    available = [m for m in members if m != agent]
                    
                    if len(available) >= n_contacts:
                        contacts = np.random.choice(available, size=n_contacts, replace=False)
                    else:
                        contacts = available
                    
                    for contact in contacts:
                        edge_tuple = (min(agent, contact), max(agent, contact))
                        if edge_tuple not in existing:
                            edges.append({
                                'agent_1': edge_tuple[0],
                                'agent_2': edge_tuple[1],
                                'edge_type': edge_type,
                                'contact_rate': contact_rate
                            })
                            existing.add(edge_tuple)
        
        return edges
    
    def _ensure_no_isolated_agents(
        self,
        profiles: List[Dict],
        edges: List[Dict]
    ) -> List[Dict]:
        """Ensure all agents have at least 1 connection."""
        degree = defaultdict(int)
        for edge in edges:
            degree[edge['agent_1']] += 1
            degree[edge['agent_2']] += 1
        
        all_agents = set(p['agent_id'] for p in profiles)
        isolated = [agent_id for agent_id in all_agents if degree[agent_id] == 0]
        
        if not isolated:
            return edges
        
        print(f"  ⚠ Found {len(isolated)} isolated agent(s), adding connections...")
        
        existing_pairs = set((e['agent_1'], e['agent_2']) for e in edges)
        
        for isolated_agent in isolated:
            available = []
            for other_id in all_agents:
                if other_id == isolated_agent:
                    continue
                pair = (min(isolated_agent, other_id), max(isolated_agent, other_id))
                if pair not in existing_pairs:
                    available.append((other_id, degree[other_id]))
            
            if available:
                available.sort(key=lambda x: x[1])
                
                n_connect = min(2, len(available))
                for other_id, _ in available[:n_connect]:
                    pair = (min(isolated_agent, other_id), max(isolated_agent, other_id))
                    edges.append({
                        'agent_1': pair[0],
                        'agent_2': pair[1],
                        'edge_type': 'community',
                        'contact_rate': 0.40
                    })
                    existing_pairs.add(pair)
                    degree[isolated_agent] += 1
                    degree[other_id] += 1
        
        print(f"  ✓ All agents now connected")
        return edges
    
    def _validate_data(self, profiles: List[Dict], edges: List[Dict]):
        """Validate extracted data."""
        print("\nValidating data...")
        
        agent_ids = set(p['agent_id'] for p in profiles)
        
        assert len(profiles) <= self.n_agents
        assert len(agent_ids) == len(profiles)
        
        for edge in edges:
            assert edge['agent_1'] in agent_ids
            assert edge['agent_2'] in agent_ids
            assert edge['agent_1'] < edge['agent_2']
        
        degree = defaultdict(int)
        for edge in edges:
            degree[edge['agent_1']] += 1
            degree[edge['agent_2']] += 1
        
        if degree:
            degrees = list(degree.values())
            avg_degree = np.mean(degrees)
            
            print(f"  ✓ Profiles: {len(profiles)} agents")
            print(f"  ✓ Edges: {len(edges)} connections")
            print(f"  ✓ Degree: min={min(degrees)}, avg={avg_degree:.1f}, max={max(degrees)}")
            
            assert min(degrees) >= 1, "Isolated agents detected!"
        
        print("\n✓ All validation checks passed")


def extract_singapore_data(
    case_file_path: str,
    n_agents: int = 1000,
    output_dir: str = "data/processed",
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Main entry point for extracting Singapore COVID-19 data.
    
    Args:
        case_file_path: Path to Kaggle CSV file (must have day_0 column)
        n_agents: Number of agents to use (chronological)
        output_dir: Where to save extracted data
        random_seed: Random seed
    
    Returns:
        (profiles, edges) tuple
    """
    extractor = SingaporeDataExtractor(
        case_file_path=case_file_path,
        n_agents=n_agents,
        random_seed=random_seed
    )
    
    profiles, edges = extractor.extract_all_data()
    
    return profiles, edges


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python agent_data_extractor.py <path_to_singapore_covid_csv>")
        print("\nExample:")
        print("  python agent_data_extractor.py data/raw/singapore_covid19_cases.csv")
        print("\nNOTE: CSV must have 'day_0' column with initial states (S/E/I/R/D)")
        sys.exit(1)
    
    case_file = sys.argv[1]
    
    profiles, edges = extract_singapore_data(
        case_file_path=case_file,
        n_agents=1000,
        random_seed=42
    )
    
    print(f"\n✓ Successfully extracted {len(profiles)} profiles and {len(edges)} edges")
