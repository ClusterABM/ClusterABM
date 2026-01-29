"""
CLI script to extract Singapore COVID-19 agent data.
Reads initial states from Kaggle day_0 column.

Usage:
    python scripts/extract_singapore_data.py data/raw/singapore_covid19_cases.csv
"""

import sys
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
from collections import Counter

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.agent_data_extractor import SingaporeDataExtractor
from src.data_processing.data_validator import DataValidator


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, 'isoformat'):  # Handles pandas Timestamp and datetime
        return obj.isoformat()
    elif pd.isna(obj):  # Handle pandas NaT and NaN
        return None
    return obj


def status_to_state(status: int) -> str:
    """Convert disease_status number to SEIRD state letter."""
    return {0: 'S', 1: 'E', 2: 'I', 3: 'R', 4: 'D'}[status]


def main():
    parser = argparse.ArgumentParser(
        description='Extract Singapore COVID-19 data from Kaggle CSV with day_0 column',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/extract_singapore_data.py data/raw/singapore_covid19_cases.csv

Kaggle CSV must have these columns:
  - case, date, age, gender, nationality
  - day_0 (with S/E/I/R/D states on 2020-01-23)
  - Optional: imported, cluster_local, link, date discharged, hospital

Download from: https://www.kaggle.com/datasets/hoonbeng/singapores-covid19-cases
        """
    )
    
    parser.add_argument('case_file', type=str,
                       help='Path to Singapore COVID-19 case CSV file (with day_0 column)')
    parser.add_argument('--n_agents', type=int, default=1000, 
                       help='Number of agents to extract (chronologically) (default: 1000)')
    parser.add_argument('--simulation_start_date', type=str, default='2020-01-23',
                       help='Simulation start date (default: 2020-01-23)')
    parser.add_argument('--output_dir', type=str, default='data/singapore',
                       help='Output directory (default: data/singapore)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for network generation (default: 42)')
    parser.add_argument('--skip_validation', action='store_true',
                       help='Skip data validation (not recommended)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    case_file_path = Path(args.case_file)
    if not case_file_path.exists():
        print(f"❌ Error: Case file not found: {args.case_file}")
        print("\n📥 Expected: Kaggle Singapore COVID-19 case CSV")
        print("   Download from: https://www.kaggle.com/datasets/hoonbeng/singapores-covid19-cases")
        print("\n⚠️  CSV must have 'day_0' column with initial states (S/E/I/R/D)")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("SINGAPORE COVID-19 DATA EXTRACTION")
    print("Initial states from Kaggle day_0 column")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   Input file: {args.case_file}")
    print(f"   Agents to extract: {args.n_agents} (first N chronologically)")
    print(f"   Simulation start: {args.simulation_start_date}")
    print(f"   Random seed: {args.random_seed}")
    print(f"   Output directory: {args.output_dir}")
    print("="*80)
    
    # Create extractor
    try:
        extractor = SingaporeDataExtractor(
            case_file_path=args.case_file,
            n_agents=args.n_agents,
            simulation_start_date=args.simulation_start_date,
            random_seed=args.random_seed
        )
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n⚠️  Make sure your CSV has a 'day_0' column with initial states!")
        sys.exit(1)
    
    # Extract data
    try:
        profiles, edges = extractor.extract_all_data()
    except Exception as e:
        print(f"\n❌ ERROR during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Validate
    if not args.skip_validation:
        print("\n" + "="*80)
        print("VALIDATING EXTRACTED DATA")
        print("="*80)
        validator = DataValidator()
        validation_passed = validator.validate_all(profiles, edges)
        
        if not validation_passed:
            print("\n❌ Validation failed! Please check the data.")
            sys.exit(1)
    else:
        print("\n⚠️  Skipping validation (--skip_validation flag)")
    
    # Convert numpy types for JSON serialization
    print("\nConverting data types for JSON serialization...")
    profiles = convert_numpy_types(profiles)
    edges = convert_numpy_types(edges)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save files
    print("\n" + "="*80)
    print("SAVING EXTRACTED DATA")
    print("="*80)
    
    # Save profiles
    profiles_file = output_dir / 'profiles.json'
    with open(profiles_file, 'w') as f:
        json.dump(profiles, f, indent=2, cls=NumpyEncoder)
    print(f"✅ Saved profiles: {profiles_file}")
    
    # Save edges
    edges_file = output_dir / 'edges.json'
    with open(edges_file, 'w') as f:
        json.dump(edges, f, indent=2, cls=NumpyEncoder)
    print(f"✅ Saved edges: {edges_file}")
    
    # Generate extraction summary
    print("\nGenerating summary...")
    
    # State distribution from disease_status
    state_counts = Counter(status_to_state(p['disease_status']) for p in profiles)
    
    summary = {
        'dataset': 'Singapore COVID-19 (Kaggle)',
        'extraction_date': args.simulation_start_date,
        'n_agents': len(profiles),
        'n_edges': len(edges),
        'avg_degree': float(2 * len(edges) / len(profiles)) if len(profiles) > 0 else 0,
        
        # State distribution (from day_0 column)
        'initial_state_distribution': {
            'S': state_counts.get('S', 0),
            'E': state_counts.get('E', 0),
            'I': state_counts.get('I', 0),
            'R': state_counts.get('R', 0),
            'D': state_counts.get('D', 0),
        },
        'infected_seed': state_counts.get('E', 0) + state_counts.get('I', 0),
        
        # Demographics
        'age_stats': {
            'min': int(min(p['age'] for p in profiles)),
            'max': int(max(p['age'] for p in profiles)),
            'mean': float(np.mean([p['age'] for p in profiles])),
            'median': float(np.median([p['age'] for p in profiles])),
        },
        'gender_distribution': dict(Counter(p['gender'] for p in profiles)),
        'nationality_distribution': dict(Counter(p['nationality'] for p in profiles)),
        
        # Epidemiological
        'imported_cases': {
            'count': sum(1 for p in profiles if p['is_imported']),
            'percentage': float(100 * sum(1 for p in profiles if p['is_imported']) / len(profiles))
        },
        'vaccination_distribution': dict(Counter(p['vaccination_status'] for p in profiles)),
        
        # Occupations
        'top_5_occupations': dict(Counter(p['occupation'] for p in profiles).most_common(5)),
        
        # Social structure
        'household_count': len(set(p['household_id'] for p in profiles)),
        'cluster_count': len(set(p['cluster'] for p in profiles if p['cluster'] and str(p['cluster']).lower() not in ['', 'nan', 'none'])),
        
        # Network
        'edge_type_distribution': dict(Counter(e['edge_type'] for e in edges)),
        'contact_rate_stats': {
            'min': float(min(e['contact_rate'] for e in edges)) if edges else 0,
            'max': float(max(e['contact_rate'] for e in edges)) if edges else 0,
            'mean': float(np.mean([e['contact_rate'] for e in edges])) if edges else 0,
        },
        
        # Temporal
        'days_since_start_stats': {
            'min': int(min(p['days_since_start'] for p in profiles)),
            'max': int(max(p['days_since_start'] for p in profiles)),
            'mean': float(np.mean([p['days_since_start'] for p in profiles])),
        },
    }
    
    # Save summary
    summary_file = output_dir / 'extraction_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"✅ Saved summary: {summary_file}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("✅ EXTRACTION COMPLETE")
    print("="*80)
    
    print(f"\n📊 Summary:")
    print(f"   Agents: {summary['n_agents']}")
    print(f"   Edges: {summary['n_edges']}")
    print(f"   Average degree: {summary['avg_degree']:.1f}")
    
    print(f"\n🦠 Initial States (from day_0):")
    for state in ['S', 'E', 'I', 'R', 'D']:
        count = summary['initial_state_distribution'][state]
        pct = 100 * count / summary['n_agents'] if summary['n_agents'] > 0 else 0
        print(f"   {state}: {count:4d} ({pct:5.1f}%)")
    print(f"   Infected seed: {summary['infected_seed']} agents")
    
    print(f"\n👥 Demographics:")
    print(f"   Age range: {summary['age_stats']['min']}-{summary['age_stats']['max']} (mean: {summary['age_stats']['mean']:.1f})")
    print(f"   Imported cases: {summary['imported_cases']['count']} ({summary['imported_cases']['percentage']:.1f}%)")
    print(f"   Clusters: {summary['cluster_count']}")
    print(f"   Households: {summary['household_count']}")
    
    print(f"\n🔗 Network:")
    print(f"   Edge types: {summary['edge_type_distribution']}")
    print(f"   Contact rates: min={summary['contact_rate_stats']['min']:.2f}, mean={summary['contact_rate_stats']['mean']:.2f}, max={summary['contact_rate_stats']['max']:.2f}")
    
    print(f"\n📁 Output location: {output_dir}/")
    print(f"   - profiles.json ({len(profiles)} agents)")
    print(f"   - edges.json ({len(edges)} connections)")
    print(f"   - extraction_summary.json (metadata)")
    
    # Check if infected seed is adequate
    if summary['infected_seed'] == 0:
        print("\n⚠️  WARNING: No infected agents (E or I) at simulation start!")
        print("   Epidemic cannot spread with zero initial infections.")
        print("   Check your Kaggle day_0 column has proper state data.")
    elif summary['infected_seed'] < 10:
        print(f"\n⚠️  WARNING: Only {summary['infected_seed']} infected agents.")
        print("   Epidemic may spread slowly with limited initial infections.")
    else:
        print(f"\n✅ Infected seed is adequate: {summary['infected_seed']} agents")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Review extracted data:")
    print(f"   cat {summary_file}")
    
    print("\n2. Train GraphSAGE embeddings:")
    print(f"   python scripts/train_graphsage.py --profiles {profiles_file} --edges {edges_file}")
    
    print("\n3. Perform HSBC³ clustering:")
    print(f"   python scripts/cluster_agents_hsbc.py")
    
    print("\n4. Train neural model:")
    print("   python scripts/train_neural_model.py")
    
    print("\n5. Run full 83-day simulation:")
    print("   python scripts/run_rolling_window_simulation.py")
    
    print("\n✨ Data is ready for StateAgentNet simulation!")
    print()


if __name__ == "__main__":
    main()
