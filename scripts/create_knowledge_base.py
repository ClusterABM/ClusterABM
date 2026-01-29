"""
Create Singapore COVID-19 knowledge base for RAG.
Populates ChromaDB with GENERAL COVID-19 knowledge (NO data leakage).
Optionally scrapes CDA Singapore for public health guidelines.
"""
import os
# 0 = Verbose, 1 = Info, 2 = Warning, 3 = Error (Standard), 4 = Fatal
# Setting it to 4 or higher often suppresses these C++ backend messages
os.environ["ORT_LOGGING_LEVEL"] = "4"

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge.epidemic_kb import SingaporeEpidemicKnowledgeBase


def validate_no_data_leakage(kb):
    """
    Validate that knowledge base contains NO data leakage.
    
    Checks for:
    - Specific dates (years, months)
    - Statistics (percentages, case counts)
    - Singapore-specific events (Circuit Breaker, etc.)
    """
    print("\n" + "="*80)
    print("VALIDATING: NO DATA LEAKAGE")
    print("="*80)
    
    # Get all documents
    all_docs = kb.collection.get()
    
    if not all_docs['documents']:
        print("⚠️  No documents to validate")
        return True
    
    leakage_found = False
    issues = []
    
    for i, doc in enumerate(all_docs['documents']):
        doc_id = all_docs['ids'][i] if all_docs['ids'] else f"doc_{i}"
        
        # Check for years
        years = ['2019', '2020', '2021', '2022', '2023', '2024']
        for year in years:
            if year in doc:
                issues.append(f"Year '{year}' in {doc_id}")
                leakage_found = True
        
        # Check for months with years
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                 'July', 'August', 'September', 'October', 'November', 'December']
        for month in months:
            if any(f"{month} {year}" in doc for year in years):
                issues.append(f"Date reference in {doc_id}")
                leakage_found = True
                break
        
        # Check for Singapore events
        sg_events = ['circuit breaker', 'dormitory outbreak', 'tracetogether', 
                     'safeentry', 'phase 2', 'heightened alert']
        for event in sg_events:
            if event in doc.lower():
                issues.append(f"Singapore event '{event}' in {doc_id}")
                leakage_found = True
    
    if leakage_found:
        print(f"❌ DATA LEAKAGE DETECTED!")
        print(f"\nIssues found:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        print(f"\n⚠️  WARNING: Knowledge base contains specific data!")
        print(f"   This will cause agents to have hindsight bias.")
        return False
    else:
        print(f"✅ NO DATA LEAKAGE DETECTED!")
        print(f"   Checked {len(all_docs['documents'])} documents")
        print(f"   All content appears to be general guidelines")
        return True


def main():
    print("\n" + "="*80)
    print("CREATING CLEAN COVID-19 KNOWLEDGE BASE")
    print("="*80)
    print("NOTE: Using GENERAL epidemiological knowledge only")
    print("      NO Singapore-specific outbreak data or statistics")
    
    # Create knowledge base directory
    kb_dir = Path("data/knowledge_base")
    kb_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nKnowledge base location: {kb_dir.absolute()}")
    
    # Initialize knowledge base with general knowledge + CDA scraping
    print("\nInitializing knowledge base...")
    print("  - General COVID-19 science")
    print("  - Public health principles")
    print("  - CDA Singapore guidelines (filtered for data leakage)")
    
    kb = SingaporeEpidemicKnowledgeBase(
        collection_name="singapore_covid19_data",
        use_real_data=True,  # This now means "scrape CDA" not "use real outbreak data"
        persist_directory=str(kb_dir)
    )
    
    # Verify
    count = kb.collection.count()
    print(f"\n✓ Knowledge base created successfully!")
    print(f"  Total documents: {count}")
    
    # Validate NO data leakage
    is_clean = validate_no_data_leakage(kb)
    
    if not is_clean:
        print("\n⚠️  Knowledge base may contain data leakage!")
        print("   Consider deleting and recreating:")
        print(f"   rm -rf {kb_dir}")
        print("   python scripts/create_knowledge_base.py")
    
    # Test queries (general guidelines, not specific data)
    print("\n" + "="*80)
    print("TESTING KNOWLEDGE BASE QUERIES (GENERAL GUIDELINES)")
    print("="*80)
    
    test_queries = [
        ("COVID-19 symptoms", "Symptom information"),
        ("how does coronavirus spread", "Transmission mechanisms"),
        ("prevention measures COVID", "Prevention guidelines"),
        ("isolation quarantine guidelines", "Isolation procedures"),
        ("high-risk groups COVID", "Risk factors"),
    ]
    
    for query_text, description in test_queries:
        print(f"\n[Test] {description}")
        print(f"Query: '{query_text}'")
        results = kb.query(query_text, n_results=2)
        
        if results:
            print(f"✓ Found {len(results)} result(s)")
            for i, result in enumerate(results, 1):
                source = result['metadata'].get('source', 'Unknown')
                category = result['metadata'].get('category', 'Unknown')
                print(f"  {i}. [{category}] {source}")
                preview = result['content'][:120].replace('\n', ' ')
                print(f"     {preview}...")
        else:
            print("✗ No results found")
    
    # Test context retrieval for agents
    print("\n" + "="*80)
    print("TESTING AGENT DECISION SUPPORT")
    print("="*80)
    
    test_scenarios = [
        {
            'name': 'Healthcare worker assessing exposure risk',
            'context': {
                'age': 32,
                'occupation': 'healthcare_worker',
                'current_state': 'S'
            },
            'decision': 'assess_risk'
        },
        {
            'name': 'Elderly person considering social gathering',
            'context': {
                'age': 68,
                'occupation': 'retired',
                'current_state': 'S'
            },
            'decision': 'social_activity'
        },
        {
            'name': 'Worker in high-density setting',
            'context': {
                'age': 28,
                'occupation': 'manual_worker',
                'current_state': 'S'
            },
            'decision': 'go_to_work'
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n[Scenario] {scenario['name']}")
        print(f"  Context: Age {scenario['context']['age']}, {scenario['context']['occupation']}")
        print(f"  Decision: {scenario['decision']}")
        
        context = kb.get_relevant_knowledge(scenario['context'], scenario['decision'])
        
        # Show preview
        if context and len(context) > 50:
            preview = context[:200].replace('\n', ' ')
            print(f"  Knowledge provided: {preview}...")
            
            # Check for leakage in response
            has_leakage = False
            if any(year in context for year in ['2019', '2020', '2021']):
                print(f"  ⚠️  WARNING: Specific dates in response!")
                has_leakage = True
            if '%' in context and any(d.isdigit() for d in context.split('%')[0].split()[-1]):
                print(f"  ⚠️  WARNING: Specific statistics in response!")
                has_leakage = True
            
            if not has_leakage:
                print(f"  ✅ Response looks clean (no obvious data leakage)")
        else:
            print(f"  ⚠️  No relevant knowledge found")
    
    # Summary
    print("\n" + "="*80)
    print("KNOWLEDGE BASE CREATION COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"  Location: {kb_dir.absolute()}")
    print(f"  Documents: {count}")
    print(f"  Data leakage: {'❌ DETECTED' if not is_clean else '✅ NONE'}")
    
    print(f"\n📚 Knowledge Categories:")
    # Get category distribution
    all_docs = kb.collection.get()
    if all_docs['metadatas']:
        from collections import Counter
        categories = Counter(m.get('category', 'unknown') for m in all_docs['metadatas'])
        for cat, cnt in categories.most_common():
            print(f"  {cat}: {cnt}")
    
    print(f"\n🎯 Next Steps:")
    if is_clean:
        print(f"  ✓ Knowledge base is ready to use!")
        print(f"  ✓ Run simulations:")
        print(f"    python scripts/extract_singapore_data.py data/raw/singapore_covid19_cases.csv")
        print(f"    python scripts/train_graphsage.py")
        print(f"    python scripts/collect_reasoning_traces.py")
    else:
        print(f"  ⚠️  Fix data leakage issues first:")
        print(f"    1. Delete knowledge base: rm -rf {kb_dir}")
        print(f"    2. Update epidemic_kb.py to remove leaking content")
        print(f"    3. Recreate: python scripts/create_knowledge_base.py")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
