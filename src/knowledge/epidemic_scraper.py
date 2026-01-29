"""
Web scraper for Singapore CDA COVID-19 guidelines.
Extracts ONLY general public health guidelines, filters out data leakage.

FILTERING RULES:
- Remove specific dates, statistics, case counts
- Keep only general principles and guidelines
- Remove Singapore-specific outbreak patterns
- Keep timeless public health advice
"""

import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional
from pathlib import Path
import json
import time


class CDAScraper:
    """
    Scrape CDA Singapore COVID-19 guidelines with data leakage filtering.
    """
    
    def __init__(self, cache_dir: str = "data/cda_cache"):
        """
        Initialize scraper.
        
        Args:
            cache_dir: Directory to cache scraped content
        """
        self.base_url = "https://www.cda.gov.sg/public/diseases/covid-19/"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Research/Educational Purpose)'
        }
        
    def scrape_and_filter(self) -> List[Dict]:
        """
        Scrape CDA website and extract filtered guidelines.
        
        Returns:
            List of filtered guideline documents
        """
        print("\n" + "="*80)
        print("SCRAPING CDA SINGAPORE COVID-19 GUIDELINES")
        print("="*80)
        print(f"URL: {self.base_url}")
        print("Filtering: Removing data leakage (dates, statistics, specific events)")
        
        try:
            # Fetch page
            print("\nFetching page...")
            html_content = self._fetch_page()
            
            if not html_content:
                print("❌ Failed to fetch page")
                return self._get_fallback_guidelines()
            
            # Parse content
            print("Parsing content...")
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract sections
            print("Extracting sections...")
            raw_sections = self._extract_sections(soup)
            
            print(f"✓ Extracted {len(raw_sections)} raw sections")
            
            # Filter for general guidelines only
            print("\nFiltering for general guidelines (removing data leakage)...")
            filtered_docs = self._filter_sections(raw_sections)
            
            print(f"✓ Filtered to {len(filtered_docs)} clean guideline documents")
            
            # Cache results
            self._cache_results(filtered_docs)
            
            return filtered_docs
            
        except Exception as e:
            print(f"❌ Error during scraping: {e}")
            print("Using fallback guidelines...")
            return self._get_fallback_guidelines()
    
    def _fetch_page(self) -> Optional[str]:
        """Fetch page content with caching."""
        cache_file = self.cache_dir / "cda_page.html"
        
        # Use cache if recent (within 7 days)
        if cache_file.exists():
            age_days = (time.time() - cache_file.stat().st_mtime) / 86400
            if age_days < 7:
                print(f"  Using cached page (age: {age_days:.1f} days)")
                return cache_file.read_text(encoding='utf-8')
        
        # Fetch fresh
        try:
            print(f"  Fetching from {self.base_url}...")
            response = requests.get(
                self.base_url,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"  ✓ Fetched successfully ({len(response.text)} bytes)")
                
                # Cache
                cache_file.write_text(response.text, encoding='utf-8')
                
                return response.text
            else:
                print(f"  ✗ HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"  ✗ Fetch failed: {e}")
            return None
    
    def _extract_sections(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract content sections from page."""
        sections = []
        
        # Try to find main content sections
        # This is a generic approach - may need adjustment for actual site structure
        
        # Look for common section markers
        section_tags = soup.find_all(['section', 'article', 'div'], class_=re.compile(r'(content|section|article|guideline)', re.I))
        
        if not section_tags:
            # Fallback: get all major divs
            section_tags = soup.find_all('div', class_=True)
        
        for i, tag in enumerate(section_tags):
            # Extract text
            text = tag.get_text(separator='\n', strip=True)
            
            if len(text) < 50:  # Skip very short sections
                continue
            
            # Try to find a heading
            heading = None
            heading_tags = tag.find_all(['h1', 'h2', 'h3', 'h4'])
            if heading_tags:
                heading = heading_tags[0].get_text(strip=True)
            
            sections.append({
                'id': f'cda_section_{i}',
                'heading': heading,
                'content': text,
                'raw': True
            })
        
        # Also extract any list items that might be guidelines
        lists = soup.find_all(['ul', 'ol'])
        for i, ul in enumerate(lists):
            items = ul.find_all('li')
            if len(items) >= 3:  # Only keep substantial lists
                list_text = '\n'.join(f"- {li.get_text(strip=True)}" for li in items)
                if len(list_text) > 100:
                    sections.append({
                        'id': f'cda_list_{i}',
                        'heading': 'Guidelines',
                        'content': list_text,
                        'raw': True
                    })
        
        return sections
    
    def _filter_sections(self, raw_sections: List[Dict]) -> List[Dict]:
        """
        Filter sections to remove data leakage.
        
        REMOVES:
        - Specific dates (Jan 2020, March 15, etc.)
        - Statistics and numbers (1000 cases, 18.3%, etc.)
        - Singapore-specific events (Circuit Breaker, dormitory outbreak)
        - Temporal references (last week, April data, etc.)
        
        KEEPS:
        - General symptoms and clinical info
        - Prevention guidelines
        - Testing/isolation/quarantine principles
        - General transmission information
        - Risk factors (qualitative)
        """
        filtered = []
        
        for section in raw_sections:
            content = section['content']
            heading = section.get('heading', 'General Guidelines')
            
            # Check if section contains useful guideline content
            if not self._is_guideline_content(content):
                continue
            
            # Filter out data leakage
            cleaned = self._remove_data_leakage(content)
            
            if len(cleaned) < 50:  # Skip if too little content remains
                continue
            
            # Categorize
            category = self._categorize_content(heading, cleaned)
            
            filtered.append({
                'id': f"cda_{category}_{len(filtered)}",
                'category': category,
                'source': 'Singapore CDA (General Guidelines)',
                'heading': heading,
                'content': cleaned
            })
        
        return filtered
    
    def _is_guideline_content(self, content: str) -> bool:
        """Check if content contains useful guidelines."""
        # Positive indicators
        guideline_keywords = [
            'symptom', 'transmission', 'prevent', 'protect', 'isolate',
            'quarantine', 'test', 'vaccine', 'mask', 'distance', 'wash',
            'contact', 'spread', 'risk', 'guideline', 'recommendation',
            'should', 'avoid', 'reduce', 'increase', 'maintain'
        ]
        
        content_lower = content.lower()
        
        # Must have at least 3 guideline keywords
        keyword_count = sum(1 for kw in guideline_keywords if kw in content_lower)
        
        if keyword_count < 3:
            return False
        
        # Negative indicators (not guidelines)
        non_guideline = [
            'copyright', 'privacy policy', 'terms of service',
            'cookie', 'navigation', 'menu', 'footer', 'header'
        ]
        
        if any(ng in content_lower for ng in non_guideline):
            return False
        
        return True
    
    def _remove_data_leakage(self, content: str) -> str:
        """
        Remove data leakage from content.
        
        This is aggressive filtering to ensure NO specific data remains.
        """
        # Remove sentences with specific dates
        date_patterns = [
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # March 15, 2020
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # 03/15/2020 or 15-03-2020
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b',  # March 2020
            r'\b\d{4}-\d{2}-\d{2}\b',  # 2020-03-15
            r'\bon \w+ \d+\b',  # on March 15
            r'\bsince \w+ \d{4}\b',  # since March 2020
        ]
        
        for pattern in date_patterns:
            # Remove sentences containing dates
            content = re.sub(
                r'[^.!?]*' + pattern + r'[^.!?]*[.!?]',
                '',
                content,
                flags=re.IGNORECASE
            )
        
        # Remove sentences with specific statistics
        stat_patterns = [
            r'\b\d+[,.]?\d*\s*%',  # 18.3%, 20%
            r'\b\d+[,.]?\d*\s+(?:cases|deaths|infections|patients)',  # 1000 cases
            r'\b\d+[,.]?\d*\s+per\s+\w+',  # 50 per million
            r'\b(?:increased|decreased|rose|fell)\s+by\s+\d+',  # increased by 50
            r'\b\d+[,.]?\d*\s+times\s+(?:higher|lower)',  # 2.5 times higher
        ]
        
        for pattern in stat_patterns:
            content = re.sub(
                r'[^.!?]*' + pattern + r'[^.!?]*[.!?]',
                '',
                content,
                flags=re.IGNORECASE
            )
        
        # Remove Singapore-specific event references
        singapore_events = [
            r'circuit breaker',
            r'dormitory outbreak',
            r'heightened alert',
            r'phase \d+',
            r'safe management measures',
            r'TraceTogether',
            r'SafeEntry',
        ]
        
        for event in singapore_events:
            content = re.sub(
                r'[^.!?]*' + event + r'[^.!?]*[.!?]',
                '',
                content,
                flags=re.IGNORECASE
            )
        
        # Remove temporal references
        temporal_refs = [
            r'\b(?:last|this|next)\s+(?:week|month|year)\b',
            r'\bcurrently\b',
            r'\bas of\s+\w+\s+\d+',
            r'\bto date\b',
            r'\brecently\b',
            r'\bin recent \w+',
        ]
        
        for ref in temporal_refs:
            content = re.sub(ref, '', content, flags=re.IGNORECASE)
        
        # Clean up multiple spaces and blank lines
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        content = re.sub(r'  +', ' ', content)
        
        return content.strip()
    
    def _categorize_content(self, heading: str, content: str) -> str:
        """Categorize content by topic."""
        heading_lower = (heading or '').lower()
        content_lower = content.lower()
        
        combined = heading_lower + ' ' + content_lower
        
        # Category keywords
        categories = {
            'symptoms': ['symptom', 'sign', 'fever', 'cough', 'shortness of breath'],
            'transmission': ['transmit', 'spread', 'contagious', 'infectious', 'airborne', 'droplet'],
            'prevention': ['prevent', 'protect', 'avoid', 'reduce risk', 'hygiene', 'mask', 'distance'],
            'testing': ['test', 'PCR', 'antigen', 'swab', 'diagnos'],
            'isolation': ['isolate', 'isolat', 'quarantine', 'stay home', 'separate'],
            'vaccination': ['vaccin', 'immuniz', 'shot', 'dose'],
            'treatment': ['treat', 'therapy', 'medication', 'hospital'],
            'risk_factors': ['risk factor', 'elderly', 'age', 'comorbid', 'vulnerable'],
        }
        
        # Count keywords per category
        scores = {}
        for cat, keywords in categories.items():
            score = sum(1 for kw in keywords if kw in combined)
            scores[cat] = score
        
        # Return category with highest score
        if scores:
            best_cat = max(scores.items(), key=lambda x: x[1])
            if best_cat[1] > 0:
                return best_cat[0]
        
        return 'general_guidelines'
    
    def _cache_results(self, filtered_docs: List[Dict]):
        """Cache filtered results."""
        cache_file = self.cache_dir / "cda_filtered_guidelines.json"
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source': self.base_url,
                'num_documents': len(filtered_docs),
                'documents': filtered_docs
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Cached filtered guidelines: {cache_file}")
    
    def _get_fallback_guidelines(self) -> List[Dict]:
        """
        Fallback guidelines if scraping fails.
        General CDA-style public health guidelines.
        """
        print("\nUsing fallback CDA-style guidelines...")
        
        return [
            {
                'id': 'cda_fallback_symptoms',
                'category': 'symptoms',
                'source': 'General Public Health Guidelines',
                'content': """Common COVID-19 Symptoms:

MOST COMMON:
- Fever or chills
- Cough (usually dry)
- Fatigue or tiredness
- Loss of taste or smell (distinctive symptom)

OTHER SYMPTOMS:
- Shortness of breath or difficulty breathing
- Muscle or body aches
- Headache
- Sore throat
- Congestion or runny nose
- Nausea or vomiting
- Diarrhea

SEEK MEDICAL ATTENTION IF:
- Difficulty breathing or shortness of breath
- Persistent chest pain or pressure
- Confusion or inability to wake
- Bluish lips or face

Note: Some people may have no symptoms (asymptomatic) but can still spread the virus."""
            },
            {
                'id': 'cda_fallback_prevention',
                'category': 'prevention',
                'source': 'General Public Health Guidelines',
                'content': """COVID-19 Prevention Measures:

PERSONAL PROTECTION:
- Wear a well-fitting mask in crowded indoor spaces
- Maintain physical distance from others (at least 1 meter)
- Practice good hand hygiene (wash with soap for 20 seconds)
- Use alcohol-based hand sanitizer when soap unavailable
- Avoid touching eyes, nose, and mouth with unwashed hands

RESPIRATORY HYGIENE:
- Cover coughs and sneezes with elbow or tissue
- Dispose of tissues immediately
- Wash hands after coughing or sneezing

ENVIRONMENTAL MEASURES:
- Ensure good ventilation indoors
- Clean and disinfect frequently touched surfaces
- Avoid crowded, poorly ventilated spaces

WHEN SICK:
- Stay home if you have symptoms
- Seek testing if symptomatic
- Isolate from others to prevent spread"""
            },
            {
                'id': 'cda_fallback_isolation',
                'category': 'isolation',
                'source': 'General Public Health Guidelines',
                'content': """Isolation and Quarantine Guidelines:

ISOLATION (If You Test Positive or Have Symptoms):
- Stay in a separate room from others
- Use a separate bathroom if available
- Avoid contact with household members and pets
- Wear a mask when around others
- Duration: Follow local health authority guidance (typically 5-10 days)

ENDING ISOLATION:
- After specified isolation period
- If fever-free for 24 hours without medication
- If symptoms are improving

QUARANTINE (If Exposed to COVID-19):
- Stay home and away from others
- Monitor for symptoms
- Get tested as recommended
- Duration: Follow local health authority guidance

HOUSEHOLD PRECAUTIONS:
- Infected person should isolate in separate space
- Minimize contact with household members
- Do not share personal items
- Clean shared spaces regularly"""
            },
        ]


def add_cda_guidelines_to_kb(kb):
    """
    Add filtered CDA guidelines to knowledge base.
    
    Args:
        kb: SingaporeEpidemicKnowledgeBase instance
    """
    scraper = CDAScraper()
    filtered_docs = scraper.scrape_and_filter()
    
    if not filtered_docs:
        print("\n⚠️  No CDA guidelines extracted")
        return
    
    print(f"\n" + "="*80)
    print(f"ADDING {len(filtered_docs)} CDA GUIDELINES TO KNOWLEDGE BASE")
    print("="*80)
    
    added = 0
    for doc in filtered_docs:
        try:
            kb.collection.add(
                documents=[doc['content']],
                metadatas=[{
                    'category': doc['category'],
                    'source': doc['source']
                }],
                ids=[doc['id']]
            )
            added += 1
        except Exception as e:
            # Skip if already exists
            if 'already exists' in str(e).lower():
                continue
            print(f"  ⚠️  Skipped {doc['id']}: {e}")
    
    print(f"\n✓ Added {added} CDA guideline documents")
    print(f"✓ Total documents in KB: {kb.collection.count()}")
    
    # Print categories
    from collections import Counter
    categories = Counter(doc['category'] for doc in filtered_docs)
    print(f"\n📊 CDA Guidelines by Category:")
    for cat, count in categories.most_common():
        print(f"   {cat}: {count}")


if __name__ == "__main__":
    # Test scraper
    scraper = CDAScraper()
    docs = scraper.scrape_and_filter()
    
    print(f"\n" + "="*80)
    print("SCRAPING RESULTS")
    print("="*80)
    print(f"Documents extracted: {len(docs)}")
    
    if docs:
        print("\nSample documents:")
        for i, doc in enumerate(docs[:3], 1):
            print(f"\n{i}. {doc.get('heading', 'No heading')} ({doc['category']})")
            print(f"   Length: {len(doc['content'])} chars")
            print(f"   Preview: {doc['content'][:200]}...")
