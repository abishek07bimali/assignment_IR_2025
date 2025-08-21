import json
import os
from datetime import datetime

class DocumentCollector:
    def __init__(self):
        self.documents = {
            'politics': [],
            'business': [],
            'health': []
        }
        
    def add_document(self, category, text, source="Manual Entry"):
        if category.lower() in self.documents:
            self.documents[category.lower()].append({
                'text': text,
                'source': source,
                'date_collected': datetime.now().isoformat()
            })
            return True
        return False
    
    def save_documents(self, filepath='data/documents.json'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
    
    def load_documents(self, filepath='data/documents.json'):
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
        return self.documents
    
    def get_statistics(self):
        stats = {}
        for category, docs in self.documents.items():
            stats[category] = len(docs)
        return stats

def collect_sample_documents():
    collector = DocumentCollector()
    
    politics_samples = [
        "The Prime Minister announced new economic policies aimed at reducing inflation and boosting employment across the nation.",
        "Opposition parties criticized the government's handling of the recent budget crisis, calling for immediate reforms.",
        "International diplomats gathered at the UN summit to discuss climate change agreements and global cooperation.",
        "The election commission announced new voting procedures to ensure transparency in the upcoming general elections.",
        "Senate passed the controversial healthcare reform bill after months of heated debate and negotiations.",
        "The President's approval ratings have declined following recent foreign policy decisions in the Middle East.",
        "Local government officials unveiled plans for infrastructure development in rural constituencies.",
        "Political analysts predict major shifts in voter demographics ahead of the midterm elections.",
        "The ruling party faces internal divisions over proposed constitutional amendments.",
        "International sanctions were imposed following violations of human rights agreements.",
        "Parliamentary sessions were disrupted as opposition members protested against new taxation policies.",
        "The foreign minister embarked on a diplomatic tour to strengthen bilateral relations with neighboring countries.",
        "Campaign finance reforms have been proposed to limit corporate influence in political elections.",
        "The government announced emergency measures to address the ongoing refugee crisis at the borders.",
        "Coalition talks between major political parties continue as no single party secured a majority.",
        "The supreme court ruling on electoral boundaries has sparked nationwide political debates.",
        "Ministers faced tough questions during the parliamentary inquiry into government spending.",
        "Political tensions escalated following allegations of election fraud in several key districts.",
        "The cabinet reshuffle saw several senior ministers replaced amid corruption investigations.",
        "New legislation on data privacy rights passed through the lower house of parliament.",
        "The governor signed an executive order addressing climate change mitigation strategies.",
        "Political parties are mobilizing grassroots campaigns ahead of local council elections.",
        "The attorney general launched an investigation into campaign finance violations.",
        "Diplomatic relations strained following trade disputes between major economic powers.",
        "The parliament voted to extend emergency powers amid ongoing security concerns.",
        "Opposition leaders called for transparency in government procurement processes.",
        "The electoral commission announced new guidelines for political advertising on social media.",
        "International observers raised concerns about press freedom following recent media regulations.",
        "The president's state of the union address focused on economic recovery and social justice.",
        "Political deadlock continues as parties fail to agree on budget allocations.",
        "The foreign ministry condemned military actions in the disputed border region.",
        "Local elections saw record voter turnout amid heightened political awareness.",
        "The government faces criticism over its response to recent civil unrest.",
        "New anti-corruption legislation aims to increase accountability in public offices.",
        "Political analysts debate the impact of social media on modern election campaigns."
    ]
    
    business_samples = [
        "Tech giant Apple reported record quarterly earnings, exceeding analyst expectations by 15 percent.",
        "The stock market experienced significant volatility following Federal Reserve's interest rate announcement.",
        "Amazon announced plans to expand its logistics network with 50 new distribution centers worldwide.",
        "Small businesses struggle to adapt to new digital transformation requirements in the post-pandemic economy.",
        "Oil prices surged to yearly highs amid supply chain disruptions and geopolitical tensions.",
        "The merger between two pharmaceutical giants created the world's largest drug manufacturer.",
        "Cryptocurrency markets faced regulatory scrutiny as governments worldwide implement new oversight measures.",
        "Retail sales showed unexpected growth despite concerns about consumer spending and inflation.",
        "The automotive industry shifts focus to electric vehicles with major investments in battery technology.",
        "Global supply chains continue to face disruptions affecting manufacturing and distribution sectors.",
        "The central bank's monetary policy decision impacts mortgage rates and housing market dynamics.",
        "E-commerce platforms report unprecedented growth as consumer shopping habits permanently shift online.",
        "Investment firms increase allocations to renewable energy projects amid climate change concerns.",
        "The gig economy expands as more workers seek flexible employment opportunities.",
        "Corporate earnings reports show mixed results across different sectors of the economy.",
        "Startup valuations reach record highs despite concerns about market sustainability.",
        "International trade agreements reshape global commerce patterns and supply chain strategies.",
        "The banking sector implements new fintech solutions to improve customer service efficiency.",
        "Manufacturing output increases as companies invest in automation and artificial intelligence.",
        "Consumer confidence indices suggest optimistic outlook for economic recovery.",
        "The real estate market experiences cooling after years of unprecedented growth.",
        "Technology companies face antitrust investigations over market dominance concerns.",
        "Venture capital funding reaches new heights in biotechnology and healthcare sectors.",
        "The hospitality industry shows signs of recovery as travel restrictions ease globally.",
        "Agricultural commodities prices fluctuate due to weather patterns and climate change.",
        "Financial markets react to inflation data and economic growth projections.",
        "Corporate restructuring efforts aim to improve operational efficiency and profitability.",
        "The renewable energy sector attracts significant investment from institutional investors.",
        "Retail chains adapt business models to compete with online marketplace dominance.",
        "The semiconductor shortage continues to impact various industries worldwide.",
        "Private equity firms increase acquisition activity in healthcare and technology sectors.",
        "The sharing economy model expands into new markets and service categories.",
        "Corporate sustainability initiatives drive changes in business practices and reporting.",
        "International currency fluctuations affect multinational corporation profit margins.",
        "The insurance industry adapts to climate change risks and emerging technologies."
    ]
    
    health_samples = [
        "Researchers discovered a breakthrough treatment for Alzheimer's disease showing promising clinical trial results.",
        "The World Health Organization issued new guidelines for preventing the spread of infectious diseases.",
        "Mental health awareness campaigns highlight the importance of seeking professional help for depression and anxiety.",
        "New study reveals the long-term effects of COVID-19 on cardiovascular and respiratory systems.",
        "Hospitals implement telemedicine services to improve healthcare accessibility in rural areas.",
        "Vaccination campaigns successfully reduced measles cases by 80 percent in developing countries.",
        "Scientists develop new gene therapy techniques for treating rare genetic disorders.",
        "Public health officials warn about the rising obesity rates and associated health complications.",
        "Medical researchers identify potential biomarkers for early cancer detection and diagnosis.",
        "The healthcare system faces challenges in addressing the growing elderly population's needs.",
        "New dietary guidelines emphasize the importance of plant-based foods for heart health.",
        "Clinical trials begin for innovative immunotherapy treatments targeting autoimmune diseases.",
        "Mental health services expand to include digital therapy platforms and mobile applications.",
        "Antibiotic resistance poses significant challenges to treating bacterial infections globally.",
        "Healthcare providers implement artificial intelligence tools for improved diagnostic accuracy.",
        "Public health campaigns focus on reducing tobacco use and promoting smoking cessation.",
        "Researchers investigate the gut microbiome's role in overall health and disease prevention.",
        "New surgical techniques using robotics improve patient outcomes and recovery times.",
        "The opioid crisis continues to impact communities, prompting new treatment approaches.",
        "Preventive healthcare measures show effectiveness in reducing chronic disease incidence.",
        "Medical breakthroughs in regenerative medicine offer hope for organ transplant patients.",
        "Healthcare disparities persist in underserved communities despite policy interventions.",
        "Nutrition studies reveal the benefits of Mediterranean diet for longevity and wellness.",
        "Emergency medical services adapt protocols to handle pandemic-related challenges.",
        "Personalized medicine approaches revolutionize cancer treatment strategies.",
        "Sleep disorders affect millions, leading to increased focus on sleep health education.",
        "Maternal and child health programs show improvements in infant mortality rates.",
        "Healthcare technology innovations streamline patient care and medical record management.",
        "Environmental health concerns rise as pollution impacts respiratory disease rates.",
        "Medical education evolves to include training in digital health technologies.",
        "Preventive screening programs detect diseases earlier, improving treatment outcomes.",
        "Healthcare costs continue to rise, prompting discussions about system reforms.",
        "Research advances in neuroscience provide insights into brain function and disorders.",
        "Global health initiatives address infectious disease outbreaks in developing regions.",
        "Fitness and wellness trends emphasize holistic approaches to health maintenance."
    ]
    
    for text in politics_samples:
        collector.add_document('politics', text, "Sample political news content")
    
    for text in business_samples:
        collector.add_document('business', text, "Sample business news content")
    
    for text in health_samples:
        collector.add_document('health', text, "Sample health news content")
    
    collector.save_documents()
    print(f"Collected documents statistics: {collector.get_statistics()}")
    return collector

if __name__ == "__main__":
    collector = collect_sample_documents()
    print("Sample documents collected and saved successfully!")