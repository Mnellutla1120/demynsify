# Reliable Data Sources for Medical Misinformation Detection

This document outlines reliable data sources and datasets that can be used to train and improve a medical misinformation detection model.

## Medical Fact-Checking Databases

### 1. **Health Feedback** (healthfeedback.org)
- **Description**: Non-profit organization that fact-checks health and medical claims
- **Data**: Expert-reviewed articles with labeled misinformation
- **Access**: Public website, can be scraped (with permission) or accessed via API if available
- **Use Case**: Training data with expert-verified labels

### 2. **PolitiFact Health** (politifact.com/topics/health)
- **Description**: Health-related fact-checks from PolitiFact
- **Data**: Rated claims (True, Mostly True, Half True, Mostly False, False, Pants on Fire)
- **Access**: Public database, structured data available
- **Use Case**: Labeled dataset with confidence ratings

### 3. **Snopes Health** (snopes.com/category/facts/health)
- **Description**: Health-related fact-checks and debunking articles
- **Data**: Verified claims with explanations
- **Access**: Public website
- **Use Case**: Training examples with detailed explanations

### 4. **FactCheck.org Health** (factcheck.org/topic/health)
- **Description**: Health and science fact-checking
- **Data**: Detailed fact-checks with sources
- **Access**: Public articles
- **Use Case**: High-quality labeled examples

## Academic Datasets

### 5. **Monant Medical Misinformation Dataset** ⭐ **PRIMARY DATASET**
- **Source**: [GitHub Repository](https://github.com/kinit-sk/medical-misinformation-dataset)
- **Description**: Comprehensive dataset mapping articles to fact-checked claims with manual and predicted labels
- **Features**:
  - Articles from news sites and blogs
  - Fact-checking articles from fact-checking portals
  - Source reliability labels (binary: reliable/unreliable)
  - Claim presence detection
  - Article stance classification
  - Article veracity labels
  - 95% of articles have source credibility labels
- **Access**: 
  - Sample data available in GitHub repo
  - Full dataset available via [Zenodo portal](https://zenodo.org) (requires research access request)
  - Must be from official university/research institution email
- **Use Case**: Primary training dataset for medical misinformation detection
- **Paper**: "Monant Medical Misinformation Dataset: Mapping Articles to Fact-Checked Claims" (SIGIR '22)
- **Citation**: See dataset repository for proper citations

### 6. **COVID-19 Misinformation Dataset**
- **Sources**: 
  - Kaggle: Multiple COVID-19 misinformation datasets
  - Papers with Code: Various academic datasets
- **Description**: Labeled COVID-19 related misinformation
- **Access**: Public datasets on Kaggle, GitHub
- **Use Case**: Specific domain training (pandemic misinformation)

### 7. **HealthMisinfo Dataset**
- **Source**: Research papers and GitHub repositories
- **Description**: Medical misinformation detection datasets from academic research
- **Access**: Check arXiv papers and associated GitHub repos
- **Use Case**: Pre-labeled medical misinformation examples

### 8. **Fake Health News Dataset**
- **Source**: Various research institutions
- **Description**: Health-related fake news articles
- **Access**: Academic repositories, research paper supplements
- **Use Case**: Binary classification (fake vs. real)

## Medical Authority Sources (Ground Truth)

### 8. **PubMed** (pubmed.ncbi.nlm.nih.gov)
- **Description**: Database of medical research papers
- **Data**: Peer-reviewed medical literature
- **Access**: Free API access (PubMed API)
- **Use Case**: Ground truth for accurate medical information

### 9. **WHO (World Health Organization)**
- **Description**: Official health information and guidelines
- **Data**: Fact sheets, Q&A documents, guidelines
- **Access**: Public website, structured content
- **Use Case**: Authoritative source for accurate medical claims

### 10. **CDC (Centers for Disease Control and Prevention)**
- **Description**: Official US health information
- **Data**: Health topics, guidelines, fact sheets
- **Access**: Public website, structured data
- **Use Case**: Reliable ground truth for US health information

### 11. **NIH (National Institutes of Health)**
- **Description**: Medical research and health information
- **Data**: Research findings, health information
- **Access**: Public website
- **Use Case**: Authoritative medical information

### 12. **Mayo Clinic** (mayoclinic.org)
- **Description**: Trusted medical information
- **Data**: Health information articles
- **Access**: Public website
- **Use Case**: Reliable medical content

### 13. **WebMD** (webmd.com)
- **Description**: Medical information portal
- **Data**: Health articles, symptom checkers
- **Access**: Public website
- **Use Case**: Common medical information patterns

## Social Media Misinformation Sources

### 14. **Twitter/X Health Misinformation**
- **Description**: Social media posts flagged as health misinformation
- **Data**: Tweets, retweets, replies
- **Access**: Twitter API (requires API access)
- **Use Case**: Real-world misinformation patterns

### 15. **Reddit Health Subreddits**
- **Description**: Health-related discussions (some with misinformation)
- **Data**: Posts and comments from health subreddits
- **Access**: Reddit API (free tier available)
- **Use Case**: Natural language patterns of misinformation

## Specialized Medical Databases

### 16. **MedlinePlus** (medlineplus.gov)
- **Description**: Consumer health information from NIH
- **Data**: Health topics, drug information, medical encyclopedia
- **Access**: Public website, structured content
- **Use Case**: Reliable medical information patterns

### 17. **UpToDate** (uptodate.com)
- **Description**: Evidence-based clinical information
- **Data**: Medical topics, treatment guidelines
- **Access**: Subscription required (but summaries available)
- **Use Case**: High-quality medical content

### 18. **Cochrane Library** (cochranelibrary.com)
- **Description**: Systematic reviews of medical evidence
- **Data**: Evidence-based medical reviews
- **Access**: Some free access, subscription for full
- **Use Case**: Evidence-based medical information

## Data Collection Strategies

### 1. **Web Scraping** (with permission)
- Scrape fact-checking websites
- Respect robots.txt and rate limits
- Check terms of service

### 2. **API Access**
- PubMed API for medical literature
- Twitter API for social media data
- Reddit API for discussion data

### 3. **Academic Datasets**
- Search arXiv for medical misinformation papers
- Check associated GitHub repositories
- Look for shared datasets in research papers

### 4. **Crowdsourcing**
- Create labeled datasets through platforms like:
  - Amazon Mechanical Turk
  - Prolific
  - Labelbox

### 5. **Partnerships**
- Partner with fact-checking organizations
- Collaborate with medical institutions
- Work with health departments

## Data Labeling Guidelines

When creating your own dataset:

1. **Binary Classification**: True/False
2. **Multi-class**: Accurate, Misleading, Unverified, False
3. **Confidence Scores**: 0-1 scale
4. **Source Attribution**: Link to authoritative sources
5. **Expert Review**: Have medical professionals review labels

## Recommended Approach

1. **Start with existing datasets** from fact-checking organizations
2. **Augment with authoritative sources** (WHO, CDC, PubMed) as positive examples
3. **Collect real-world examples** from social media (with proper labeling)
4. **Validate with medical experts** before using in production
5. **Continuously update** as new misinformation patterns emerge

## Legal and Ethical Considerations

- **Respect copyright**: Don't scrape copyrighted content without permission
- **Privacy**: Anonymize personal information in datasets
- **Terms of Service**: Follow platform terms when collecting data
- **Bias**: Ensure diverse representation in training data
- **Transparency**: Document data sources and labeling process

## Tools for Data Collection

- **BeautifulSoup/Scrapy**: Web scraping
- **Tweepy**: Twitter API access
- **PRAW**: Reddit API access
- **PubMed API**: Medical literature access
- **Label Studio**: Data labeling platform

## Next Steps

1. Start with publicly available fact-checking datasets
2. Use PubMed API to collect accurate medical information
3. Create a balanced dataset (50% accurate, 50% misinformation)
4. Implement active learning to improve model with expert feedback
5. Regularly retrain with new data as misinformation evolves



