#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import wikipediaapi
import re


# In[90]:


import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    user_agent='Tyler/DSAN5400_project',
    language='en'
)

page = wiki.page('List of last words')

if page.exists():

    for section in page.sections:
        print(f"Section: {section.title}")
        print(f"Level: {section.level}")  
        
        for subsection in section.sections:
            print(f"  Subsection: {subsection.title}")


# # 21st Century Last Words Scraping

# In[91]:


page = wiki.page('List of last words (21st century)')

if page.exists():

    for section in page.sections:
        print(f"Section: {section.title}")
        print(f"Level: {section.level}")  
        print("-" * 50)
        
        for subsection in section.sections:
            print(f"  Subsection: {subsection.title}")



# In[92]:


text = ""
page = wiki.page('List of last words (21st century)')
if page.exists():

    for section in page.sections[:3]:  
        text += section.text
        


# In[131]:


def parse_text(text):
    entries = []

    def parse_entry(entry_text):
        lines = [l.strip() for l in entry_text.split('\n') if l.strip()]

        if len(lines) < 2:
            return None
        
        rest = ' '.join(lines[1:]).lstrip('—-–').strip()
        date = re.search(r'\(([^)]*(?:\d{4}|c\.\s*\d+|(?:\d+th|c\.\s*\d+)\s*century)(?:\s*(?:BC|AD))?[^)]*)\)', rest)
        name = rest.split(',')[0]

        if date:
            before_date = rest[:date.start()].strip()
            title = before_date.split(',', 1)[1].strip() if ',' in before_date else ""
            context = rest[date.end():].strip(', .')
        else:
            title = rest.split(',', 1)[1].strip() if ',' in rest else ""
            context = ""
        
        return {
            'name': name,
            'title': title,
            'quote': lines[0].strip('"''"'),
            'date': date.group(1) if date else "",
            'context': context
        }

    blocks = re.split(r'\n(?=[""""])', text)

    for block in blocks:
        if block.strip():
            parsed = parse_entry(block)
            if parsed:
                entries.append(parsed)

    return pd.DataFrame(entries)

df = parse_text(text)


# In[94]:


df.shape


# In[95]:


df.to_csv('data/raw_data/last_words_21st_century.csv', index=False)


# # 20th

# In[96]:


page = wiki.page('List of last words (20th century)')

if page.exists():

    for section in page.sections:
        print(f"Section: {section.title}")
        print(f"Level: {section.level}")  
        print("-" * 50)
        
        for subsection in section.sections:
            print(f"  Subsection: {subsection.title}")


# In[97]:


text_20 = ""
page = wiki.page('List of last words (20th century)')
if page.exists():

    for section in page.sections[:10]:  
        text_20 += section.text
        


# In[98]:


print(len(text_20))


# In[99]:


df = parse_text(text_20)


# In[100]:


df.shape


# In[101]:


df.to_csv('data/raw_data/last_words_20th_century.csv', index=False)


# # 19th

# In[102]:


page = wiki.page('List of last words (19th century)')

if page.exists():

    for section in page.sections:
        print(f"Section: {section.title}")
        print(f"Level: {section.level}")  
        print("-" * 50)
        
        for subsection in section.sections:
            print(f"  Subsection: {subsection.title}")


# In[103]:


text_19 = ""
page = wiki.page('List of last words (19th century)')
if page.exists():

    for section in page.sections[:10]:  
        text_19 += section.text

df = parse_text(text_19)

df.to_csv('data/raw_data/last_words_19th_century.csv', index=False)


# # 18th

# In[104]:


page = wiki.page('List of last words (18th century)')

if page.exists():

    for section in page.sections:
        print(f"Section: {section.title}")
        print(f"Level: {section.level}")  
        print("-" * 50)
        
        for subsection in section.sections:
            print(f"  Subsection: {subsection.title}")


# In[105]:


text_18 = ""
page = wiki.page('List of last words (18th century)')
if page.exists():

    for section in page.sections[:10]:  
        text_18 += section.text

df = parse_text(text_18)

df.to_csv('data/raw_data/last_words_18th_century.csv', index=False)


# # Other

# In[132]:


page = wiki.page('List of last words')

if page.exists():
    chronological_section = page.sections[0]
    text_pre5_to_17 = ""
    for subsection in chronological_section.sections[:5]:
        text_pre5_to_17 += subsection.text
    
    text_ironic = page.sections[1].text
    text_notable = page.sections[2].text


# In[133]:


df_pre5_to_17 = parse_text(text_pre5_to_17)
df_ironic = parse_text(text_ironic)
df_notable = parse_text(text_notable)


# In[136]:


df_notable.tail()


# In[137]:


df_pre5_to_17.to_csv('data/raw_data/last_words_pre5_to_17_century.csv', index=False)
df_ironic.to_csv('data/raw_data/last_words_ironic.csv', index=False)
df_notable.to_csv('data/raw_data/last_words_notable.csv', index=False)

