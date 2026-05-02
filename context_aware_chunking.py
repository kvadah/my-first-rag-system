import re 
from pypdf import PdfReader



def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("- ", "") 
    return text.strip()



def extract_pages(file_path):
  reader =PdfReader(file_path)
  current_section="UNKNOWN"
  for i, page in enumerate(reader.pages):
    text =page.extract_text()
    if text:
      lines = text.split("\n")
      
      for line in lines[:10]: 
          if is_section_header(line):
              current_section = line.strip()
              print(line)
              break

      yield {
          "page": i + 1,
          "text": text,
          "section": current_section
      }
def extract_articles(file_path):
   reader = PdfReader(file_path)
   text=""
   for i, page in enumerate(reader.pages):
       # text += f"\n[PAGE {i+1}]\n"
        text += page.extract_text()
   pattern = r"(Article\s+\d+.*?)(?=Article\s+\d+|$)"
   articles = re.findall(pattern, text, re.DOTALL)
   return articles


def structured_articles(file_path):
  structured_articles = []
  articles = extract_articles(file_path)
  for article in articles:
    lines = article.strip().split("\n")
    title_line = lines[0]
    structured_articles.append({
        "title": title_line,
        "content": article
    })
  return structured_articles

def is_section_header(line):
    line = line.strip()

    if len(line) < 5:
        return False

    if line.isupper() and len(line.split()) < 10:
        return True

    if re.match(r'^\d+(\.\d+)*\s+[A-Z]', line):
        return True

    return False


def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

def advanced_smart_chunk(file_path,max_tokens=400,overlap=70):
  chunks=[]
  articles = structured_articles(file_path)
 
   for article in articles:
     text = clean_text(article["text"])
     title = article["title"]
     article_number=extract_article_number(title) 
     sentences = split_sentences(text)
     current_chunk = []
     current_length = 0
     for sentence in sentences:
        sentence_len = len(sentence.split())

        if current_length + sentence_len <= max_tokens:
            current_chunk.append(sentence)
            current_length += sentence_len
        else:
            chunk_text = " ".join(current_chunk)

            chunks.append({
                "text": chunk_text,
               "metadata": {
               "article": article["title"],
               "article_number":article_number
            }
            })

            overlap_sentences = current_chunk[-3:]
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s.split()) for s in current_chunk)

        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "metadata": {
                "article": article["title"],
                "article_number":article_number
            }
            })

  return chunks