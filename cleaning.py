import pdfplumber
import re
import sys
from tqdm import tqdm  
def clean_text(text):
## Clean text by removing extra whitespaces and newlines
    text = text.strip()
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)    
    text = re.sub(r'\n+', '\n', text)
    
    return text

def extract_text_from_pdf(pdf_path):
## Extract text from a PDF file
    all_pages = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages, desc="Parsing pages"):
            page_text = page.extract_text()
            if not page_text:
                continue 
            
            lines = page_text.split('\n')
            
            if len(lines) > 2:
                content_lines = lines[1:-1]
            else:
                content_lines = lines
            
            page_content = "\n".join(content_lines)
            page_content = clean_text(page_content)
            
            if page_content:
                all_pages.append(page_content)
    
    full_text = "\n\n".join(all_pages)
    return full_text

def split_into_chunks(text, delimiter="\n\n"):
## Split text into chunks based on a delimiter
    chunks = text.split(delimiter)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return chunks

def main(pdf_path, output_txt):
    print(f"Extracting text from {pdf_path} ...")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    print("Splitting text into chunks ...")
    chunks = split_into_chunks(extracted_text)
    
    print(f"Writing output to {output_txt} ...")
    with open(output_txt, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n")
    
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_pdf.py input.pdf output.txt")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    output_file = sys.argv[2]
    main(pdf_file, output_file)
