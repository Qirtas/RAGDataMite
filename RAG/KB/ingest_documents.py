import pickle
import os
import pandas as pd
import logging
import json
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from langchain.docstore.document import Document
from typing import List, Dict, Any, Optional

def ingest_documents(output_dir="RAG/ProcessedDocuments"):

    csv_files = {
        'KPIs': 'RAG/Content/ProcessedFiles/clean_KPIs.csv',
        'Objectives': 'RAG/Content/ProcessedFiles/clean_Objectives.csv',
        'BSC_families': 'RAG/Content/ProcessedFiles/clean_BSC_families.csv',
        'BSC_subfamilies': 'RAG/Content/ProcessedFiles/clean_BSC_subfamilies.csv',
        'Criteria': 'RAG/Content/ProcessedFiles/clean_Criteria.csv'
    }

    # missing_files = [path for path in csv_files.values() if not os.path.exists(path)]
    # if missing_files:
    #     print(f"Error: The following files are missing: {missing_files}")
    #     return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    all_documents = load_all_csvs_as_documents(csv_files)
    print(f"Loaded {len(all_documents)} total documents")

    doc_types = {}
    for doc in all_documents:
        doc_type = doc.metadata.get("type", "Unknown")
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

    print("\nDocuments by type:")
    for doc_type, count in doc_types.items():
        print(f"- {doc_type}: {count} documents")

    output_path = os.path.join(output_dir, "all_documents.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(all_documents, f)

    print(f"\nSaved {len(all_documents)} documents to {output_path}")

    return all_documents



def load_all_csvs_as_documents(csv_files: Dict[str, str]) -> List[Document]:

    all_documents = []

    for file_type, file_path in csv_files.items():
        documents = load_csv_as_documents(file_path, file_type)
        all_documents.extend(documents)
        logger.info(f"Added {len(documents)} documents from {file_path}")

    logger.info(f"Loaded {len(all_documents)} total documents from {len(csv_files)} files")
    return all_documents


def clean_value(value: Any) -> str:

    if pd.isna(value) or value is None:
        return ""

    if not isinstance(value, str):
        return str(value).strip()

    if not value.strip():
        return ""


    if value.startswith('[') and value.endswith(']'):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return ", ".join(parsed)
        except Exception:
            # If parsing fails, try to clean it manually
            value = value.strip('[]')
            value = value.replace('\\\\', '').replace('\\"', '"')
            parts = [p.strip().strip('"\'') for p in value.split(',')]
            return ", ".join(filter(None, parts))

    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1].strip()

    value = value.replace('\\\\', '').replace('\\"', '"')

    return value

def process_kpis(df: pd.DataFrame, source_file: str) -> List[Document]:

    documents = []

    logger.info(f"KPI DataFrame columns: {df.columns.tolist()}")

    for idx, row in df.iterrows():
        if idx < 3:
            logger.info(f"Raw row {idx}: {row.to_dict()}")

        name = clean_value(row.get('name', ''))

        alt_names_col = 'alternative_names' if 'alternative_names' in df.columns else ' alternative_names'
        subfamilies_col = 'bsc_subfamilies' if 'bsc_subfamilies' in df.columns else ' bsc_subfamilies'
        short_def_col = 'short_definition' if 'short_definition' in df.columns else ' short_definition'
        explanation_col = 'explanation' if 'explanation' in df.columns else ' explanation'

        alt_names = clean_value(row.get(alt_names_col, ''))
        subfamilies = clean_value(row.get(subfamilies_col, ''))
        short_def = clean_value(row.get(short_def_col, ''))
        explanation = clean_value(row.get(explanation_col, ''))

        content = f"""
KPI Name: {name}
Alternative Names: {alt_names}
BSC Subfamily: {subfamilies}
Short Definition: {short_def}
Explanation: {explanation}
        """.strip()

        doc = Document(
            page_content=content,
            metadata={
                "source": source_file,
                "type": "KPI",
                "name": name
            }
        )

        documents.append(doc)

    # Log a sample for verification
    if documents:
        logger.info(f"Sample KPI document content: {documents[0].page_content[:100]}...")

    logger.info(f"Created {len(documents)} KPI documents")
    return documents

def process_objectives(df: pd.DataFrame, source_file: str) -> List[Document]:

    documents = []

    logger.info(f"Objectives DataFrame columns: {df.columns.tolist()}")

    for _, row in df.iterrows():
        name_col = 'name' if 'name' in df.columns else ' name'
        short_def_col = 'short_definition' if 'short_definition' in df.columns else ' short_definition'
        explanation_col = 'explanation' if 'explanation' in df.columns else ' explanation'

        name = clean_value(row.get(name_col, ''))
        short_def = clean_value(row.get(short_def_col, ''))
        explanation = clean_value(row.get(explanation_col, ''))

        content = f"""
Objective Name: {name}
Short Definition: {short_def}
Explanation: {explanation}
        """.strip()

        doc = Document(
            page_content=content,
            metadata={
                "source": source_file,
                "type": "Objective",
                "name": name
            }
        )

        documents.append(doc)

    logger.info(f"Created {len(documents)} Objective documents")
    return documents

def process_bsc_families(df: pd.DataFrame, source_file: str) -> List[Document]:

    documents = []

    logger.info(f"BSC Families DataFrame columns: {df.columns.tolist()}")

    for _, row in df.iterrows():
        name_col = 'name' if 'name' in df.columns else ' name'
        short_name_col = 'short_name' if 'short_name' in df.columns else ' short_name'
        short_def_col = 'short_definition' if 'short_definition' in df.columns else ' short_definition'
        explanation_col = 'explanation' if 'explanation' in df.columns else ' explanation'

        name = clean_value(row.get(name_col, ''))
        short_name = clean_value(row.get(short_name_col, ''))
        short_def = clean_value(row.get(short_def_col, ''))
        explanation = clean_value(row.get(explanation_col, ''))

        content = f"""
BSC Family Name: {name}
Short Name: {short_name}
Short Definition: {short_def}
Explanation: {explanation}
        """.strip()

        doc = Document(
            page_content=content,
            metadata={
                "source": source_file,
                "type": "BSC Family",
                "name": name
            }
        )

        documents.append(doc)

    logger.info(f"Created {len(documents)} BSC Family documents")
    return documents

def process_bsc_subfamilies(df: pd.DataFrame, source_file: str) -> List[Document]:

    documents = []

    logger.info(f"BSC Subfamilies DataFrame columns: {df.columns.tolist()}")

    for _, row in df.iterrows():
        name_col = 'name' if 'name' in df.columns else ' name'
        family_col = 'bsc_family' if 'bsc_family' in df.columns else ' bsc_family'
        short_def_col = 'short_definition' if 'short_definition' in df.columns else ' short_definition'
        explanation_col = 'explanation' if 'explanation' in df.columns else ' explanation'

        name = clean_value(row.get(name_col, ''))
        family = clean_value(row.get(family_col, ''))
        short_def = clean_value(row.get(short_def_col, ''))
        explanation = clean_value(row.get(explanation_col, ''))

        content = f"""
BSC Subfamily Name: {name}
BSC Family: {family}
Short Definition: {short_def}
Explanation: {explanation}
        """.strip()

        doc = Document(
            page_content=content,
            metadata={
                "source": source_file,
                "type": "BSC Subfamily",
                "name": name
            }
        )

        documents.append(doc)

    logger.info(f"Created {len(documents)} BSC Subfamily documents")
    return documents

def process_criteria(df: pd.DataFrame, source_file: str) -> List[Document]:

    documents = []

    logger.info(f"Criteria DataFrame columns: {df.columns.tolist()}")

    for _, row in df.iterrows():
        name_col = 'name' if 'name' in df.columns else ' name'
        explanation_col = 'explanation' if 'explanation' in df.columns else ' explanation'

        name = clean_value(row.get(name_col, ''))
        explanation = clean_value(row.get(explanation_col, ''))

        content = f"""
Criteria Name: {name}
Explanation: {explanation}
        """.strip()

        doc = Document(
            page_content=content,
            metadata={
                "source": source_file,
                "type": "Criteria",
                "name": name
            }
        )

        documents.append(doc)

    logger.info(f"Created {len(documents)} Criteria documents")
    return documents

def load_csv_as_documents(file_path: str, file_type: Optional[str] = None) -> List[Document]:

    logger.info(f"Loading CSV file: {file_path}")

    if file_type is None:
        file_name = os.path.basename(file_path)
        if 'KPI' in file_name:
            file_type = 'KPIs'
        elif 'Objective' in file_name:
            file_type = 'Objectives'
        elif 'BSC_families' in file_name:
            file_type = 'BSC_families'
        elif 'BSC_subfamilies' in file_name:
            file_type = 'BSC_subfamilies'
        elif 'Criteria' in file_name:
            file_type = 'Criteria'
        else:
            file_type = 'Unknown'
            logger.warning(f"Could not infer file type for {file_path}, using 'Unknown'")

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        logger.info(f"Successfully loaded {len(df)} rows from {file_path}")

        if file_type == 'KPIs':
            return process_kpis(df, file_path)
        elif file_type == 'Objectives':
            return process_objectives(df, file_path)
        elif file_type == 'BSC_families':
            return process_bsc_families(df, file_path)
        elif file_type == 'BSC_subfamilies':
            return process_bsc_subfamilies(df, file_path)
        elif file_type == 'Criteria':
            return process_criteria(df, file_path)
        else:
            logger.warning(f"Unknown file type: {file_type}")
            return []

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return []