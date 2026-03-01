import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO, Entrez
import time

# email set-up for verification
Entrez.email = '2025proj022@goa.bits-pilani.ac.in'

# name for the entrez-tool
Entrez.tool = 'gen-sequence-extractor-2'

def single_sequence_extractor(accession_id: str):
  '''
    Fetches a single DNA/RNA sequence from the NCBI Nucleotide database.

    Args:
        accession_id (str): A unique NCBI identifier (e.g., 'NC_045512').

    Returns:
        str: The genetic sequence as a string if successful.
        None: If the sequence could not be retrieved due to a network or ID error.

    Example:
        >>> seq = single_sequence_extractor("MN908947")
        >>> print(seq[:10])
        ATTAAAGGTT
  '''

  # # term to search
  # search_term = 'saras-cov-2'

  try:
    # efetch object handler
    handler = Entrez.efetch(db='nucleotide', id=accession_id, rettype='genbank', retmode='text')
    record = SeqIO.read(handler, 'genbank')
    handler.close()
  except:
    print(f'Error in accessing accession id {accession_id}.')
    return ''

  sequence = record.seq
  return str(sequence)

def multi_sequence_extractor(accession_ids: list):
  '''
    Retrieves multiple sequences from NCBI for a list of accession IDs.

    This function iterates through the provided list and calls 
    `single_sequence_extractor` for each ID.

    Args:
        accession_ids (list of str): A list containing NCBI accession strings.

    Returns:
        list of str: A list of sequences. If an ID failed, a None value 
            is included in the list at that index.

    Note:
        For very large lists, consider using Entrez.epost to avoid 
        multiple individual network requests.
  '''
  sequences = []
  for accession_id in accession_ids:
    sequence = single_sequence_extractor(accession_id)
    sequences.append(sequence)
    # ncbi allows to access data after five minutes buffer
    # time.sleep(5)

  results = {
    'accession_id': accession_ids,
    'sequence': sequences
  }

  return pd.DataFrame(results)

# # testing the functions

# # single sequence extractor
# sequence = single_sequence_extractor("NC_045512.2")
# print(sequence[:10])

# multiple sequence extractor
# id_list = ['PX583645', 'PX575891', 'PV998713', 'PV998714', 'PV998716', 'PV998717', 'PV714139', 'MZ317934']
# df = multi_sequence_extractor(id_list)
# print(df)