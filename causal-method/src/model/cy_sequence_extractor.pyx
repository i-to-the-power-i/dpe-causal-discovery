# cython: language_level=3
import pandas as pd
from Bio import SeqIO, Entrez
import cython
import time
import ssl
from urllib.error import URLError

ssl._create_default_https_context = ssl._create_unverified_context

Entrez.email = '2025proj022@goa.bits-pilani.ac.in'
Entrez.tool = 'gen-sequence-extractor'

cpdef str single_sequence_extractor(str accession_id):
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
    cdef int attempts = 3
    cdef int i
    
    for i in range(attempts):
        try:
            handler = Entrez.efetch(db='nucleotide', id=accession_id, 
                                   rettype='genbank', retmode='text')
            record = SeqIO.read(handler, 'genbank')
            handler.close()
            return str(record.seq)
            
        except (URLError, Exception) as e:
            if i < attempts - 1:
                time.sleep(2)    # Wait 2 seconds before retrying
                continue
            else:
                print(f"Permanent Error for {accession_id}: {e}")
                return ""

cpdef object multi_sequence_extractor(list accession_ids):
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
    cdef int n = len(accession_ids)
    cdef int i
    cdef list sequences = []
    cdef str current_id
    cdef str seq_result

    for i in range(n):
        current_id = accession_ids[i]
        seq_result = single_sequence_extractor(current_id)
        sequences.append(seq_result)
        # ncbi allows to access data after five minutes buffer
        time.sleep(5)

    # Dictionary and DataFrame creation remain high-level Python operations
    results = {
        'accession_id': accession_ids,
        'sequence': sequences
    }

    return pd.DataFrame(results)