"""
Message Parser for Co-Investigator App
Parses agent messages to extract structured data
"""
import re
from typing import Dict, List, Any, Tuple


def parse_agent_message(content: str) -> Dict[str, Any]:
    """
    Parse agent message to extract structured data for interactive display
    
    Returns:
        dict with sections: {
            'text': str,
            'researchers': List[Dict],
            'genes': List[Dict],
            'preprints': List[Dict],
            'concepts': List[str],
            'graphs': List[Dict],
            'metrics': Dict
        }
    """
    result = {
        'text': content,
        'researchers': [],
        'genes': [],
        'preprints': [],
        'concepts': [],
        'graphs': [],
        'metrics': {},
        'sections': []
    }
    
    # Extract researchers table
    researcher_pattern = r'Name\s+Citations\s+H_Index\s+Institution\s+\n-+\n((?:.+\n)+)'
    match = re.search(researcher_pattern, content)
    if match:
        lines = match.group(1).strip().split('\n')
        for line in lines:
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 4:
                result['researchers'].append({
                    'name': parts[0],
                    'cited_by_count': int(parts[1]) if parts[1].isdigit() else 0,
                    'h_index': int(parts[2]) if parts[2].isdigit() else 0,
                    'last_known_institution': parts[3]
                })
    
    # Extract gene table
    gene_pattern = r'Gene_Symbol\s+Disease_Label\s+MOI\s+Classification\s+\n-+\n((?:.+\n)+?)(?:\n\n|$)'
    match = re.search(gene_pattern, content)
    if match:
        lines = match.group(1).strip().split('\n')
        for line in lines:
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 4:
                result['genes'].append({
                    'Gene_Symbol': parts[0],
                    'Disease_Label': parts[1],
                    'MOI': parts[2],
                    'Classification': parts[3]
                })
    
    # Extract preprints
    preprint_pattern = r'Title\s+Authors\s+Date\s+Source\s+\n-+\n((?:.+\n)+?)(?:\n\n|👁️)'
    match = re.search(preprint_pattern, content, re.DOTALL)
    if match:
        lines = match.group(1).strip().split('\n')
        for line in lines:
            if line.strip():
                # Simple parsing - title is first long part
                parts = re.split(r'\s{2,}', line.strip())
                if len(parts) >= 3:
                    result['preprints'].append({
                        'Title': parts[0],
                        'Authors': parts[1] if len(parts) > 1 else 'Unknown',
                        'Date': parts[2] if len(parts) > 2 else 'Unknown',
                        'Source': parts[3] if len(parts) > 3 else 'Unknown'
                    })
    
    # Extract concepts
    concept_pattern = r'🧠 Key Scientific Concepts:\s*\n((?:.+\n)+?)(?:\n=|$)'
    match = re.search(concept_pattern, content)
    if match:
        lines = match.group(1).strip().split('\n')
        result['concepts'] = [line.strip() for line in lines if line.strip()]
    
    # Extract image paths
    img_pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
    for match in re.finditer(img_pattern, content):
        result['graphs'].append({
            'alt_text': match.group(1),
            'path': match.group(2)
        })
    
    # Extract metrics from OBSERVE statements
    observe_pattern = r'👁️\s+OBSERVE:\s*(.+?)(?:\n\n|$)'
    match = re.search(observe_pattern, content, re.DOTALL)
    if match:
        result['metrics']['observe'] = match.group(1).strip()
    
    # Extract step info
    step_pattern = r'🔎 REASON:\s*Step\s*(\d+)/(\d+)\s*—\s*(.+?)\n'
    match = re.search(step_pattern, content)
    if match:
        result['metrics']['step'] = int(match.group(1))
        result['metrics']['total_steps'] = int(match.group(2))
        result['metrics']['step_description'] = match.group(3).strip()
    
    # Extract results counts
    results_patterns = [
        (r'(\d+)\s+researchers', 'researchers_count'),
        (r'(\d+)\s+total.*gene', 'genes_count'),
        (r'(\d+)\s+total.*preprint', 'preprints_count'),
        (r'(\d+)\s+knowledge connections', 'knowledge_connections')
    ]
    
    for pattern, key in results_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            result['metrics'][key] = int(match.group(1))
    
    return result


def extract_sections(content: str) -> List[Tuple[str, str]]:
    """
    Extract major sections from agent message
    
    Returns:
        List of (section_name, section_content) tuples
    """
    sections = []
    
    # Match section headers (with emoji or all caps)
    section_pattern = r'((?:🧠|📋|🔄|🔎|⚡|📊|👁️|🤖)\s*[A-Z\s—:]+)\n'
    
    parts = re.split(section_pattern, content)
    
    current_section = None
    current_content = []
    
    for part in parts:
        if re.match(section_pattern, part):
            if current_section:
                sections.append((current_section, '\n'.join(current_content)))
                current_content = []
            current_section = part.strip()
        else:
            if part.strip():
                current_content.append(part)
    
    if current_section and current_content:
        sections.append((current_section, '\n'.join(current_content)))
    
    return sections


def simplify_message_for_display(content: str) -> str:
    """
    Simplify message content by removing verbose tables and replacing with summaries
    """
    # Remove large ASCII tables
    content = re.sub(r'[-=]{20,}', '', content)
    
    # Remove verbose headers
    content = re.sub(r'══+', '', content)
    content = re.sub(r'──+', '', content)
    
    # Simplify multiple newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content.strip()
