import re
import requests
import pandas as pd
from io import StringIO

KGMETA_SPARQLendpointUrl = "http://206.12.98.118:8890/sparql/" ## The trained GNN Models Meta-data KG
SPARQLendpointUrl_dict={"biokg":"http://206.12.97.2:8890/sparql/"} ## SPARQL endpoint per KG
NamedGraph_URI_dict={"biokg":"http://www.biokg.com"} ## Named graph URI per KG
generic_ignore_predicates= ['http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://www.w3.org/2002/07/owl#sameAs',
    'http://schema.org/image', 'http://schema.org/sameAs',
    'http://www.w3.org/2000/01/rdf-schema#comment', 'http://schema.org/logo',
    'http://schema.org/url', 'http://www.w3.org/2002/07/owl#differentFrom',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#first',
    'http://purl.org/spar/datacite/hasIdentifier', 'http://www.w3.org/2000/01/rdf-schema#seeAlso',
    'http://xmlns.com/foaf/0.1/thumbnail', 'http://www.w3.org/2002/07/owl#differentFrom',
    'http://xmlns.com/foaf/0.1/isPrimaryTopicOf', 'http://purl.org/dc/elements/1.1/type',
    'http://xmlns.com/foaf/0.1/primaryTopic', 'http://xmlns.com/foaf/0.1/logo',
    'http://purl.org/dc/elements/1.1/rights', 'http://www.w3.org/2000/01/rdf-schema#label',
    'http://dbpedia.org/ontology/thumbnail', 'http://dbpedia.org/ontology/wikiPageID',
    'http://purl.org/dc/terms/subject', 'http://purl.org/linguistics/gold/hypernym',
    'http://xmlns.com/foaf/0.1/name', 'http://dbpedia.org/ontology/wikiPageRevisionID',
    'http://dbpedia.org/ontology/wikiPageRedirects', 'http://www.w3.org/ns/prov#wasDerivedFrom',
    'http://dbpedia.org/ontology/wikiPageExternalLink', 'http://dbpedia.org/ontology/abstract',
    'http://xmlns.com/foaf/0.1/depiction', 'http://xmlns.com/foaf/0.1/homepage',
    'http://dbpedia.org/property/no', 'https://dbpedia.org/property/oclc',
    'http://dbpedia.org/ontology/dcc'] ## List of predicats to ignore

def defrag_uri(uri):
    pattern = r'[#/]([^/#]+)$'
    match = re.search(pattern, uri)
    if match:
        name = match.group(1)
        p2 = re.compile(r"([a-z0-9])([A-Z])")
        name = p2.sub(r"\1 \2", name)
        return name
    else:
        return uri
def get_triple_representation(triple):
    output = list()
    for el in triple:
        output.append(defrag_uri(str(el)))
        # output.append(str(el))
    return tuple(output)
def serialize_subgraph(triples, ignore_subject=False):
    triple_list = []
    for triple in triples:
        # This converts triple to tuple you can skip it if the triple is a tuple
        triple = get_triple_representation(triple)
        # print(triple)
        if ignore_subject:
          triple_list.append(triple[1:])
        else:
          triple_list.append(triple)
    if ignore_subject:
      double_quote_triples = [f'("{triple[0]}", "{triple[1]}")' for triple in triple_list]
    else:
      double_quote_triples = [f'("{triple[0]}", "{triple[1]}", "{triple[2]}")' for triple in triple_list]
    return '[' + ', '.join(double_quote_triples) + ']'
def executeSparqlQuery(query,SPARQLendpointUrl,firstRowHeader=True):
    """
    Execute sparql query through dopost and return results in form of datafarme.
    :param query:the sparql query string.
    :type query: str
    :param firstRowHeader: wherther to assume frist line as the dataframe columns header.

    """
    body = {'query': query}
    headers = {
        # 'Content-Type': 'application/sparql-update',
        'Content-Type': "application/x-www-form-urlencoded",
        'Accept-Encoding': 'gzip',
        'Accept':  ('text/tab-separated-values; charset=UTF-8')
        # 'Accept': 'text/tsv'
    }
    r = requests.post(SPARQLendpointUrl, data=body, headers=headers)
    if firstRowHeader:
        res_df=pd.DataFrame([x.split('\t') for x in r.text.split('\n')[1:] if x],columns=r.text.split('\n')[0].replace("\"","").replace("?","").split('\t'))
    else:
        res_df=pd.DataFrame([x.split('\t') for x in r.text.split('\n')])
    for col in res_df.columns:
      res_df[col] = res_df[col].str.strip('"')
    return res_df
def pretty_format_sparql(query):
    keywords = ['SELECT', 'WHERE', 'PREFIX', 'LIMIT', 'ORDER BY', 'GROUP BY']
    formatted_query = query
    for keyword in keywords:
        formatted_query = formatted_query.replace(keyword, f"\n{keyword}")

    return formatted_query.strip()