from Utils.utils import executeSparqlQuery
from Utils.utils import generic_ignore_predicates
from Utils.utils import serialize_subgraph
import pandas as pd
import numpy as np
import pickle

KGMETA_SPARQLendpointUrl = "http://206.12.98.118:8890/sparql/" ## The trained GNN Models Meta-data KG
SPARQLendpointUrl_dict={"biokg":"http://206.12.97.2:8890/sparql/",
                        "yago4":"http://206.12.97.2:8890/sparql/",
                        "crunchbase":"http://206.12.97.2:8890/sparql/",
                        "linkedIMDB":"http://206.12.98.118:8890/sparql/",
                        "arxiv2023":"http://206.12.98.118:8890/sparql/",
                        "ogbnArxiv":"http://206.12.98.118:8890/sparql/",
                        "ogbnProduct":"http://206.12.98.118:8890/sparql/"} ## SPARQL endpoint per KG
NamedGraph_URI_dict={"biokg":"http://www.biokg.com",
                     "yago4":"https://yago-knowledge.org",
                     "crunchbase":"http://crunchbase-dump-2015-10",
                     "linkedIMDB":"https://linkedmdb.org",
                     "arxiv2023": "http://arxiv2023.org",
                     "ogbnArxiv": "http://ogbn_arxiv.org/",
                     "ogbnProduct": "http://ogbn_product.org/"
                     } ## Named graph URI per KG
kg_metadata={"biokg":["chemical","Biomedical knoweldge graph"],
             "yago4":["","YAGO4 knowledge graph"],
             "crunchbase":["Investement","investments knowledge graph (crunchbase)"],
             "linkedIMDB":["IMDB","Linked IMDB knowledge graph"],
             "arxiv2023": ["Academic","Arxiv paper Knowledge Graph"],
             "ogbnArxiv": ["Academic","Arxiv paper Knowledge Graph"],
             "ogbnProduct": ["Commercial","Amazon Product Knowledge Graph"],
             } ## Named graph URI per KG

class biokg_GLOW_Bench:
  drug_dict_pred=None
  drug_dict_pred_class=None
  by_pubmid_BGP=None
  def __init__(self,SPARQLendpointUrl):
    drug_superclass_class_lst=["http://www.biokg.com/DRUG_SUPERCLASS/Alkaloids_and_derivatives",
    "http://www.biokg.com/DRUG_SUPERCLASS/Benzenoids",
    "http://www.biokg.com/DRUG_SUPERCLASS/Homogeneous_non_metal_compounds",
    "http://www.biokg.com/DRUG_SUPERCLASS/Lipids_and_lipid_like_molecules",
    "http://www.biokg.com/DRUG_SUPERCLASS/Mixed_metal_non_metal_compounds",
    "http://www.biokg.com/DRUG_SUPERCLASS/Nucleosides__nucleotides__and_analogues",
    "http://www.biokg.com/DRUG_SUPERCLASS/Organic_Acids",
    "http://www.biokg.com/DRUG_SUPERCLASS/Organic_acids_and_derivatives",
    "http://www.biokg.com/DRUG_SUPERCLASS/Organic_nitrogen_compounds",
    "http://www.biokg.com/DRUG_SUPERCLASS/Organic_oxygen_compounds",
    "http://www.biokg.com/DRUG_SUPERCLASS/Organoheterocyclic_compounds",
    "http://www.biokg.com/DRUG_SUPERCLASS/Organosulfur_compounds",
    "http://www.biokg.com/DRUG_SUPERCLASS/Phenylpropanoids_and_polyketides"]
    ##################################################################################
    drug_class_class_lst=["http://www.biokg.com/DRUG_CLASS/Azoles",
    "http://www.biokg.com/DRUG_CLASS/Benzene_and_substituted_derivatives",
    "http://www.biokg.com/DRUG_CLASS/Benzimidazoles",
    "http://www.biokg.com/DRUG_CLASS/Benzodiazepines",
    "http://www.biokg.com/DRUG_CLASS/Benzothiazines",
    "http://www.biokg.com/DRUG_CLASS/Carboxylic_Acids_and_Derivatives",
    "http://www.biokg.com/DRUG_CLASS/Carboxylic_acids_and_derivatives",
    "http://www.biokg.com/DRUG_CLASS/Diazanaphthalenes",
    "http://www.biokg.com/DRUG_CLASS/Diazinanes",
    "http://www.biokg.com/DRUG_CLASS/Diazines",
    "http://www.biokg.com/DRUG_CLASS/Fatty_Acyls",
    "http://www.biokg.com/DRUG_CLASS/Imidazopyrimidines",
    "http://www.biokg.com/DRUG_CLASS/Indoles_and_derivatives",
    "http://www.biokg.com/DRUG_CLASS/Lactams",
    "http://www.biokg.com/DRUG_CLASS/Naphthalenes",
    "http://www.biokg.com/DRUG_CLASS/Organonitrogen_compounds",
    "http://www.biokg.com/DRUG_CLASS/Organooxygen_compounds",
    "http://www.biokg.com/DRUG_CLASS/Peptidomimetics",
    "http://www.biokg.com/DRUG_CLASS/Phenol_ethers",
    "http://www.biokg.com/DRUG_CLASS/Phenols",
    "http://www.biokg.com/DRUG_CLASS/Piperidines",
    "http://www.biokg.com/DRUG_CLASS/Prenol_lipids",
    "http://www.biokg.com/DRUG_CLASS/Purine_nucleosides",
    "http://www.biokg.com/DRUG_CLASS/Purine_nucleotides",
    "http://www.biokg.com/DRUG_CLASS/Pyridines_and_derivatives",
    "http://www.biokg.com/DRUG_CLASS/Pyrimidine_nucleotides",
    "http://www.biokg.com/DRUG_CLASS/Quinolines_and_derivatives",
    "http://www.biokg.com/DRUG_CLASS/Steroids_and_steroid_derivatives",
    "http://www.biokg.com/DRUG_CLASS/Stilbenes"]
    ##################################################################################
    drug_kingdom_class_lst=["http://www.biokg.com/DRUG_KINGDOM/Inorganic_compounds",
    "http://www.biokg.com/DRUG_KINGDOM/Organic_compounds"]
    ##################################################################################
    protien_keyword_class_lst=["3D-structure",
    "Alternative splicing",
    "Cell membrane",
    "Cytoplasm",
    "Glycoprotein",
    "Membrane",
    "Metal-binding",
    "Nucleus",
    "Phosphoprotein",
    "Proteomics identification",
    "Reference proteome",
    "Repeat",
    "Signal",
    "Transmembrane",
    "Transmembrane helix"]
    protien_keyword_class_dict= dict(zip(protien_keyword_class_lst, protien_keyword_class_lst))
    # protien_keyword_class_dict
    ###################################################################################
    protien_SPECIES_class_lst=["http://www.biokg.com/protein_SPECIES/ARATH",
    "http://www.biokg.com/protein_SPECIES/BACSU",
    "http://www.biokg.com/protein_SPECIES/BOVIN",
    "http://www.biokg.com/protein_SPECIES/CHICK",
    "http://www.biokg.com/protein_SPECIES/DANRE",
    "http://www.biokg.com/protein_SPECIES/DICDI",
    "http://www.biokg.com/protein_SPECIES/ECO57",
    "http://www.biokg.com/protein_SPECIES/ECOLI",
    "http://www.biokg.com/protein_SPECIES/HUMAN",
    "http://www.biokg.com/protein_SPECIES/MOUSE",
    "http://www.biokg.com/protein_SPECIES/MYCTO",
    "http://www.biokg.com/protein_SPECIES/MYCTU",
    "http://www.biokg.com/protein_SPECIES/PIG",
    "http://www.biokg.com/protein_SPECIES/PONAB",
    "http://www.biokg.com/protein_SPECIES/RAT",
    "http://www.biokg.com/protein_SPECIES/SCHPO",
    "http://www.biokg.com/protein_SPECIES/XENLA",
    "http://www.biokg.com/protein_SPECIES/YEAST"]
    protien_SPECIES_class_dict= dict(zip(protien_SPECIES_class_lst, [elem.split("/")[-1] for elem in protien_SPECIES_class_lst]))
    # protien_SPECIES_class_dict
    ###################################################################################
    protien_FAMILY_class_lst=["http://www.biokg.com/FAMILY/IPR000276",
    "http://www.biokg.com/FAMILY/IPR000725",
    "http://www.biokg.com/FAMILY/IPR001128",
    "http://www.biokg.com/FAMILY/IPR011701"]
    protien_FAMILY_class_dict= dict(zip(protien_FAMILY_class_lst, [elem.split("/")[-1] for elem in protien_FAMILY_class_lst]))
    # protien_FAMILY_class_dict
    #####################################################################################
    drugSuperClass_urls=" ".join(["<"+elem+">" for elem in drug_superclass_class_lst])
    drug_superclass_lables_query=f"""select ?s ?s as ?val
    from <http://www.biokg.com>
    {{
    ?s a <http://www.biokg.com/DRUG_SUPERCLASS>.
    values ?s {{{drugSuperClass_urls}}}.
    }}
    limit 100"""
    superclass_lables_df=executeSparqlQuery(drug_superclass_lables_query,SPARQLendpointUrl)
    superclass_lables_df['val']=superclass_lables_df['val'].apply(lambda x: x.split('/')[-1].replace('_',' ').strip().lower())
    drug_superclass_class_dict= dict(zip(superclass_lables_df['s'], superclass_lables_df['val']))
    # drug_superclass_class_dict
    ############################33
    drug_class_urls=" ".join(["<"+elem+">" for elem in drug_class_class_lst])
    drug_class_lables_query=f"""select ?s ?s as ?val
    from <http://www.biokg.com>
    {{
    ?s a <http://www.biokg.com/DRUG_CLASS>.
    values ?s {{{drug_class_urls}}}.
    }}
    limit 100"""
    class_lables_df=executeSparqlQuery(drug_class_lables_query,SPARQLendpointUrl)
    class_lables_df['val']=class_lables_df['val'].apply(lambda x: x.split('/')[-1].replace('_',' ').strip().lower())
    drug_class_class_dict= dict(zip(class_lables_df['s'], class_lables_df['val']))
    # drug_class_class_dict
    ###################
    drug_kingdom_urls=" ".join(["<"+elem+">" for elem in drug_kingdom_class_lst])
    drug_kingdom_lables_query=f"""select ?s ?s as ?val
    from <http://www.biokg.com>
    {{
    ?s a <http://www.biokg.com/DRUG_KINGDOM>.
    values ?s {{{drug_kingdom_urls}}}.
    }}
    limit 100"""
    kingdom_lables_df=executeSparqlQuery(drug_kingdom_lables_query,SPARQLendpointUrl)
    kingdom_lables_df['val']=kingdom_lables_df['val'].apply(lambda x: x.split('/')[-1].replace('_',' ').strip().lower())
    drug_kingdom_class_dict= dict(zip(kingdom_lables_df['s'], kingdom_lables_df['val']))
    # drug_kingdom_class_dict

    self.drug_dict_pred={"drug-superclass":{'predicate':"http://www.biokg.com/drug-property/SUPERCLASS","title":None},
                    "drug-class":{'predicate':"http://www.biokg.com/drug-property/CLASS","title":None},
                    "drug-kingdom":{'predicate':"http://www.biokg.com/drug-property/KINGDOM","title":None},
                    "protein-related_keyword":{'predicate':"http://www.biokg.com/protein-property/RELATED_KEYWORD","title":None},
                    "protein-SPECIES":{'predicate':"http://www.biokg.com/protein-property/SPECIES","title":None},
                    "protein-FAMILY":{'predicate':"http://www.biokg.com/FAMILY","title":None}}
    self.drug_dict_pred_class={"drug-superclass":{"classes":drug_superclass_class_dict,"mid":"24c3ff5f7399f2dff2bd2e6f26a216eec248323af695a6ecc67d01dab97158bf"},
                          "drug-class":{"classes":drug_class_class_dict,"mid":"b81ecc18c7c8700db2f92f5ead3096586c8a0d4c9cb9b68984dd8e9bf9b30d05"},
                          "drug-kingdom":{"classes":drug_kingdom_class_dict,"mid":"0b977c8df951ce4b1a117d5ddcc3e95be002d4435fb025cf7e7b0e18253bc8d9"},
                          "protein-related_keyword":{"classes":protien_keyword_class_dict,"mid":"b14fedb990ba1a57e5c8108c8c81b9eac65fa3a83af16770f2c3031a280f4c94"},
                          "protein-SPECIES":{"classes":protien_SPECIES_class_dict,"mid":"23cef3c023eb978de13f7f2b52524d799a6cdf4f0804ba22f65686b55bc7bda9"},
                          "protein-FAMILY":{"classes":protien_FAMILY_class_dict,"mid":"1d942651f976fe651780b8e6101379da933da0a5b9bd3e70023b5e0cbdf8ac1b"}}


    self.by_pubmid_BGP={"drug-class":"""?s ?p_by ?by.
        values ?p_by {<http://www.biokg.com/drug-property/PUBMED_ARTICLE>}
        values ?by {<http://www.biokg.com/DRUG_PUBMED_ARTICLE/17567513>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/17845503>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/25851629>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/25884661>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/32433465>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/30075127>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/32690640>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/29544147>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/18243430>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/32376603>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/30218687>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/32468087>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/28515226>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/33245898>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/32916310>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/34070997>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/33786632>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/33529751>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/20089935>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/20473325>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/23302720>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/33737119>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/31876334>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/29386193>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/27911230>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/27516093>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/33795638>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/29537116>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/36485127>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/35830173>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/36239905>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/26318916>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/18251718>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/19117887>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/17111207>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/17318068>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/18303190>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/31394818>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/33703984>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/34850359>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/30689738>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/35847448>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/24451000>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/23815106>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/21685861>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/17636630>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/25920571>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/21501034>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/19463072>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/28341939>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/29122372>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/17134907>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/18332080>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/30184207>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/30805897>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/17596167>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/18079742>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/29572459>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/31254295>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/18025536>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/33131243>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/24852768>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/27335049>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/27265781>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/17187586>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/17259948>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/31063770>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/18372395>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/17940749>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/17913896>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/24105299>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/20099988>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/18246523>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/20347527>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/23159111>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/36841226>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/22811343>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/21770474>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/22620717>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/22849428>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/28523596>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/22231104>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/21635236>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/24957842>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/21067468>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/24259625>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/25397996>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/25110138>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/23812940>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/18096568>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/27389324>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/26406774>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/26729184>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/19897799>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/22532463>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/24259556>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/21949059>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/27558232>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/21507963>
    <http://www.biokg.com/DRUG_PUBMED_ARTICLE/26183611>}""",
                   "protein-related_keyword":"""?s ?p_by ?by.
        values ?p_by {<http://www.biokg.com/protein-property/RELATED_PUBMED_ID>}
        values ?by {<http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:22139424>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:28626029>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:26541337>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:29285825>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:37342957>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:21129373>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:28362257>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:24097954>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:24026985>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:20568242>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:28402691>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:28028224>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:23871637>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:19651892>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:26232532>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:35312765>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:17624691>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:18064521>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:23073385>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:28254886>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:28263986>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:32973303>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:17173329>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:18789916>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:22522421>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:17976317>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:20193664>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:20445169>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:30659109>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:31298765>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:19449125>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:22343900>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:32554502>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:26801003>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:28911205>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:23587805>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:21952246>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:17498629>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:17584769>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:19043417>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:20732993>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:17143547>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:28981088>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:30833296>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:31413123>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:26436293>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:26479776>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:17699603>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:23661805>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:21765411>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:27390838>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:18322275>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:17141209>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:17478059>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:25881887>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:31422819>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:30250252>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:17336467>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:18684905>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:30427554>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:23806337>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:17914355>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:25875846>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:29518376>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:28928283>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:19531357>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:23908157>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:30080879>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:19656802>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:17555741>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:17681134>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:19705447>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:29955957>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:19572019>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:22484487>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:17556508>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:38287013>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:23791524>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:31474366>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:34918187>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:20417602>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:20859253>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:19245862>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:24803460>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:24910095>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:28894085>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:26538025>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:19830590>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:35708608>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:21349154>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:27189455>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:22581229>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:22128170>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:18068130>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:26663717>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:17717047>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:34365506>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:24119684>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:17623043>
    <http://www.biokg.com/protein_RELATED_PUBMED_ID/pubmed:20133602>}"""
    }
class yago4_GLOW_Bench:
  person_dict_pred,person_dict_pred_class,creativeWork_dict_pred,creativeWork_dict_pred_class={},{},{},{}
  by_performer_BGP =None
  def __init__(self,SPARQLendpointUrl):
    person_award_class_lst = ['http://yago-knowledge.org/resource/100_Women_(BBC)',
                              'http://yago-knowledge.org/resource/Order_of_Friendship',
                              'http://yago-knowledge.org/resource/Silbernes_Lorbeerblatt']
    person_award_class_dict = {elem: str(elem).split("/")[-1].replace(',', '_') for elem in person_award_class_lst}
    ####################################################################
    person_graduate_of_class_lst = ['http://yago-knowledge.org/resource/Columbia_University',
                                    'http://yago-knowledge.org/resource/Harvard_University',
                                    'http://yago-knowledge.org/resource/University_of_California,_Los_Angeles',
                                    'http://yago-knowledge.org/resource/University_of_Michigan']
    person_graduate_of_class_dict = {elem: str(elem).split("/")[-1].replace(',', '_') for elem in
                                     person_graduate_of_class_lst}
    ####################################################################
    person_language_class_lst = ['http://yago-knowledge.org/resource/Catalan_language',
                                 'http://yago-knowledge.org/resource/English_language',
                                 'http://yago-knowledge.org/resource/French_language',
                                 'http://yago-knowledge.org/resource/German_language',
                                 'http://yago-knowledge.org/resource/Japanese_language',
                                 'http://yago-knowledge.org/resource/Korean_language',
                                 'http://yago-knowledge.org/resource/Russian_language',
                                 'http://yago-knowledge.org/resource/Spanish_language']
    person_language_class_dict = {elem: str(elem).split("/")[-1].replace(',', '_') for elem in
                                  person_language_class_lst}
    #####################################################################
    person_occupation_class_lst = ['http://yago-knowledge.org/resource/Actor',
                                   'http://yago-knowledge.org/resource/Aircraft_pilot',
                                   'http://yago-knowledge.org/resource/Animator',
                                   'http://yago-knowledge.org/resource/Announcer',
                                   'http://yago-knowledge.org/resource/Anthropologist',
                                   'http://yago-knowledge.org/resource/Architect',
                                   'http://yago-knowledge.org/resource/Artist',
                                   'http://yago-knowledge.org/resource/Astronomer',
                                   'http://yago-knowledge.org/resource/Audio_engineer',
                                   'http://yago-knowledge.org/resource/Ballet_dancer',
                                   'http://yago-knowledge.org/resource/Biologist',
                                   'http://yago-knowledge.org/resource/Business_executive',
                                   'http://yago-knowledge.org/resource/Businessperson',
                                   'http://yago-knowledge.org/resource/Cartoonist',
                                   'http://yago-knowledge.org/resource/Chef',
                                   'http://yago-knowledge.org/resource/Chemist',
                                   'http://yago-knowledge.org/resource/Chess_player',
                                   'http://yago-knowledge.org/resource/Chief_executive_officer',
                                   'http://yago-knowledge.org/resource/Child_actor',
                                   'http://yago-knowledge.org/resource/Cinematographer',
                                   'http://yago-knowledge.org/resource/Coach_(basketball)',
                                   'http://yago-knowledge.org/resource/Coach_(sport)',
                                   'http://yago-knowledge.org/resource/Columnist',
                                   'http://yago-knowledge.org/resource/Comedian',
                                   'http://yago-knowledge.org/resource/Comics_artist',
                                   'http://yago-knowledge.org/resource/Computer_scientist',
                                   'http://yago-knowledge.org/resource/Conductor_(music)',
                                   'http://yago-knowledge.org/resource/Curator',
                                   'http://yago-knowledge.org/resource/Dancer',
                                   'http://yago-knowledge.org/resource/Designer',
                                   'http://yago-knowledge.org/resource/Diplomat',
                                   'http://yago-knowledge.org/resource/Directeur_sportif',
                                   'http://yago-knowledge.org/resource/Disc_jockey',
                                   'http://yago-knowledge.org/resource/Drummer',
                                   'http://yago-knowledge.org/resource/Economist',
                                   'http://yago-knowledge.org/resource/Environmentalist',
                                   'http://yago-knowledge.org/resource/Erotic_photography_model',
                                   'http://yago-knowledge.org/resource/Essayist',
                                   'http://yago-knowledge.org/resource/Executive_producer',
                                   'http://yago-knowledge.org/resource/Film_director',
                                   'http://yago-knowledge.org/resource/Film_producer',
                                   'http://yago-knowledge.org/resource/Football_player',
                                   'http://yago-knowledge.org/resource/Go_professional',
                                   'http://yago-knowledge.org/resource/Graphic_designer',
                                   'http://yago-knowledge.org/resource/Guitarist',
                                   'http://yago-knowledge.org/resource/Head_coach',
                                   'http://yago-knowledge.org/resource/Human_rights_defender',
                                   'http://yago-knowledge.org/resource/Illustrator',
                                   'http://yago-knowledge.org/resource/Jazz_guitarist',
                                   'http://yago-knowledge.org/resource/Jockey',
                                   'http://yago-knowledge.org/resource/Judge',
                                   'http://yago-knowledge.org/resource/Jurist',
                                   'http://yago-knowledge.org/resource/Lawyer',
                                   'http://yago-knowledge.org/resource/Lyricist',
                                   'http://yago-knowledge.org/resource/Manager_(association_football)',
                                   'http://yago-knowledge.org/resource/Mangaka',
                                   'http://yago-knowledge.org/resource/Mathematician',
                                   'http://yago-knowledge.org/resource/Motivational_speaker',
                                   'http://yago-knowledge.org/resource/News_presenter',
                                   'http://yago-knowledge.org/resource/Novelist',
                                   'http://yago-knowledge.org/resource/Officer_(armed_forces)',
                                   'http://yago-knowledge.org/resource/Opera_singer',
                                   'http://yago-knowledge.org/resource/Philosopher',
                                   'http://yago-knowledge.org/resource/Photographer',
                                   'http://yago-knowledge.org/resource/Physicist',
                                   'http://yago-knowledge.org/resource/Pianist',
                                   'http://yago-knowledge.org/resource/Playboy_Playmate',
                                   'http://yago-knowledge.org/resource/Playwright',
                                   'http://yago-knowledge.org/resource/Poet',
                                   'http://yago-knowledge.org/resource/Police_officer',
                                   'http://yago-knowledge.org/resource/Politician',
                                   'http://yago-knowledge.org/resource/Pornographic_film_actor',
                                   'http://yago-knowledge.org/resource/Professional_shogi_player',
                                   'http://yago-knowledge.org/resource/Professor',
                                   'http://yago-knowledge.org/resource/Psychologist',
                                   'http://yago-knowledge.org/resource/Radio_personality',
                                   'http://yago-knowledge.org/resource/Referee_(association_football)',
                                   'http://yago-knowledge.org/resource/Regisseur',
                                   'http://yago-knowledge.org/resource/Researcher',
                                   'http://yago-knowledge.org/resource/Restaurateur',
                                   'http://yago-knowledge.org/resource/Rikishi',
                                   'http://yago-knowledge.org/resource/Saxophonist',
                                   'http://yago-knowledge.org/resource/Screenwriter',
                                   'http://yago-knowledge.org/resource/Singer-songwriter',
                                   'http://yago-knowledge.org/resource/Skateboarder',
                                   'http://yago-knowledge.org/resource/Socialite',
                                   'http://yago-knowledge.org/resource/Songwriter',
                                   'http://yago-knowledge.org/resource/Sport_shooter',
                                   'http://yago-knowledge.org/resource/Sports_commentator',
                                   'http://yago-knowledge.org/resource/Stage_actor',
                                   'http://yago-knowledge.org/resource/Stunt_performer',
                                   'http://yago-knowledge.org/resource/Tarento',
                                   'http://yago-knowledge.org/resource/Teacher',
                                   'http://yago-knowledge.org/resource/Television_director',
                                   'http://yago-knowledge.org/resource/Television_presenter',
                                   'http://yago-knowledge.org/resource/Television_producer',
                                   'http://yago-knowledge.org/resource/Theatre_director',
                                   'http://yago-knowledge.org/resource/Trampolinist',
                                   'http://yago-knowledge.org/resource/Translator',
                                   'http://yago-knowledge.org/resource/Violinist',
                                   'http://yago-knowledge.org/resource/Voice_acting_in_Japan',
                                   'http://yago-knowledge.org/resource/Writer']
    person_occupation_class_dict = {elem: str(elem).split("/")[-1].replace(',', '_') for elem in
                                    person_occupation_class_lst}
    ####################################################################
    person_nationality_class_lst = ['http://yago-knowledge.org/resource/Albania',
                                    'http://yago-knowledge.org/resource/Algeria',
                                    'http://yago-knowledge.org/resource/Argentina',
                                    'http://yago-knowledge.org/resource/Armenia',
                                    'http://yago-knowledge.org/resource/Australia',
                                    'http://yago-knowledge.org/resource/Austria',
                                    'http://yago-knowledge.org/resource/Azerbaijan',
                                    'http://yago-knowledge.org/resource/Bangladesh',
                                    'http://yago-knowledge.org/resource/Belarus',
                                    'http://yago-knowledge.org/resource/Belgium',
                                    'http://yago-knowledge.org/resource/Bosnia_and_Herzegovina',
                                    'http://yago-knowledge.org/resource/Brazil',
                                    'http://yago-knowledge.org/resource/Bulgaria',
                                    'http://yago-knowledge.org/resource/Cameroon',
                                    'http://yago-knowledge.org/resource/Canada',
                                    'http://yago-knowledge.org/resource/Chile',
                                    'http://yago-knowledge.org/resource/China',
                                    'http://yago-knowledge.org/resource/Colombia',
                                    'http://yago-knowledge.org/resource/Croatia',
                                    'http://yago-knowledge.org/resource/Cuba',
                                    'http://yago-knowledge.org/resource/Czech_Republic',
                                    'http://yago-knowledge.org/resource/Denmark',
                                    'http://yago-knowledge.org/resource/Dominican_Republic',
                                    'http://yago-knowledge.org/resource/Ecuador',
                                    'http://yago-knowledge.org/resource/Egypt',
                                    'http://yago-knowledge.org/resource/Estonia',
                                    'http://yago-knowledge.org/resource/Finland',
                                    'http://yago-knowledge.org/resource/France',
                                    'http://yago-knowledge.org/resource/Georgia_(country)',
                                    'http://yago-knowledge.org/resource/Germany',
                                    'http://yago-knowledge.org/resource/Ghana',
                                    'http://yago-knowledge.org/resource/Greece',
                                    'http://yago-knowledge.org/resource/Hungary',
                                    'http://yago-knowledge.org/resource/Iceland',
                                    'http://yago-knowledge.org/resource/India',
                                    'http://yago-knowledge.org/resource/Indonesia',
                                    'http://yago-knowledge.org/resource/Iran',
                                    'http://yago-knowledge.org/resource/Israel',
                                    'http://yago-knowledge.org/resource/Italy',
                                    'http://yago-knowledge.org/resource/Ivory_Coast',
                                    'http://yago-knowledge.org/resource/Jamaica',
                                    'http://yago-knowledge.org/resource/Japan',
                                    'http://yago-knowledge.org/resource/Kazakhstan',
                                    'http://yago-knowledge.org/resource/Kenya',
                                    'http://yago-knowledge.org/resource/Kingdom_of_the_Netherlands',
                                    'http://yago-knowledge.org/resource/Latvia',
                                    'http://yago-knowledge.org/resource/Lithuania',
                                    'http://yago-knowledge.org/resource/Malaysia',
                                    'http://yago-knowledge.org/resource/Mexico',
                                    'http://yago-knowledge.org/resource/Moldova',
                                    'http://yago-knowledge.org/resource/Montenegro',
                                    'http://yago-knowledge.org/resource/Morocco',
                                    'http://yago-knowledge.org/resource/New_Zealand',
                                    'http://yago-knowledge.org/resource/Nigeria',
                                    'http://yago-knowledge.org/resource/North_Macedonia',
                                    'http://yago-knowledge.org/resource/Norway',
                                    'http://yago-knowledge.org/resource/Pakistan',
                                    'http://yago-knowledge.org/resource/Paraguay',
                                    'http://yago-knowledge.org/resource/Peru',
                                    'http://yago-knowledge.org/resource/Philippines',
                                    'http://yago-knowledge.org/resource/Poland',
                                    'http://yago-knowledge.org/resource/Portugal',
                                    'http://yago-knowledge.org/resource/Republic_of_Ireland',
                                    'http://yago-knowledge.org/resource/Romania',
                                    'http://yago-knowledge.org/resource/Russia',
                                    'http://yago-knowledge.org/resource/Saudi_Arabia',
                                    'http://yago-knowledge.org/resource/Senegal',
                                    'http://yago-knowledge.org/resource/Serbia',
                                    'http://yago-knowledge.org/resource/Singapore',
                                    'http://yago-knowledge.org/resource/Slovakia',
                                    'http://yago-knowledge.org/resource/Slovenia',
                                    'http://yago-knowledge.org/resource/South_Africa',
                                    'http://yago-knowledge.org/resource/South_Korea',
                                    'http://yago-knowledge.org/resource/Soviet_Union',
                                    'http://yago-knowledge.org/resource/Spain',
                                    'http://yago-knowledge.org/resource/Sri_Lanka',
                                    'http://yago-knowledge.org/resource/Sweden',
                                    'http://yago-knowledge.org/resource/Switzerland',
                                    'http://yago-knowledge.org/resource/Taiwan',
                                    'http://yago-knowledge.org/resource/Thailand',
                                    'http://yago-knowledge.org/resource/Tunisia',
                                    'http://yago-knowledge.org/resource/Turkey',
                                    'http://yago-knowledge.org/resource/Ukraine',
                                    'http://yago-knowledge.org/resource/United_Kingdom',
                                    'http://yago-knowledge.org/resource/United_States',
                                    'http://yago-knowledge.org/resource/Uruguay',
                                    'http://yago-knowledge.org/resource/Uzbekistan',
                                    'http://yago-knowledge.org/resource/Venezuela',
                                    'http://yago-knowledge.org/resource/Vietnam',
                                    'http://yago-knowledge.org/resource/Wales',
                                    'http://yago-knowledge.org/resource/Zimbabwe']
    person_nationality_class_dict = {elem: str(elem).split("/")[-1].replace(',', '_') for elem in
                                     person_nationality_class_lst}
    creativeWork_publisher_class_lst = ["http://yago-knowledge.org/resource/Activision",
                                        "http://yago-knowledge.org/resource/Alfred_A._Knopf",
                                        "http://yago-knowledge.org/resource/Del_Rey_Books",
                                        "http://yago-knowledge.org/resource/Doubleday_(publisher)",
                                        "http://yago-knowledge.org/resource/Electronic_Arts",
                                        "http://yago-knowledge.org/resource/HarperCollins",
                                        "http://yago-knowledge.org/resource/Konami",
                                        "http://yago-knowledge.org/resource/Nintendo",
                                        "http://yago-knowledge.org/resource/Random_House",
                                        "http://yago-knowledge.org/resource/Sega",
                                        "http://yago-knowledge.org/resource/Simon_&_Schuster",
                                        "http://yago-knowledge.org/resource/Sony_Interactive_Entertainment",
                                        "http://yago-knowledge.org/resource/Tor_Books",
                                        "http://yago-knowledge.org/resource/Ubisoft"]
    creativeWork_publisher_dict = {elem: str(elem).split("/")[-1].replace(',', '_') for elem in
                                   creativeWork_publisher_class_lst}
    ####################################################################
    creativeWork_productionCompany_class_lst = ["http://yago-knowledge.org/resource/20th_Century_Studios",
                                                "http://yago-knowledge.org/resource/Columbia_Pictures",
                                                "http://yago-knowledge.org/resource/Lenfilm",
                                                "http://yago-knowledge.org/resource/Metro-Goldwyn-Mayer",
                                                "http://yago-knowledge.org/resource/Mosfilm",
                                                "http://yago-knowledge.org/resource/National_Film_Board_of_Canada",
                                                "http://yago-knowledge.org/resource/Paramount_Pictures",
                                                "http://yago-knowledge.org/resource/Shaw_Brothers_Studio",
                                                "http://yago-knowledge.org/resource/The_Asylum",
                                                "http://yago-knowledge.org/resource/Troma_Entertainment",
                                                "http://yago-knowledge.org/resource/Universal_Pictures",
                                                "http://yago-knowledge.org/resource/Walt_Disney_Pictures",
                                                "http://yago-knowledge.org/resource/Warner_Bros"]
    creativeWork_productionCompany_dict = {elem: str(elem).split("/")[-1].replace(',', '_') for elem in
                                           creativeWork_productionCompany_class_lst}
    ####################################################################
    creativeWork_countryOfOrigin_class_lst = ['http://yago-knowledge.org/resource/Canada',
                                              'http://yago-knowledge.org/resource/France',
                                              'http://yago-knowledge.org/resource/India',
                                              'http://yago-knowledge.org/resource/Japan',
                                              'http://yago-knowledge.org/resource/United_Kingdom',
                                              'http://yago-knowledge.org/resource/United_States']
    creativeWork_countryOfOrigin_dict = {elem: str(elem).split("/")[-1].replace(',', '_') for elem in
                                         creativeWork_countryOfOrigin_class_lst}
    ####################################################################
    creativeWork_inlanguage_class_lst = ['http://yago-knowledge.org/resource/American_English',
                                         'http://yago-knowledge.org/resource/Arabic',
                                         'http://yago-knowledge.org/resource/Bengali_language',
                                         'http://yago-knowledge.org/resource/Chinese_language',
                                         'http://yago-knowledge.org/resource/Dutch_language',
                                         'http://yago-knowledge.org/resource/English_language',
                                         'http://yago-knowledge.org/resource/Finnish_language',
                                         'http://yago-knowledge.org/resource/French_language',
                                         'http://yago-knowledge.org/resource/German_language',
                                         'http://yago-knowledge.org/resource/Greek_language',
                                         'http://yago-knowledge.org/resource/Hebrew_language',
                                         'http://yago-knowledge.org/resource/Hindi',
                                         'http://yago-knowledge.org/resource/Italian_language',
                                         'http://yago-knowledge.org/resource/Japanese_language',
                                         'http://yago-knowledge.org/resource/Korean_language',
                                         'http://yago-knowledge.org/resource/Norwegian_language',
                                         'http://yago-knowledge.org/resource/Polish_language',
                                         'http://yago-knowledge.org/resource/Portuguese_language',
                                         'http://yago-knowledge.org/resource/Russian_language',
                                         'http://yago-knowledge.org/resource/Serbian_language',
                                         'http://yago-knowledge.org/resource/Slovene_language',
                                         'http://yago-knowledge.org/resource/Spanish_language',
                                         'http://yago-knowledge.org/resource/Swedish_language',
                                         'http://yago-knowledge.org/resource/Turkish_language',
                                         'http://yago-knowledge.org/resource/Urdu']
    creativeWork_inlanguage_dict = {elem: str(elem).split("/")[-1].replace(',', '_') for elem in
                                    creativeWork_inlanguage_class_lst}
    ####################################################################
    creativeWork_genre_class_lst = ['http://yago-knowledge.org/resource/Action_film',
                                    'http://yago-knowledge.org/resource/Alternative_rock',
                                    'http://yago-knowledge.org/resource/Country_music',
                                    'http://yago-knowledge.org/resource/Hard_rock',
                                    'http://yago-knowledge.org/resource/Hip_hop_music',
                                    'http://yago-knowledge.org/resource/Horror_film',
                                    'http://yago-knowledge.org/resource/Indie_rock',
                                    'http://yago-knowledge.org/resource/J-pop',
                                    'http://yago-knowledge.org/resource/Jazz',
                                    'http://yago-knowledge.org/resource/Pop_rock',
                                    'http://yago-knowledge.org/resource/Rock_music']
    creativeWork_genre_dict = {elem: str(elem).split("/")[-1].replace(',', '_') for elem in
                               creativeWork_genre_class_lst}
    self.person_dict_pred = {"person-country_of_nationality": "http://schema.org/nationality",
                        "parson-graduate_of_organization": "http://schema.org/alumniOf",
                        "person-occupation": "http://schema.org/hasOccupation",
                        "person-spoken_language": "http://schema.org/knowsLanguage",
                        "person-given_award": "http://schema.org/award"}
    self.person_dict_pred_class = {"person-country_of_nationality": {"classes": person_nationality_class_dict,
                                                                "mid": "2193a0adecf7ebcf3941f110c9769b6cd54d1ffb6b24c6fd250ed9d836661f3d"},
                              "parson-graduate_of_organization": {"classes": person_graduate_of_class_dict,
                                                                  "mid": "c7eebc561b8322d98175bc805540e61d4b5e417199e6a4758ba7887a0573e414"},
                              "person-occupation": {"classes": person_occupation_class_dict,
                                                    "mid": "5c4197e91f84a3507800432f35b6a34bcacdeff6104ffd32cff0c730be13ff79"},
                              "person-spoken_language": {"classes": person_language_class_dict,
                                                         "mid": "4ac83d65f0816966183f35e06c5c0309a577e6ea16a7e9dba78c64921ca92eea"},
                              "person-given_award": {"classes": person_award_class_dict,
                                                     "mid": "7f0b974ffc00fb421f926cd4b9c9dcbfd412a5e39b338f8f286c51525e13caa4"}}

    self.creativeWork_dict_pred = {"Creative_Work-Published_in_Language": "http://schema.org/inLanguage",
                              "Creative_Work-Production_Company": "http://schema.org/productionCompany",
                              "Creative_Work-country_Of_Origin": "http://schema.org/countryOfOrigin",
                              "Creative_Work-Genere": "http://schema.org/genre",
                              "Creative_Work-publisher": "http://schema.org/publisher"}
    self.creativeWork_dict_pred_class = {"Creative_Work-Published_in_Language": {"classes": creativeWork_inlanguage_dict,
                                                                            "mid": "956af320a731363001800c06ea96737602e8124b91c1ded80df699e687b45adc"},
                                    "Creative_Work-Production_Company": {"classes": creativeWork_productionCompany_dict,
                                                                         "mid": "28c32c902eaa2674bf749f6177ab4df8f4f48f74e49d70f98c1722df23648bed"},
                                    "Creative_Work-country_Of_Origin": {"classes": creativeWork_countryOfOrigin_dict,
                                                                        "mid": "030556a7a75e1d56c742417a416e9ea24bd4af630e02e2e7d0a5aab4603868bc"},
                                    "Creative_Work-Genere": {"classes": creativeWork_genre_dict,
                                                             "mid": "9bebb4ae8e3656e37157cd7174dbc1036b595447f3f298d202a7c882390a0634"},
                                    "Creative_Work-publisher": {"classes": creativeWork_publisher_dict,
                                                                "mid": "29891b0fb74c8b150364b0de110733b89e443335299466d5b3df0572821ea346"}}
    self.by_performer_BGP = {"Creative_Work-Genere": """?s ?p_by ?by.
        values ?p_by {<http://schema.org/byArtist> <http://schema.org/musicBy>}
        values ?by {<http://yago-knowledge.org/resource/R._D._Burman>
    <http://yago-knowledge.org/resource/Richard_Gibbs>
    <http://yago-knowledge.org/resource/Steve_Wariner>
    <http://yago-knowledge.org/resource/Jim_Goodwin>
    <http://yago-knowledge.org/resource/The_Gandharvas>
    <http://yago-knowledge.org/resource/Goya_Dress>
    <http://yago-knowledge.org/resource/Talk_Show_(band)>
    <http://yago-knowledge.org/resource/Jez_Spencer>
    <http://yago-knowledge.org/resource/Skid_Row_(American_band)>
    <http://yago-knowledge.org/resource/Psycho_Motel>
    <http://yago-knowledge.org/resource/DFC_(group)>
    <http://yago-knowledge.org/resource/The_Van_Pelt>
    <http://yago-knowledge.org/resource/Christie_Front_Drive>
    <http://yago-knowledge.org/resource/Black_Swan_Network>
    <http://yago-knowledge.org/resource/Splean>
    <http://yago-knowledge.org/resource/Katrina_and_the_Waves>
    <http://yago-knowledge.org/resource/Luscious_Jackson>
    <http://yago-knowledge.org/resource/Oblivion_Dust>
    <http://yago-knowledge.org/resource/Tokyo_Blade>
    <http://yago-knowledge.org/resource/Skyhooks_(band)>
    <http://yago-knowledge.org/resource/M._Ashraf>
    <http://yago-knowledge.org/resource/Electronic_(band)>
    <http://yago-knowledge.org/resource/Monster_Magnet>
    <http://yago-knowledge.org/resource/Criminal_Nation>
    <http://yago-knowledge.org/resource/Cella_Dwellas>
    <http://yago-knowledge.org/resource/Vince_Clarke>
    <http://yago-knowledge.org/resource/The_Hang_Ups>
    <http://yago-knowledge.org/resource/The_Mooney_Suzuki>
    <http://yago-knowledge.org/resource/MDO_(band)>
    <http://yago-knowledge.org/resource/DC_Talk>
    <http://yago-knowledge.org/resource/Pink_Floyd>
    <http://yago-knowledge.org/resource/Comes_with_the_Fall>
    <http://yago-knowledge.org/resource/Major_Figgas>
    <http://yago-knowledge.org/resource/Midfield_General>
    <http://yago-knowledge.org/resource/Peter_Dasent>
    <http://yago-knowledge.org/resource/Skunkhour>
    <http://yago-knowledge.org/resource/Unwed_Sailor>
    <http://yago-knowledge.org/resource/Gang_Gajang>
    <http://yago-knowledge.org/resource/People_in_Planes>
    <http://yago-knowledge.org/resource/Arrogance_(band)>}""",
                        "Creative_Work-Production_Company": """?s ?p_by ?by.
        values ?p_by {<http://schema.org/byArtist> <http://schema.org/musicBy>}
        values ?by {<http://yago-knowledge.org/resource/J._A._C._Redford>
    <http://yago-knowledge.org/resource/Barry_Mann>
    <http://yago-knowledge.org/resource/Matthew_Wilder>
    <http://yago-knowledge.org/resource/Igor_Stravinsky>
    <http://yago-knowledge.org/resource/Joe_Kraemer_(composer)>
    <http://yago-knowledge.org/resource/Richard_Harvey>
    <http://yago-knowledge.org/resource/Pinar_Toprak>
    <http://yago-knowledge.org/resource/Amotz_Plessner>
    <http://yago-knowledge.org/resource/Leoncio_Lara>
    <http://yago-knowledge.org/resource/Mark_Snow>
    <http://yago-knowledge.org/resource/Nathan_Wang>
    <http://yago-knowledge.org/resource/Transcenders>
    <http://yago-knowledge.org/resource/Steve_Jablonsky>
    <http://yago-knowledge.org/resource/Shalabi_Effect>
    <http://yago-knowledge.org/resource/Adrian_Belew>
    <http://yago-knowledge.org/resource/David_Kitay>
    <http://yago-knowledge.org/resource/Stewart_Copeland>
    <http://yago-knowledge.org/resource/Michel_Legrand>
    <http://yago-knowledge.org/resource/Manu_Chao>
    <http://yago-knowledge.org/resource/Jeff_Rona>
    <http://yago-knowledge.org/resource/Joe_Delia>
    <http://yago-knowledge.org/resource/Adam_Lewis>
    <http://yago-knowledge.org/resource/Brent_Belke>
    <http://yago-knowledge.org/resource/Alexandre_Desplat>
    <http://yago-knowledge.org/resource/Hans_Zimmer>
    <http://yago-knowledge.org/resource/Paul_Dukas>
    <http://yago-knowledge.org/resource/Kenny_Craddock>
    <http://yago-knowledge.org/resource/Eugen_Doga>
    <http://yago-knowledge.org/resource/Tupac_Shakur>
    <http://yago-knowledge.org/resource/Mervyn_Warren>
    <http://yago-knowledge.org/resource/Alexander_Knaifel>
    <http://yago-knowledge.org/resource/Chris_Walden>
    <http://yago-knowledge.org/resource/John_Van_Tongeren>
    <http://yago-knowledge.org/resource/Tim_Wynn>
    <http://yago-knowledge.org/resource/Ray_Evans>
    <http://yago-knowledge.org/resource/Alec_Puro>
    <http://yago-knowledge.org/resource/John_Lurie>
    <http://yago-knowledge.org/resource/Vladimir_Dashkevich>
    <http://yago-knowledge.org/resource/Cynthia_Weil>
    <http://yago-knowledge.org/resource/Danny_Elfman>
    <http://yago-knowledge.org/resource/Jay_Gruska>
    <http://yago-knowledge.org/resource/John_Debney>
    <http://yago-knowledge.org/resource/Mark_Isham>
    <http://yago-knowledge.org/resource/Edward_Elgar>
    <http://yago-knowledge.org/resource/Oliver_Jones_(pianist)>
    <http://yago-knowledge.org/resource/Michael_Wandmacher>
    <http://yago-knowledge.org/resource/Jim_Lang_(composer)>
    <http://yago-knowledge.org/resource/John_Massari>
    <http://yago-knowledge.org/resource/Leigh_Gorman>
    <http://yago-knowledge.org/resource/Rachid_Taha>
    <http://yago-knowledge.org/resource/Michael_Oesterle>
    <http://yago-knowledge.org/resource/Philip_Glass>
    <http://yago-knowledge.org/resource/Child_(band)>
    <http://yago-knowledge.org/resource/Jon_Fratelli>
    <http://yago-knowledge.org/resource/Wendy_Melvoin>
    <http://yago-knowledge.org/resource/Anthony_Marinelli>
    <http://yago-knowledge.org/resource/Graeme_Revell>
    <http://yago-knowledge.org/resource/Rubén_Blades>
    <http://yago-knowledge.org/resource/André_Abujamra>
    <http://yago-knowledge.org/resource/Ryan_Adams>
    <http://yago-knowledge.org/resource/Theodore_Shapiro>
    <http://yago-knowledge.org/resource/James_Dooley_(composer)>
    <http://yago-knowledge.org/resource/Ed_Harcourt>
    <http://yago-knowledge.org/resource/Fernando_Velázquez_(composer)>
    <http://yago-knowledge.org/resource/Igor_Kornelyuk>
    <http://yago-knowledge.org/resource/Ne-Yo>
    <http://yago-knowledge.org/resource/Randy_Newman>
    <http://yago-knowledge.org/resource/Robert_Lopez>
    <http://yago-knowledge.org/resource/John_Paesano>
    <http://yago-knowledge.org/resource/Elmer_Bernstein>
    <http://yago-knowledge.org/resource/Evan_Lurie>
    <http://yago-knowledge.org/resource/Roy_Hay_(musician)>
    <http://yago-knowledge.org/resource/Randy_Edelman>
    <http://yago-knowledge.org/resource/John_McCarthy_(composer)>
    <http://yago-knowledge.org/resource/Don_Davis_(composer)>}""",
                        "Creative_Work-publisher": """?s ?p_by ?by.
        values ?p_by {<http://schema.org/isBasedOn>}
        values ?by {<http://yago-knowledge.org/resource/Teenage_Mutant_Ninja_Turtles_(Mirage_Studios)>
    <http://yago-knowledge.org/resource/The_World_Is_Not_Enough>
    <http://yago-knowledge.org/resource/Star_Wars:_Episode_II_–_Attack_of_the_Clones>
    <http://yago-knowledge.org/resource/The_Lord_of_the_Rings:_The_Fellowship_of_the_Ring>
    <http://yago-knowledge.org/resource/Star_Trek:_Voyager>
    <http://yago-knowledge.org/resource/Catwoman_(film)>
    <http://yago-knowledge.org/resource/MediEvil>
    <http://yago-knowledge.org/resource/Transformers:_Revenge_of_the_Fallen>
    <http://yago-knowledge.org/resource/Avatar_(2009_film)>
    <http://yago-knowledge.org/resource/Pokémon_Black_and_White>
    <http://yago-knowledge.org/resource/Deadpool>
    <http://yago-knowledge.org/resource/Star_Wars>
    <http://yago-knowledge.org/resource/The_Tempest>
    <http://yago-knowledge.org/resource/The_Legend_of_Zelda:_Link's_Awakening>
    <http://yago-knowledge.org/resource/Buzz_Lightyear_of_Star_Command>
    <http://yago-knowledge.org/resource/Minority_Report_(film)>
    <http://yago-knowledge.org/resource/CSI:_Miami>
    <http://yago-knowledge.org/resource/Star_Trek_(film)>
    <http://yago-knowledge.org/resource/Where's_Wally%3F>
    <http://yago-knowledge.org/resource/Cloudy_with_a_Chance_of_Meatballs_(film)>
    <http://yago-knowledge.org/resource/Men_in_Black_3>
    <http://yago-knowledge.org/resource/Harry_Potter_and_the_Chamber_of_Secrets_(film)>
    <http://yago-knowledge.org/resource/Harry_Potter_and_the_Order_of_the_Phoenix_(film)>
    <http://yago-knowledge.org/resource/Harry_Potter_and_the_Deathly_Hallows_–_Part_2>
    <http://yago-knowledge.org/resource/Warhammer_40,000>
    <http://yago-knowledge.org/resource/The_Godfather_(novel)>
    <http://yago-knowledge.org/resource/Harry_Potter_and_the_Half-Blood_Prince>
    <http://yago-knowledge.org/resource/Iron_Man_2>
    <http://yago-knowledge.org/resource/The_Adventures_of_Tintin_(film)>
    <http://yago-knowledge.org/resource/The_Legend_of_Zelda:_Majora's_Mask>
    <http://yago-knowledge.org/resource/The_Empire_Strikes_Back>
    <http://yago-knowledge.org/resource/GoldenEye>
    <http://yago-knowledge.org/resource/Star_Wars:_Episode_I_–_The_Phantom_Menace>
    <http://yago-knowledge.org/resource/The_Road_to_El_Dorado>
    <http://yago-knowledge.org/resource/Charlie's_Angels:_Full_Throttle>
    <http://yago-knowledge.org/resource/Metal_Gear_Solid_(1998_video_game)>
    <http://yago-knowledge.org/resource/Harry_Potter_and_the_Goblet_of_Fire_(film)>
    <http://yago-knowledge.org/resource/From_Russia_with_Love_(film)>
    <http://yago-knowledge.org/resource/CSI:_NY>
    <http://yago-knowledge.org/resource/The_Godfather_Part_II>
    <http://yago-knowledge.org/resource/The_Legend_of_Korra>
    <http://yago-knowledge.org/resource/Theme_Hospital>
    <http://yago-knowledge.org/resource/Batman_Returns>
    <http://yago-knowledge.org/resource/Hook_(film)>
    <http://yago-knowledge.org/resource/Animaniacs>
    <http://yago-knowledge.org/resource/Super_Mario_Bros._3>
    <http://yago-knowledge.org/resource/Shrek_2>
    <http://yago-knowledge.org/resource/Star_Wars:_Episode_III_–_Revenge_of_the_Sith>
    <http://yago-knowledge.org/resource/24_(TV_series)>
    <http://yago-knowledge.org/resource/Bee_Movie>
    <http://yago-knowledge.org/resource/Monsters_vs._Aliens>
    <http://yago-knowledge.org/resource/Punisher>
    <http://yago-knowledge.org/resource/Rango_(2011_film)>
    <http://yago-knowledge.org/resource/Kid_Dracula_(1990_video_game)>
    <http://yago-knowledge.org/resource/Harry_Potter_and_the_Philosopher's_Stone_(film)>
    <http://yago-knowledge.org/resource/Metroid_(video_game)>
    <http://yago-knowledge.org/resource/Super_Mario>
    <http://yago-knowledge.org/resource/The_Godfather>
    <http://yago-knowledge.org/resource/The_Legend_of_Zelda:_Twilight_Princess>
    <http://yago-knowledge.org/resource/Age_of_Chivalry>
    <http://yago-knowledge.org/resource/Donkey_Kong_(video_game)>
    <http://yago-knowledge.org/resource/Shadowrun>
    <http://yago-knowledge.org/resource/Aladdin_(1992_Disney_film)>
    <http://yago-knowledge.org/resource/Batman_Beyond:_Return_of_the_Joker>
    <http://yago-knowledge.org/resource/Charlie's_Angels_(2000_film)>
    <http://yago-knowledge.org/resource/Harry_Potter_and_the_Prisoner_of_Azkaban_(film)>
    <http://yago-knowledge.org/resource/Gravity_Falls>
    <http://yago-knowledge.org/resource/XIII_(comics)>
    <http://yago-knowledge.org/resource/Pokémon_Gold_and_Silver>
    <http://yago-knowledge.org/resource/The_Legend_of_Zelda:_Ocarina_of_Time>
    <http://yago-knowledge.org/resource/The_Legend_of_Zelda:_The_Wind_Waker>
    <http://yago-knowledge.org/resource/Pinocchio_(1940_film)>
    <http://yago-knowledge.org/resource/The_Lord_of_the_Rings:_The_Return_of_the_King>
    <http://yago-knowledge.org/resource/Batman_Begins>
    <http://yago-knowledge.org/resource/Spider-Man_3>
    <http://yago-knowledge.org/resource/Book_of_Negroes>
    <http://yago-knowledge.org/resource/Harry_Potter_and_the_Deathly_Hallows_–_Part_1>
    <http://yago-knowledge.org/resource/MapleStory>
    <http://yago-knowledge.org/resource/Iron_Man>
    <http://yago-knowledge.org/resource/Syndicate_(1993_video_game)>
    <http://yago-knowledge.org/resource/Tomorrow_Never_Dies>
    <http://yago-knowledge.org/resource/Batman_Beyond>
    <http://yago-knowledge.org/resource/Marvel_Nemesis:_Rise_of_the_Imperfects_(comics)>
    <http://yago-knowledge.org/resource/X-Men_Origins:_Wolverine>
    <http://yago-knowledge.org/resource/Rogue_One>}"""}
class crunchbase_GLOW_Bench:
  person_dict_pred=None
  person_dict_pred_class=None
  def __init__(self,SPARQLendpointUrl):
      ###################################Person Title################################################
      person_title_class_lst = ["CEO", "CTO", "Co-Founder", "Co-founder", "Founder", "Founder & CEO"]
      person_title_class_dict = dict(
          zip(person_title_class_lst, [elem.split("/")[-1] for elem in person_title_class_lst]))
      person_title_class_dict
      ###################################Person organization_name################################################
      person_organization_name_class_lst = ["Customer360", "Facebook", "Google", "Harri", "Microsoft",
                                            "Procter & Gamble", "Shuraa", "TechCrunch", "Trucker Path", "Twitter",
                                            "Yahoo!"]
      person_organization_name_class_dict = dict(
          zip(person_organization_name_class_lst, [elem.split("/")[-1] for elem in person_organization_name_class_lst]))
      person_organization_name_class_dict
      ###################################Person organization_name################################################
      person_region_name_class_lst = ["California", "Delhi", "England", "Florida", "Ile-de-France", "Illinois",
                                      "Istanbul", "Karnataka", "Maharashtra", "Massachusetts", "New York", "Ontario",
                                      "Texas", "Washington"]
      person_region_name_class_dict = dict(
          zip(person_region_name_class_lst, [elem.split("/")[-1] for elem in person_region_name_class_lst]))
      person_region_name_class_dict
      ###################################Person organization_name################################################
      person_country_code_class_lst = ["AUS", "BRA", "CAN", "DEU", "ESP", "FRA", "GBR", "IND", "ISR", "ITA", "TUR",
                                       "USA"]
      person_country_code_class_dict = dict(
          zip(person_country_code_class_lst, [elem.split("/")[-1] for elem in person_country_code_class_lst]))
      person_country_code_class_dict

      self.person_dict_pred = {
          "person-title": {'predicate': "http://ontologycentral.com/2010/05/cb/vocab#title", "title": None},
          "person-organization_name": {'predicate': "http://ontologycentral.com/2010/05/cb/vocab#organization_name",
                                       "title": None},
          "person-region_name": {'predicate': "http://ontologycentral.com/2010/05/cb/vocab#region_name", "title": None},
          "person-country_code": {'predicate': "http://ontologycentral.com/2010/05/cb/vocab#country_code",
                                  "title": None}}
      self.person_dict_pred_class = {"person-title": {"classes": person_title_class_dict,
                                                 "mid": "ef6a412be66ee62260751e4df47fb3bed9d6aec8a059376d4bd0c3c0b6231109"},
                                "person-organization_name": {"classes": person_organization_name_class_dict,
                                                             "mid": "bfd3f5ac629133174cbded2f243e7ad1940ae0ca72e1390c5a160442aa59b295"},
                                "person-region_name": {"classes": person_region_name_class_dict,
                                                       "mid": "6a8f30a762dddd0535c799310651ffe8d8ae5770921d7a67ce4d3373953bbf67"},
                                "person-country_code": {"classes": person_country_code_class_dict,
                                                        "mid": "c3107eca0eb53b29bdb0e862e03efa079e2be1869844528c99423e2a78dd3bde"}}
class linkedIMDB_GLOW_Bench:
  film_dict_pred=None
  film_dict_pred_class=None
  def __init__(self,SPARQLendpointUrl):
      film_genre_class_lst = ['http://data.linkedmdb.org/resource/film_genre/14',
                              'http://data.linkedmdb.org/resource/film_genre/23',
                              'http://data.linkedmdb.org/resource/film_genre/27',
                              'http://data.linkedmdb.org/resource/film_genre/31',
                              'http://data.linkedmdb.org/resource/film_genre/4',
                              'http://data.linkedmdb.org/resource/film_genre/47',
                              'http://data.linkedmdb.org/resource/film_genre/9']
      genre_urls = " ".join(["<" + elem + ">" for elem in film_genre_class_lst])
      filem_genre_lables_query = f"""select ?s ?val
      from <https://linkedmdb.org>
      {{
      ?s <http://data.linkedmdb.org/resource/movie/film_genre_name> ?val.
      values ?s {{{genre_urls}}}.
      }}
      limit 100"""
      genre_lables_df = executeSparqlQuery(filem_genre_lables_query, SPARQLendpointUrl)
      film_genre_class_dict = my_dict = dict(zip(genre_lables_df['s'], genre_lables_df['val']))
      film_genre_class_dict
      #################################Country###################################
      film_country_class_lst = ['http://data.linkedmdb.org/resource/country/AR',
                                'http://data.linkedmdb.org/resource/country/AU',
                                'http://data.linkedmdb.org/resource/country/BR',
                                'http://data.linkedmdb.org/resource/country/CA',
                                'http://data.linkedmdb.org/resource/country/CN',
                                'http://data.linkedmdb.org/resource/country/DE',
                                'http://data.linkedmdb.org/resource/country/DK',
                                'http://data.linkedmdb.org/resource/country/ES',
                                'http://data.linkedmdb.org/resource/country/FI',
                                'http://data.linkedmdb.org/resource/country/FR',
                                'http://data.linkedmdb.org/resource/country/GB',
                                'http://data.linkedmdb.org/resource/country/HK',
                                'http://data.linkedmdb.org/resource/country/IE',
                                'http://data.linkedmdb.org/resource/country/IL',
                                'http://data.linkedmdb.org/resource/country/IN',
                                'http://data.linkedmdb.org/resource/country/IR',
                                'http://data.linkedmdb.org/resource/country/IT',
                                'http://data.linkedmdb.org/resource/country/JP',
                                'http://data.linkedmdb.org/resource/country/KR',
                                'http://data.linkedmdb.org/resource/country/MX',
                                'http://data.linkedmdb.org/resource/country/NL',
                                'http://data.linkedmdb.org/resource/country/NZ',
                                'http://data.linkedmdb.org/resource/country/PH',
                                'http://data.linkedmdb.org/resource/country/PK',
                                'http://data.linkedmdb.org/resource/country/PL',
                                'http://data.linkedmdb.org/resource/country/SE',
                                'http://data.linkedmdb.org/resource/country/TH']
      countries_urls = " ".join(["<" + elem + ">" for elem in film_country_class_lst])
      film_country_lables_query = f"""select ?s ?val
      from <https://linkedmdb.org>
      {{
      ?s <http://data.linkedmdb.org/resource/movie/country_name> ?val.
      values ?s {{{countries_urls}}}.
      }}
      limit 100"""
      countries_lables_df = executeSparqlQuery(film_country_lables_query, SPARQLendpointUrl)
      film_country_class_dict = my_dict = dict(zip(countries_lables_df['s'], countries_lables_df['val']))
      film_country_class_dict
      #################################Producer###################################
      film_producer_class_lst = ['http://data.linkedmdb.org/resource/producer/1',
                                 'http://data.linkedmdb.org/resource/producer/10050',
                                 'http://data.linkedmdb.org/resource/producer/10087',
                                 'http://data.linkedmdb.org/resource/producer/10094',
                                 'http://data.linkedmdb.org/resource/producer/10097',
                                 'http://data.linkedmdb.org/resource/producer/10246',
                                 'http://data.linkedmdb.org/resource/producer/10347',
                                 'http://data.linkedmdb.org/resource/producer/10391',
                                 'http://data.linkedmdb.org/resource/producer/106',
                                 'http://data.linkedmdb.org/resource/producer/10678',
                                 'http://data.linkedmdb.org/resource/producer/10762',
                                 'http://data.linkedmdb.org/resource/producer/10859',
                                 'http://data.linkedmdb.org/resource/producer/10879',
                                 'http://data.linkedmdb.org/resource/producer/11041',
                                 'http://data.linkedmdb.org/resource/producer/11092',
                                 'http://data.linkedmdb.org/resource/producer/11245',
                                 'http://data.linkedmdb.org/resource/producer/11267',
                                 'http://data.linkedmdb.org/resource/producer/11359',
                                 'http://data.linkedmdb.org/resource/producer/11380',
                                 'http://data.linkedmdb.org/resource/producer/11532',
                                 'http://data.linkedmdb.org/resource/producer/117',
                                 'http://data.linkedmdb.org/resource/producer/11830',
                                 'http://data.linkedmdb.org/resource/producer/11985',
                                 'http://data.linkedmdb.org/resource/producer/12205',
                                 'http://data.linkedmdb.org/resource/producer/12749',
                                 'http://data.linkedmdb.org/resource/producer/12864',
                                 'http://data.linkedmdb.org/resource/producer/13415',
                                 'http://data.linkedmdb.org/resource/producer/13683',
                                 'http://data.linkedmdb.org/resource/producer/13943',
                                 'http://data.linkedmdb.org/resource/producer/14763',
                                 'http://data.linkedmdb.org/resource/producer/248',
                                 'http://data.linkedmdb.org/resource/producer/6248',
                                 'http://data.linkedmdb.org/resource/producer/9724',
                                 'http://data.linkedmdb.org/resource/producer/9752',
                                 'http://data.linkedmdb.org/resource/producer/9778',
                                 'http://data.linkedmdb.org/resource/producer/9855',
                                 'http://data.linkedmdb.org/resource/producer/9869',
                                 'http://data.linkedmdb.org/resource/producer/9920',
                                 'http://data.linkedmdb.org/resource/producer/9922']
      producers_urls = " ".join(["<" + elem + ">" for elem in film_producer_class_lst])
      film_producer_lables_query = f"""select ?s ?val
      from <https://linkedmdb.org>
      {{
      ?s <http://data.linkedmdb.org/resource/movie/producer_name> ?val.
      values ?s {{{producers_urls}}}.
      }}
      limit 100"""
      producer_lables_df = executeSparqlQuery(film_producer_lables_query, SPARQLendpointUrl)
      film_producer_class_dict = my_dict = dict(zip(producer_lables_df['s'], producer_lables_df['val']))
      film_producer_class_dict
      #################################Subject###################################
      film_subject_class_lst = ["http://data.linkedmdb.org/resource/film_subject/1019",
                                "http://data.linkedmdb.org/resource/film_subject/1215",
                                "http://data.linkedmdb.org/resource/film_subject/1247",
                                "http://data.linkedmdb.org/resource/film_subject/198",
                                "http://data.linkedmdb.org/resource/film_subject/201",
                                "http://data.linkedmdb.org/resource/film_subject/228",
                                "http://data.linkedmdb.org/resource/film_subject/229",
                                "http://data.linkedmdb.org/resource/film_subject/230",
                                "http://data.linkedmdb.org/resource/film_subject/232",
                                "http://data.linkedmdb.org/resource/film_subject/248",
                                "http://data.linkedmdb.org/resource/film_subject/274",
                                "http://data.linkedmdb.org/resource/film_subject/281",
                                "http://data.linkedmdb.org/resource/film_subject/298",
                                "http://data.linkedmdb.org/resource/film_subject/329",
                                "http://data.linkedmdb.org/resource/film_subject/333",
                                "http://data.linkedmdb.org/resource/film_subject/352",
                                "http://data.linkedmdb.org/resource/film_subject/438",
                                "http://data.linkedmdb.org/resource/film_subject/444",
                                "http://data.linkedmdb.org/resource/film_subject/460",
                                "http://data.linkedmdb.org/resource/film_subject/465",
                                "http://data.linkedmdb.org/resource/film_subject/506",
                                "http://data.linkedmdb.org/resource/film_subject/514",
                                "http://data.linkedmdb.org/resource/film_subject/523",
                                "http://data.linkedmdb.org/resource/film_subject/548",
                                "http://data.linkedmdb.org/resource/film_subject/556",
                                "http://data.linkedmdb.org/resource/film_subject/581",
                                "http://data.linkedmdb.org/resource/film_subject/849",
                                "http://data.linkedmdb.org/resource/film_subject/883"]
      subjects_urls = " ".join(["<" + elem + ">" for elem in film_subject_class_lst])
      film_subject_lables_query = f"""select ?s ?val
      from <https://linkedmdb.org>
      {{
      ?s <http://data.linkedmdb.org/resource/movie/film_subject_name> ?val.
      values ?s {{{subjects_urls}}}.
      }}
      limit 100"""
      subjects_lables_df = executeSparqlQuery(film_subject_lables_query, SPARQLendpointUrl)
      film_subject_class_dict = my_dict = dict(zip(subjects_lables_df['s'], subjects_lables_df['val']))
      #################################Language###################################
      film_language_class_dict = {"http://www.lingvoj.org/lingvo/ar": "Arabic",
                                  "http://www.lingvoj.org/lingvo/de": "German",
                                  "http://www.lingvoj.org/lingvo/en": "English",
                                  "http://www.lingvoj.org/lingvo/es": "Spanish",
                                  "http://www.lingvoj.org/lingvo/fr": "French",
                                  "http://www.lingvoj.org/lingvo/hi": "Hindi",
                                  "http://www.lingvoj.org/lingvo/it": "Italian",
                                  "http://www.lingvoj.org/lingvo/ja": "Japanese",
                                  "http://www.lingvoj.org/lingvo/ko": "Korean",
                                  "http://www.lingvoj.org/lingvo/ml": "Malayalam",
                                  "http://www.lingvoj.org/lingvo/ru": "Russian",
                                  "http://www.lingvoj.org/lingvo/sv": "Swedish",
                                  "http://www.lingvoj.org/lingvo/ta": "Tamil",
                                  "http://www.lingvoj.org/lingvo/ur": "Urdu"}
      self.film_dict_pred = {"film-genre": {'predicate': "http://data.linkedmdb.org/resource/movie/genre",
                                       "title": "http://data.linkedmdb.org/resource/movie/film_genre_name"},
                        "film-country": {'predicate': "http://data.linkedmdb.org/resource/movie/country",
                                         "title": "http://data.linkedmdb.org/resource/movie/country_name"},
                        "film-producer": {'predicate': "http://data.linkedmdb.org/resource/movie/producer",
                                          "title": "http://data.linkedmdb.org/resource/movie/producer_name"},
                        "film-subject": {'predicate': "http://data.linkedmdb.org/resource/movie/film_subject",
                                         "title": "http://data.linkedmdb.org/resource/movie/film_subject_name"},
                        "film-language": {'predicate': "http://data.linkedmdb.org/resource/movie/language",
                                          "title": None}}

      self.film_dict_pred_class = {"film-genre": {"classes": film_genre_class_dict,
                                             "mid": "4a6626ebd5c4357248d3cf3ce206a06cb92b56d1beed90d675e970f40faf4223"},
                              "film-country": {"classes": film_country_class_dict,
                                               "mid": "d741f905fdebbbb725aa1fb7c3dca41dbfbc128e2f9e1e62834c18f3d7c2c344"},
                              "film-producer": {"classes": film_producer_class_dict,
                                                "mid": "94b3cdd19b43a5518e61d7d27c1623c9991a465215d43e70f6ccecc3eb3ac279"},
                              "film-subject": {"classes": film_subject_class_dict,
                                               "mid": "8d1c2b3b2e782bba236b09477568760b09a554b9b40bd2e0926889e3df302202"},
                              "film-language": {"classes": film_language_class_dict,
                                                "mid": "1a3b735a1054016619d499aa2b9898bac5440c26f241cbd4009cf99304c9c292"}}
class AskGNN_Glow_Bench:
    arxiv2023_dict_pred,arxiv2023_dict_pred_class=None,None
    ogbnrxiv_dict_pred,ogbnrxiv_dict_pred_class=None,None
    def __init__(self,SPARQLendpointUrl):
        arxiv2023_paper_class_lst = ["cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "cs.CY",
                                     "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GR", "cs.GT",
                                     "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS", "cs.NA",
                                     "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD",
                                     "cs.SE", "cs.SI", "cs.SY"]
        arxiv2023_paper_class_dict = dict(zip(arxiv2023_paper_class_lst, arxiv2023_paper_class_lst))
        ogbnArxiv_paper_class_lst = ["arxiv cs cr", "arxiv cs ni", "arxiv cs pl", "arxiv cs it", "arxiv cs ro",
                                     "arxiv cs lg", "arxiv cs ir", "arxiv cs dm", "arxiv cs gt", "arxiv cs ds",
                                     "arxiv cs cv", "arxiv cs ne", "arxiv cs fl", "arxiv cs ai", "arxiv cs si",
                                     "arxiv cs sy", "arxiv cs lo", "arxiv cs cy", "arxiv cs cc", "arxiv cs cl",
                                     "arxiv cs se", "arxiv cs hc", "arxiv cs dc", "arxiv cs db", "arxiv cs gr",
                                     "arxiv cs sd", "arxiv cs mm", "arxiv cs ma", "arxiv cs et", "arxiv cs cg",
                                     "arxiv cs dl", "arxiv cs ce", "arxiv cs oh", "arxiv cs na", "arxiv cs ms",
                                     "arxiv cs sc", "arxiv cs ar", "arxiv cs os", "arxiv cs pf", "arxiv cs gl"]
        ogbnArxiv_paper_class_dict = dict(zip(ogbnArxiv_paper_class_lst, ogbnArxiv_paper_class_lst))
        ogbn_product_class_lst = ["Office Products", "Pet Supplies", "Electronics", "CDs & Vinyl", "Beauty",
                                  "Movies & TV", "Health & Personal Care", "Toys & Games", "Baby", "Software", "Books",
                                  "Tools & Home Improvement", "Sports & Outdoors", "Home & Kitchen",
                                  "Patio, Lawn & Garden", "Cell Phones & Accessories", "Arts, Crafts & Sewing",
                                  "Industrial & Scientific", "Automotive", "nan", "Grocery & Gourmet Food",
                                  "Musical Instruments", "Video Games", "Clothing, Shoes & Jewelry", "Baby Products",
                                  "Computers", "GPS & Navigation", "Office & School Supplies", "All Electronics",
                                  "Appliances", "All Beauty", "Home Improvement", "Collectibles & Fine Art",
                                  "Luxury Beauty", "MP3 Players & Accessories", "Kindle Store", "Car Electronics",
                                  "Buy a Kindle", "Kitchen & Dining", "Magazine Subscriptions", "Camera & Photo",
                                  "Amazon Fashion", "Gift Cards", "#508510", "Digital Music", "Purchase Circles",
                                  "Furniture & D&#233;cor"]
        ogbn_product_class_dict = dict(zip(ogbn_product_class_lst, ogbn_product_class_lst))
        self.arxiv2023_dict_pred = {"paper-label": {'predicate': "http://arxiv2023.org/label", "title": None}}
        self.arxiv2023_dict_pred_class = {"paper-label": {"classes": arxiv2023_paper_class_dict,
                                                     "mid": "76e1c15192b656efa39b988e577395f5c9c04cb6e151a85b52705825ab225b56"}}

        self.ogbnrxiv_dict_pred = {"paper-label": {'predicate': "http://ogbn_arxiv.org/label", "title": None}}
        self.ogbnrxiv_dict_pred_class = {"paper-label": {"classes": ogbnArxiv_paper_class_dict,
                                                    "mid": "f341282d67b0fefde962c660b100a2d1f676d25c37ca512cf272e6959b755ca6"}}

def generate_biokg_targets(SPARQLendpointUrl,by_pubmid_BGP,targets_count=5,offset=5,filter_year=1996,dict_pred=None,class_dict=None):
  Q1=f""" prefix biokg:<http://www.biokg.com/>
  select distinct ?s as ?target  ?p_val as $p_val$ ?drug_name ?pred_txt
  from <http://www.biokg.com>
  where
   {{ ?s a biokg:$biokg_type$.
    ?s <http://www.biokg.com/$biokg_type$-property/NAME> ?drug_name.
    ?s biokg:frist_publish_year ?py.
    filter (?py > {filter_year}).
    ?s $p_predicate$ ?p_val.
    $p_predicate_title$
    $S_Class$
    $2HopBGBs$
    #?s ?p ?o.
  }}
  limit 1000.
  offset {offset}."""

  ground_truth_dict={}
  for k,v in dict_pred.items():
    print(f"""#############{k}#################""")
    col_title=k.split('-')[1]
    print(col_title)
    k_query=Q1.replace("$p_predicate$",f"<{v['predicate']}>").replace("$p_val$",f"?{col_title}")
    k_query=k_query.replace("$biokg_type$",f"{k.split('-')[0]}")
    if k in['drug-class','protein-related_keyword']:
      k_query=k_query.replace("$2HopBGBs$",by_pubmid_BGP[k])
      # if k=='Creative_Work-publisher':
      #   k_query=k_query.replace(f">{str(filter_year)}^^xsd:gYear).",">1990^^xsd:gYear).")
    else:
      k_query=k_query.replace("$2HopBGBs$",f"")
    if v['title'] is not None:
      k_query=k_query.replace("$p_predicate_title$",f"?p_val <{v['title']}> ?pred_txt.")
    else:
      k_query=k_query.replace("$p_predicate_title$",f"")

    if class_dict[k]['classes']:
      class_values=" ".join(['<'+elem+'>' if elem.startswith("http") else "'"+elem+"'" for elem in list(class_dict[k]['classes'].keys())])
    k_query=k_query.replace( "$S_Class$", "" if class_dict[k]['classes'] is None else "values ?p_val {"+class_values+"}" )
    print(k_query)
    ground_truth_dict[k]=executeSparqlQuery(k_query,SPARQLendpointUrl)
    print("count of records=",len(ground_truth_dict[k]))
    ground_truth_dict[k]["target_txt"]=ground_truth_dict[k]["drug_name"]
    # ground_truth_dict[k]["target_txt"]=ground_truth_dict[k]["target"].apply(lambda x: x.split("/")[-1]) # use ID only

    if v['title'] is None: ## no label exist
      ground_truth_dict[k]["pred_txt"]=ground_truth_dict[k][col_title].apply(lambda x: class_dict[k]['classes'][x])
    ground_truth_dict[k][col_title+"_txt"]=ground_truth_dict[k]['pred_txt']
    ################ Keep Balanced instances per Class ################
    category_counts = ground_truth_dict[k].groupby('pred_txt').size()
    print("len of category_counts=",len(category_counts))
    max_count = int(targets_count/len(category_counts))
    max_count=max_count if max_count>0 else 1
    print("max_count per class=",max_count)
    balanced_df = pd.DataFrame()
    for idx ,category in enumerate(category_counts.index):
      if idx<targets_count and len(ground_truth_dict[k][ground_truth_dict[k]['pred_txt'] == category])>=max_count : #keep only targets_count records
        category_samples = ground_truth_dict[k][ground_truth_dict[k]['pred_txt'] == category].sample(n=max_count, random_state=42) # Set random_state for reproducibility
        balanced_df = pd.concat([balanced_df, category_samples])
      else:
        category_samples = ground_truth_dict[k][ground_truth_dict[k]['pred_txt'] == category]
        balanced_df = pd.concat([balanced_df, category_samples])

    ground_truth_dict[k]=balanced_df

    print("len of records=",len(ground_truth_dict[k]))
    print(ground_truth_dict[k][col_title].unique())
  return ground_truth_dict,dict_pred
def generate_biokg_target_context(SPARQLendpointUrl,ground_truth_dict,dict_pred):
  import numpy as np
  Q1_context="""  prefix biokg:<http://www.biokg.com/>
  select distinct ?s as ?target  ?p ?o
  from <http://www.biokg.com>
  where
  { ?s a biokg:$biokg_type$.
    ?s ?p ?o.
    values ?s {$p_s_list$}
  }
  limit 10000. """

  ground_truth_context_dict={}
  for k,v in dict_pred.items():
    print(f"""#############{k}#################""")
    targets_lst=ground_truth_dict[k]["target"].unique().tolist()
    target_lst_str= " ".join(["<"+elem+">" for elem in targets_lst])
    print("target_lst=",target_lst_str)
    col_title=k.split('-')[1]
    print("usecase",col_title)
    k_query=Q1_context.replace("$p_s_list$",target_lst_str)
    k_query=k_query.replace("$biokg_type$",f"{k.split('-')[0]}")
    # print(k_query)
    res=executeSparqlQuery(k_query,SPARQLendpointUrl)
    res=res.drop_duplicates()
    ################### remove prediction info from the context ###################
    # print("usecase predictions=",res[res["p"].eq(v)])
    print("dict_pred V=",v['predicate'])
    res=res[~res["p"].eq(v['predicate'])]
    ###############################################################################
    res=res[res.replace("", np.nan).notna().all(axis=1)] # drop empty or na cells
    to_ignore_predicates=generic_ignore_predicates.copy()
    res= res[~res["p"].isin(to_ignore_predicates)]
    target_context={}
    for elem in targets_lst:
      target_context_df=res[res["target"]==elem]
      ################ Keep Balanced perdicate instances ################
      category_counts = target_context_df.groupby('p').size()
      max_count=3
      balanced_df = pd.DataFrame()
      for idx ,category in enumerate(category_counts.index):
        if len(target_context_df[target_context_df['p'] == category])>max_count:
          category_samples = target_context_df[target_context_df['p'] == category].sample(n=max_count, random_state=42) # Set random_state for reproducibility
        else:
          category_samples = target_context_df[target_context_df['p'] == category]
        balanced_df = pd.concat([balanced_df, category_samples])
      res_lst_lst=balanced_df.values.tolist()
      res_txt=serialize_subgraph(res_lst_lst,ignore_subject=True)
      target_context[elem]=res_txt
    ground_truth_context_dict[k]=target_context
    print(ground_truth_context_dict[k])
  return ground_truth_context_dict
#########################
def generate_yago4_creativeWork_targets(SPARQLendpointUrl,by_performer_BGP,targets_count=5,offset=5,filter_year=1996,dict_pred=None,class_dict=None):
  Q1=f"""select distinct ?s as ?target  ?p_val as $p_val$
  from <https://yago-knowledge.org>
  where
  {{?s a <http://schema.org/CreativeWork>.
    ?s <http://schema.org/datePublished> ?pd.
    filter (?pd >"{str(filter_year)}"^^xsd:gYear).
    ?s $p_predicate$ ?p_val.
    $S_Class$
    #?s ?p ?o.
    $2HopBGBs$
  }}
  limit 1000.
  offset {offset}."""

  ground_truth_dict={}
  for k,v in dict_pred.items():
    print(f"""#############{k}#################""")
    col_title=k.split('-')[1]
    print(col_title)
    k_query=Q1.replace("$p_predicate$",f"<{v}>").replace("$p_val$",f"?{col_title}")
    if k in['Creative_Work-Genere','Creative_Work-Production_Company','Creative_Work-publisher']:
      k_query=k_query.replace("$2HopBGBs$",by_performer_BGP[k])
      if k=='Creative_Work-publisher':
        k_query=k_query.replace(f">{str(filter_year)}^^xsd:gYear).",">1990^^xsd:gYear).")
    else:
      k_query=k_query.replace("$2HopBGBs$",f"")

    if class_dict[k]['classes']:
      class_values=" ".join(['<'+elem+'>' for elem in list(class_dict[k]['classes'].keys())])
    k_query=k_query.replace( "$S_Class$", "" if class_dict[k]['classes'] is None else "values ?p_val {"+class_values+"}" )
    print(k_query)
    ground_truth_dict[k]=executeSparqlQuery(k_query,SPARQLendpointUrl)
    ground_truth_dict[k]["target_txt"]=ground_truth_dict[k]["target"].apply(lambda x:x.split("/")[-1].replace("_"," "))
    ground_truth_dict[k][col_title+"_txt"]=ground_truth_dict[k][col_title].apply(lambda x:x.split("/")[-1].replace("_"," "))
    ################ Keep Balanced instances per Class ################
    ground_truth_dict[k]['pred_txt']=ground_truth_dict[k][col_title+"_txt"]
    category_counts = ground_truth_dict[k].groupby('pred_txt').size()
    print("len of category_counts=",len(category_counts))
    max_count = int(targets_count/len(category_counts))
    max_count=max_count if max_count>0 else 1
    print("max_count per class=",max_count)
    balanced_df = pd.DataFrame()
    for idx ,category in enumerate(category_counts.index):
      if idx<targets_count and len(ground_truth_dict[k][ground_truth_dict[k]['pred_txt'] == category])>=max_count : #keep only targets_count records
        category_samples = ground_truth_dict[k][ground_truth_dict[k]['pred_txt'] == category].sample(n=max_count, random_state=42) # Set random_state for reproducibility
        balanced_df = pd.concat([balanced_df, category_samples])
    ground_truth_dict[k]=balanced_df

    print("len of records=",len(ground_truth_dict[k]))
    print(ground_truth_dict[k][col_title].unique())
  return ground_truth_dict,dict_pred
def generate_yago4_creativeWork_target_context(SPARQLendpointUrl,ground_truth_dict,dict_pred):
  Q1_context="""select distinct ?s as ?target  ?p ?o
  from <https://yago-knowledge.org>
  where
  {?s a <http://schema.org/CreativeWork>.
  ?s ?p ?o.
  values ?s {$p_s_list$}
  }
  limit 10000. """

  ground_truth_context_dict={}
  for k,v in dict_pred.items():
    print(f"""#############{k}#################""")
    targets_lst=ground_truth_dict[k]["target"].unique().tolist()
    target_lst_str= " ".join(["<"+elem+">" for elem in targets_lst])
    print("target_lst=",target_lst_str)
    col_title=k.split('-')[1]
    print("usecase",col_title)
    k_query=Q1_context.replace("$p_s_list$",target_lst_str)
    # print(k_query)
    res=executeSparqlQuery(k_query,SPARQLendpointUrl)
    res=res.drop_duplicates()
    ################### remove prediction info from the context ###################
    # print("usecase predictions=",res[res["p"].eq(v)])
    res=res[~res["p"].eq(v)]
    ###############################################################################
    res=res[res.replace("", np.nan).notna().all(axis=1)] # drop empty or na cells
    to_ignore_predicates=generic_ignore_predicates.copy()
    res= res[~res["p"].isin(to_ignore_predicates)]
    target_context={}
    for elem in targets_lst:
      target_context_df=res[res["target"]==elem]
      ################ Keep Balanced perdicate instances ################
      category_counts = target_context_df.groupby('p').size()
      max_count=3
      balanced_df = pd.DataFrame()
      for idx ,category in enumerate(category_counts.index):
        if len(target_context_df[target_context_df['p'] == category])>max_count:
          category_samples = target_context_df[target_context_df['p'] == category].sample(n=max_count, random_state=42) # Set random_state for reproducibility
        else:
          category_samples = target_context_df[target_context_df['p'] == category]
        balanced_df = pd.concat([balanced_df, category_samples])
      res_lst_lst=balanced_df.values.tolist()
      res_txt=serialize_subgraph(res_lst_lst,ignore_subject=True)
      target_context[elem]=res_txt
    ground_truth_context_dict[k]=target_context
    print(ground_truth_context_dict[k])
  return ground_truth_context_dict
def generate_yago4_person_targets(SPARQLendpointUrl,targets_count=5,offset=5,filter_year=1996,dict_pred=None,class_dict=None):
  Q1=f"""select distinct ?s as ?target  ?p_val as $p_val$
  from <https://yago-knowledge.org>
  where
  {{?s a <http://schema.org/Person>.
  ?s <http://schema.org/birthDate> ?bd.
  filter (?bd >"{str(filter_year)}"^^xsd:gYear).
  ?s $p_predicate$ ?p_val.
  $S_Class$
  #?s ?p ?o.
  }}
  limit 1000.
  offset {offset}."""
  # print("Q1=",Q1)

  ground_truth_dict={}
  for k,v in dict_pred.items():
    print(f"""#############{k}#################""")
    col_title=k.split('-')[1]
    print(col_title)
    k_query=Q1.replace("$p_predicate$",f"<{v}>").replace("$p_val$",f"?{col_title}")
    if class_dict[k]['classes']:
      class_values=" ".join(['<'+elem+'>' for elem in list(class_dict[k]['classes'].keys())])
    k_query=k_query.replace( "$S_Class$", "" if class_dict[k]['classes'] is None else "values ?p_val {"+class_values+"}" )
    print(k_query)
    ground_truth_dict[k]=executeSparqlQuery(k_query,SPARQLendpointUrl)
    ground_truth_dict[k]["target_txt"]=ground_truth_dict[k]["target"].apply(lambda x:x.split("/")[-1].replace("_"," "))
    ground_truth_dict[k][col_title+"_txt"]=ground_truth_dict[k][col_title].apply(lambda x:x.split("/")[-1].replace("_"," "))
    print("cols=",ground_truth_dict[k].columns)
    ################ Keep Balanced instances per Class ################
    ground_truth_dict[k]['pred_txt']=ground_truth_dict[k][col_title+"_txt"]
    category_counts = ground_truth_dict[k].groupby('pred_txt').size()
    print("len of category_counts=",len(category_counts))
    max_count = int(targets_count/len(category_counts))
    max_count=max_count if max_count>0 else 1
    print("max_count per class=",max_count)
    balanced_df = pd.DataFrame()
    for idx ,category in enumerate(category_counts.index):
      if idx<targets_count and len(ground_truth_dict[k][ground_truth_dict[k]['pred_txt'] == category])>=max_count : #keep only targets_count records
        category_samples = ground_truth_dict[k][ground_truth_dict[k]['pred_txt'] == category].sample(n=max_count, random_state=42) # Set random_state for reproducibility
        balanced_df = pd.concat([balanced_df, category_samples])
    ground_truth_dict[k]=balanced_df

    print("len of records=",len(ground_truth_dict[k]))
    print(ground_truth_dict[k][col_title].unique())
  return ground_truth_dict,dict_pred
def generate_yago4_person_target_context(SPARQLendpointUrl,ground_truth_dict,dict_pred):
  import numpy as np
  Q1_context="""select distinct ?s as ?target  ?p ?o
  from <https://yago-knowledge.org>
  where
  {?s a <http://schema.org/Person>.
  ?s ?p ?o.
  values ?s {$p_s_list$}
  }
  limit 10000. """

  ground_truth_context_dict={}
  for k,v in dict_pred.items():
    print(f"""#############{k}#################""")
    targets_lst=ground_truth_dict[k]["target"].unique().tolist()
    target_lst_str= " ".join(["<"+elem+">" for elem in targets_lst])
    print("target_lst=",target_lst_str)
    col_title=k.split('-')[1]
    print("usecase",col_title)
    k_query=Q1_context.replace("$p_s_list$",target_lst_str)
    # print(k_query)
    res=executeSparqlQuery(k_query,SPARQLendpointUrl)
    res=res.drop_duplicates()
    ################### remove prediction info from the context ###################
    # print("usecase predictions=",res[res["p"].eq(v)])
    res=res[~res["p"].eq(v)]
    ###############################################################################
    res=res[res.replace("", np.nan).notna().all(axis=1)] # drop empty or na cells
    to_ignore_predicates=generic_ignore_predicates.copy()
    res= res[~res["p"].isin(to_ignore_predicates)]
    target_context={}
    for elem in targets_lst:
      target_context_df=res[res["target"]==elem]
      ################ Keep Balanced perdicate instances ################
      category_counts = target_context_df.groupby('p').size()
      max_count=3
      balanced_df = pd.DataFrame()
      for idx ,category in enumerate(category_counts.index):
        if len(target_context_df[target_context_df['p'] == category])>max_count:
          category_samples = target_context_df[target_context_df['p'] == category].sample(n=max_count, random_state=42) # Set random_state for reproducibility
        else:
          category_samples = target_context_df[target_context_df['p'] == category]
        balanced_df = pd.concat([balanced_df, category_samples])
      res_lst_lst=balanced_df.values.tolist()
      res_txt=serialize_subgraph(res_lst_lst,ignore_subject=True)
      target_context[elem]=res_txt
    ground_truth_context_dict[k]=target_context
    print(ground_truth_context_dict[k])
  return ground_truth_context_dict
#########################
def generate_crunchbase_targets(SPARQLendpointUrl,targets_count=5,offset=5,filter_year=1996,dict_pred=None,class_dict=None):
  Q1=f""" prefix crunchbase:<http://ontologycentral.com/2010/05/cb/vocab#>
  select distinct ?s as ?target  ?p_val as $p_val$ concat(concat(?fn ,' '),?ln) as ?person_name ?pred_txt
  from <http://crunchbase-dump-2015-10>
  where
   {{ ?s a crunchbase:Person.
      ?s crunchbase:first_name ?fn.
      ?s crunchbase:last_name ?ln.
      ?s crunchbase:born_on ?bd.
      filter (?bd > {filter_year}).
      ?s $p_predicate$ ?p_val.
      $S_Class$
  }}
  limit 1000.
  offset {offset}."""

  ground_truth_dict={}
  for k,v in dict_pred.items():
    print(f"""#############{k}#################""")
    col_title=k.split('-')[1]
    print(col_title)
    k_query=Q1.replace("$p_predicate$",f"{'<'+v['predicate']+'>'}").replace("$p_val$",f"?{col_title}")
    if class_dict[k]['classes']:
      class_values=" ".join(['<'+elem+'>' if elem.startswith("http") else "'"+elem+"'," for elem in list(class_dict[k]['classes'].keys())])
    k_query=k_query.replace( "$S_Class$", "" if class_dict[k]['classes'] is None else "filter (str(?p_val) in ("+class_values+"'0'))." )
    print(k_query)
    ground_truth_dict[k]=executeSparqlQuery(k_query,SPARQLendpointUrl)
    ground_truth_dict[k]["target_txt"]=ground_truth_dict[k]["person_name"]

    if v['title'] is None: ## no label exist
      ground_truth_dict[k]["pred_txt"]=ground_truth_dict[k][col_title].apply(lambda x: class_dict[k]['classes'][x])
    ground_truth_dict[k][col_title+"_txt"]=ground_truth_dict[k]['pred_txt']
    ################ Keep Balanced instances per Class ################
    category_counts = ground_truth_dict[k].groupby('pred_txt').size()
    print("len of category_counts=",len(category_counts))
    max_count = int(targets_count/len(category_counts))
    max_count=max_count if max_count>0 else 1
    print("max_count per class=",max_count)
    balanced_df = pd.DataFrame()
    for idx ,category in enumerate(category_counts.index):
      if idx<targets_count and len(ground_truth_dict[k][ground_truth_dict[k]['pred_txt'] == category])>=max_count : #keep only targets_count records
        category_samples = ground_truth_dict[k][ground_truth_dict[k]['pred_txt'] == category].sample(n=max_count, random_state=42) # Set random_state for reproducibility
        balanced_df = pd.concat([balanced_df, category_samples])
    ground_truth_dict[k]=balanced_df

    print("len of records=",len(ground_truth_dict[k]))
    print(ground_truth_dict[k][col_title].unique())
  return ground_truth_dict,dict_pred
def generate_crunchbase_target_context(SPARQLendpointUrl,ground_truth_dict,dict_pred):
  Q1_context="""  prefix crunchbase:<http://ontologycentral.com/2010/05/cb/vocab#>
  select distinct ?s as ?target  ?p ?o
  from <http://crunchbase-dump-2015-10>
  where
  { ?s a crunchbase:Person.
    ?s ?p ?o.
    FILTER (!isBlank(?o))
    values ?s {$p_s_list$}
  }
  limit 10000. """

  ground_truth_context_dict={}
  for k,v in dict_pred.items():
    print(f"""#############{k}#################""")
    targets_lst=ground_truth_dict[k]["target"].unique().tolist()
    target_lst_str= " ".join(["<"+elem+">" for elem in targets_lst])
    print("target_lst=",target_lst_str)
    col_title=k.split('-')[1]
    print("usecase",col_title)
    k_query=Q1_context.replace("$p_s_list$",target_lst_str)
    res=executeSparqlQuery(k_query,SPARQLendpointUrl)
    res=res.drop_duplicates()
    ################### remove prediction info from the context ###################
    # print("usecase predictions=",res[res["p"].eq(v)])
    print("dict_pred V=",v['predicate'])
    res=res[~res["p"].eq(v['predicate'])]
    ###############################################################################
    res=res[res.replace("", np.nan).notna().all(axis=1)] # drop empty or na cells
    to_ignore_predicates=generic_ignore_predicates.copy()
    res= res[~res["p"].isin(to_ignore_predicates)]
    target_context={}
    for elem in targets_lst:
      target_context_df=res[res["target"]==elem]
      ################ Keep Balanced perdicate instances ################
      category_counts = target_context_df.groupby('p').size()
      max_count=3
      balanced_df = pd.DataFrame()
      for idx ,category in enumerate(category_counts.index):
        if len(target_context_df[target_context_df['p'] == category])>max_count:
          category_samples = target_context_df[target_context_df['p'] == category].sample(n=max_count, random_state=42) # Set random_state for reproducibility
        else:
          category_samples = target_context_df[target_context_df['p'] == category]
        balanced_df = pd.concat([balanced_df, category_samples])
      res_lst_lst=balanced_df.values.tolist()
      res_txt=serialize_subgraph(res_lst_lst,ignore_subject=True)
      target_context[elem]=res_txt
    ground_truth_context_dict[k]=target_context
    print(ground_truth_context_dict[k])
  return ground_truth_context_dict
#########################
def generate_linkedIMDB_targets(SPARQLendpointUrl,targets_count=5,offset=5,filter_year=1996,dict_pred=None,class_dict=None):
  Q1=f""" prefix lkmdb:<http://data.linkedmdb.org/resource/movie/>
  select distinct ?s as ?target  ?p_val as $p_val$ ?title ?pred_txt
  from <https://linkedmdb.org>
  where
   {{ ?s a lkmdb:film .
    ?s <http://purl.org/dc/terms/title> ?title.
    ?s <http://purl.org/dc/terms/date> ?pd.
    $Second_hop_pred$
    filter(strlen(?pd)>=4).
    filter (xsd:int(substr(?pd,0,4)) > {filter_year}).
    ?s $p_predicate$ ?p_val.
    $p_predicate_title$
    $S_Class$
    #?s ?p ?o.
  }}
  limit  1000
  offset {offset}."""

  ground_truth_dict={}
  for k,v in dict_pred.items():
    print(f"""#############{k}#################""")
    col_title=k.split('-')[1]
    print(col_title)
    k_query=Q1.replace("$p_predicate$",f"<{v['predicate']}>").replace("$p_val$",f"?{col_title}")

    if k in ['film-producer']:
      k_query=k_query.replace("$Second_hop_pred$",f"?z <http://data.linkedmdb.org/resource/movie/sequel> ?s.")
      k_query=k_query.replace(f"> {filter_year}).","> 2006).")
    elif k in ['film-genre']:
      k_query=k_query.replace("$Second_hop_pred$",f"?z <http://data.linkedmdb.org/resource/movie/sequel> ?s.")
      k_query=k_query.replace(f"> {filter_year}).","> 2001).")
    else:
      k_query=k_query.replace("$Second_hop_pred$",f"")

    if v['title'] is not None:
      k_query=k_query.replace("$p_predicate_title$",f"?p_val <{v['title']}> ?pred_txt.")
    else:
      k_query=k_query.replace("$p_predicate_title$",f"")

    if class_dict[k]['classes']:
      class_values=" ".join(['<'+elem+'>' for elem in list(class_dict[k]['classes'].keys())])
    k_query=k_query.replace( "$S_Class$", "" if class_dict[k]['classes'] is None else "values ?p_val {"+class_values+"}" )
    print(k_query)
    ground_truth_dict[k]=executeSparqlQuery(k_query,SPARQLendpointUrl)
    ground_truth_dict[k]["target_txt"]=ground_truth_dict[k]["title"]

    if v['title'] is None: ## no label exist
      ground_truth_dict[k]["pred_txt"]=ground_truth_dict[k][col_title].apply(lambda x: class_dict[k]['classes'][x])
    ground_truth_dict[k][col_title+"_txt"]=ground_truth_dict[k]['pred_txt']
    ################ Keep Balanced instances per Class ################
    category_counts = ground_truth_dict[k].groupby('pred_txt').size()
    print("len of category_counts=",len(category_counts))
    max_count = int(targets_count/len(category_counts))
    max_count=max_count if max_count>0 else 1
    print("max_count per class=",max_count)
    balanced_df = pd.DataFrame()
    for idx ,category in enumerate(category_counts.index):
      if idx<targets_count and len(ground_truth_dict[k][ground_truth_dict[k]['pred_txt'] == category])>=max_count : #keep only targets_count records
        category_samples = ground_truth_dict[k][ground_truth_dict[k]['pred_txt'] == category].sample(n=max_count, random_state=42) # Set random_state for reproducibility
        balanced_df = pd.concat([balanced_df, category_samples])
    ground_truth_dict[k]=balanced_df

    print("len of records=",len(ground_truth_dict[k]))
    print(ground_truth_dict[k][col_title].unique())
  return ground_truth_dict,dict_pred
def generate_linkedIMDB_target_context(SPARQLendpointUrl,ground_truth_dict,dict_pred):
  import numpy as np
  Q1_context="""  prefix lkmdb:<http://data.linkedmdb.org/resource/movie/>
  select distinct ?s as ?target  ?p ?o ?o_title
  from <https://linkedmdb.org>
  where
  { ?s a lkmdb:film .
   ?s ?p ?o.
   optional {?o <http://www.w3.org/2000/01/rdf-schema#label> ?o_title.}
   values ?s {$p_s_list$}
  }
  limit 10000. """

  ground_truth_context_dict={}
  for k,v in dict_pred.items():
    print(f"""#############{k}#################""")
    targets_lst=ground_truth_dict[k]["target"].unique().tolist()
    target_lst_str= " ".join(["<"+elem+">" for elem in targets_lst])
    print("target_lst=",target_lst_str)
    col_title=k.split('-')[1]
    print("usecase",col_title)
    k_query=Q1_context.replace("$p_s_list$",target_lst_str)
    # print(k_query)
    res=executeSparqlQuery(k_query,SPARQLendpointUrl)
    res=res.drop_duplicates()
    if 'o_title' in res.columns:
      res["o"]=res.apply(lambda row: row['o_title'] if len(str(row['o_title']))>0 else row['o_title'] , axis=1 )
    ################### remove prediction info from the context ###################
    # print("usecase predictions=",res[res["p"].eq(v)])
    print("dict_pred V=",v['predicate'])
    res=res[~res["p"].eq(v['predicate'])]
    if v['predicate'].split("/")[-1].strip()=='film_subject':
      print('v[predicate]=',v['predicate'])
      print(res["p"].unique())
      res=res[~res["p"].eq('http://www.w3.org/2004/02/skos/core#subject')]
    ###############################################################################
    res=res[res.replace("", np.nan).notna().all(axis=1)] # drop empty or na cells
    to_ignore_predicates=generic_ignore_predicates.copy()
    res= res[~res["p"].isin(to_ignore_predicates)]
    target_context={}
    for elem in targets_lst:
      target_context_df=res[res["target"]==elem]
      ################ Keep Balanced perdicate instances ################
      category_counts = target_context_df.groupby('p').size()
      max_count=3
      balanced_df = pd.DataFrame()
      for idx ,category in enumerate(category_counts.index):
        if len(target_context_df[target_context_df['p'] == category])>max_count:
          category_samples = target_context_df[target_context_df['p'] == category].sample(n=max_count, random_state=42) # Set random_state for reproducibility
        else:
          category_samples = target_context_df[target_context_df['p'] == category]
        balanced_df = pd.concat([balanced_df, category_samples])
      res_lst_lst=balanced_df.values.tolist()
      res_txt=serialize_subgraph(res_lst_lst,ignore_subject=True)
      target_context[elem]=res_txt
    ground_truth_context_dict[k]=target_context
    print(ground_truth_context_dict[k])
  return ground_truth_context_dict
########################AskGNN Datasets ##################
def generate_ogbnArxiv_targets(SPARQLendpointUrl,targets_count=5,offset=5,filter_year=1996,dict_pred=None,class_dict=None):
  Q1=f""" prefix ogbnArxiv:<http://ogbn_arxiv.org/>
  select distinct ?s as ?target  ?p_val as $p_val$ ?paper_title
  from <http://ogbn-arxiv>
  where
   {{ ?s a ogbnArxiv:$ogbnArxiv_type$.
    ?s <http://ogbn_arxiv.org/title> ?paper_title.
    ?s ogbnArxiv:split ?by.
    filter (?by ='2').
    ?s $p_predicate$ ?p_val.
    $p_predicate_title$
  }}
  limit 300.
  offset {offset}."""

  ground_truth_dict={}
  for k,v in dict_pred.items():
    print(f"""#############{k}#################""")
    col_title=k.split('-')[1]
    print(col_title)
    k_query=Q1.replace("$p_predicate$",f"<{v['predicate']}>").replace("$p_val$",f"?{col_title}")
    k_query=k_query.replace("$ogbnArxiv_type$",f"{k.split('-')[0]}")
    if v['title'] is not None:
      k_query=k_query.replace("$p_predicate_title$",f"?p_val <{v['title']}> ?pred_txt.")
    else:
      k_query=k_query.replace("$p_predicate_title$",f"")

    print('class_dict=',class_dict)
    if class_dict[k]['classes']:
      class_values=" ".join(['<'+elem+'>' if elem.startswith("http") else "'"+elem+"'" for elem in list(class_dict[k]['classes'].keys())])
    k_query=k_query.replace( "$S_Class$", "" if class_dict[k]['classes'] is None else "values ?p_val {"+class_values+"}" )
    print(k_query)
    ground_truth_dict[k]=executeSparqlQuery(k_query,SPARQLendpointUrl)
    print("count of records=",len(ground_truth_dict[k]))
    ground_truth_dict[k]["target_txt"]=ground_truth_dict[k]["paper_title"]
    if v['title'] is None: ## no label exist
      ground_truth_dict[k]["pred_txt"]=ground_truth_dict[k][col_title].apply(lambda x: class_dict[k]['classes'][x.replace("'","")])

    ground_truth_dict[k][col_title+"_txt"]=ground_truth_dict[k]['pred_txt']

    print("len of records=",len(ground_truth_dict[k]))
    print(ground_truth_dict[k][col_title].unique())
  return ground_truth_dict,dict_pred
def generate_ogbnArxiv_target_context(SPARQLendpointUrl,ground_truth_dict,dict_pred):
  Q1_context="""  prefix ogbnArxiv:<http://ogbn_arxiv.org/>
  select distinct ?s as ?target  ?p ?o
  from <http://ogbn-arxiv>
  where
  { ?s a ogbnArxiv:$ogbnArxiv_type$.
    ?s ?p ?o.
    values ?s {$p_s_list$}
  }
  limit 1000. """

  ground_truth_context_dict={}
  for k,v in dict_pred.items():
    print(f"""#############{k}#################""")
    targets_lst=ground_truth_dict[k]["target"].unique().tolist()
    target_lst_str= " ".join(["<"+elem+">" for elem in targets_lst])
    print("target_lst=",target_lst_str)
    col_title=k.split('-')[1]
    print("usecase",col_title)
    k_query=Q1_context.replace("$p_s_list$",target_lst_str)
    k_query=k_query.replace("$ogbnArxiv_type$",f"{k.split('-')[0]}")
    # print(k_query)
    res=executeSparqlQuery(k_query,SPARQLendpointUrl)
    res=res.drop_duplicates()
    ################### remove prediction info from the context ###################
    # print("usecase predictions=",res[res["p"].eq(v)])
    print("dict_pred V=",v['predicate'])
    res=res[~res["p"].eq(v['predicate'])]
    ###############################################################################
    res=res[res.replace("", np.nan).notna().all(axis=1)] # drop empty or na cells
    to_ignore_predicates=generic_ignore_predicates.copy()
    res= res[~res["p"].isin(to_ignore_predicates)]
    target_context={}
    for elem in targets_lst:
      target_context_df=res[res["target"]==elem]
      ################ Keep Balanced perdicate instances ################
      category_counts = target_context_df.groupby('p').size()
      max_count=10
      balanced_df = pd.DataFrame()
      for idx ,category in enumerate(category_counts.index):
        if len(target_context_df[target_context_df['p'] == category])>max_count:
          category_samples = target_context_df[target_context_df['p'] == category].sample(n=max_count, random_state=42) # Set random_state for reproducibility
        else:
          category_samples = target_context_df[target_context_df['p'] == category]
        balanced_df = pd.concat([balanced_df, category_samples])
      res_lst_lst=balanced_df.values.tolist()
      res_txt=serialize_subgraph(res_lst_lst,ignore_subject=True)
      target_context[elem]=res_txt
    ground_truth_context_dict[k]=target_context
    print(ground_truth_context_dict[k])
  return ground_truth_context_dict
def generate_arxiv2023_targets(SPARQLendpointUrl,targets_count=5,offset=5,filter_year=1996,dict_pred=None,class_dict=None):
  Q1=f""" prefix arxiv2023:<http://arxiv2023.org/>
  select distinct ?s as ?target  ?p_val as $p_val$ ?paper_title
  from <http://arxiv2023.org>
  where
   {{ ?s a arxiv2023:$arxiv2023_type$.
    #?s <http://arxiv2023.org/arxivId> ?paper_title.
    ?s <http://arxiv2023.org/title> ?paper_title.
    ?s arxiv2023:split ?by.
    filter (?by ='2').
    ?s $p_predicate$ ?p_val.
    $p_predicate_title$
    $S_Class$
    #?s ?p ?o.
  }}
  limit 1000.
  offset {offset}."""

  ground_truth_dict={}
  for k,v in dict_pred.items():
    print(f"""#############{k}#################""")
    col_title=k.split('-')[1]
    print(col_title)
    k_query=Q1.replace("$p_predicate$",f"<{v['predicate']}>").replace("$p_val$",f"?{col_title}")
    k_query=k_query.replace("$arxiv2023_type$",f"{k.split('-')[0]}")
    if v['title'] is not None:
      k_query=k_query.replace("$p_predicate_title$",f"?p_val <{v['title']}> ?pred_txt.")
    else:
      k_query=k_query.replace("$p_predicate_title$",f"")

    if class_dict[k]['classes']:
      class_values=" ".join(['<'+elem+'>' if elem.startswith("http") else "'"+elem+"'" for elem in list(class_dict[k]['classes'].keys())])
    k_query=k_query.replace( "$S_Class$", "" if class_dict[k]['classes'] is None else "values ?p_val {"+class_values+"}" )
    print(k_query)
    ground_truth_dict[k]=executeSparqlQuery(k_query,SPARQLendpointUrl)
    print("count of records=",len(ground_truth_dict[k]))
    ground_truth_dict[k]["target_txt"]=ground_truth_dict[k]["paper_title"]
    if v['title'] is None: ## no label exist
      ground_truth_dict[k]["pred_txt"]=ground_truth_dict[k][col_title].apply(lambda x: class_dict[k]['classes'][x])
    ground_truth_dict[k][col_title+"_txt"]=ground_truth_dict[k]['pred_txt']
    print("len of records=",len(ground_truth_dict[k]))
    print(ground_truth_dict[k][col_title].unique())
  return ground_truth_dict,dict_pred
def generate_arxiv2023_target_context(SPARQLendpointUrl,ground_truth_dict,dict_pred):
  import numpy as np
  Q1_context="""  prefix arxiv2023:<http://arxiv2023.org/>
  select distinct ?s as ?target  ?p ?o
  from <http://arxiv2023.org>
  where
  { ?s a arxiv2023:$arxiv2023_type$.
    ?s ?p ?o.
    values ?s {$p_s_list$}
  }
  limit 700. """

  ground_truth_context_dict={}
  for k,v in dict_pred.items():
    print(f"""#############{k}#################""")
    targets_lst=ground_truth_dict[k]["target"].unique().tolist()
    target_lst_str= " ".join(["<"+elem+">" for elem in targets_lst])
    print("target_lst=",target_lst_str)
    col_title=k.split('-')[1]
    print("usecase",col_title)
    k_query=Q1_context.replace("$p_s_list$",target_lst_str)
    k_query=k_query.replace("$arxiv2023_type$",f"{k.split('-')[0]}")
    # print(k_query)
    res=executeSparqlQuery(k_query,SPARQLendpointUrl)
    res=res.drop_duplicates()
    ################### remove prediction info from the context ###################
    # print("usecase predictions=",res[res["p"].eq(v)])
    print("dict_pred V=",v['predicate'])
    res=res[~res["p"].eq(v['predicate'])]
    ###############################################################################
    res=res[res.replace("", np.nan).notna().all(axis=1)] # drop empty or na cells
    to_ignore_predicates=generic_ignore_predicates.copy()
    res= res[~res["p"].isin(to_ignore_predicates)]
    target_context={}
    for elem in targets_lst:
      target_context_df=res[res["target"]==elem]
      ################ Keep Balanced perdicate instances ################
      category_counts = target_context_df.groupby('p').size()
      max_count=10
      balanced_df = pd.DataFrame()
      for idx ,category in enumerate(category_counts.index):
        if len(target_context_df[target_context_df['p'] == category])>max_count:
          category_samples = target_context_df[target_context_df['p'] == category].sample(n=max_count, random_state=42) # Set random_state for reproducibility
        else:
          category_samples = target_context_df[target_context_df['p'] == category]
        balanced_df = pd.concat([balanced_df, category_samples])
      res_lst_lst=balanced_df.values.tolist()
      res_txt=serialize_subgraph(res_lst_lst,ignore_subject=True)
      target_context[elem]=res_txt
    ground_truth_context_dict[k]=target_context
    print(ground_truth_context_dict[k])
  return ground_truth_context_dict
##########################################################################################################33
def save_targets_and_RC(KG,ground_truth_dict, dict_pred,ground_truth_context_dict):
  with open(f'../GLOW-QA_dataset/{KG}_ground_truth_dict.pickle', 'wb') as file:
    pickle.dump(ground_truth_dict, file)
  with open(f'../GLOW-QA_dataset/{KG}_ground_truth_context_dict.pickle', 'wb') as file:
    pickle.dump(ground_truth_context_dict, file)
  with open(f'../GLOW-QA_dataset/{KG}_ground_dict_pred_class.pickle', 'wb') as file:
    pickle.dump(dict_pred, file)
def load_targets_and_RC(KG):
  with open(f'../GLOW-QA_dataset/{KG}_ground_truth_dict.pickle', 'wb') as file:
    ground_truth_dict=pickle.load(file)
  with open(f'../GLOW-QA_dataset/{KG}_ground_truth_context_dict.pickle', 'wb') as file:
    ground_truth_context_dict=pickle.load(file)
  with open(f'../GLOW-QA_dataset/{KG}_ground_dict_pred_class.pickle', 'wb') as file:
    dict_pred=pickle.load(file)
  return ground_truth_dict, dict_pred,ground_truth_context_dict
def generate_targets_and_RC(kg="biokg",load_from_disk=False):
  ground_truth_dict, dict_pred, ground_truth_context_dict=None,None,None
  if load_from_disk:
    round_truth_dict, dict_pred, ground_truth_context_dict=load_targets_and_RC(kg)
  else:
    if kg=="biokg":
      biokg_ds=biokg_GLOW_Bench(SPARQLendpointUrl_dict[kg])
      ground_truth_dict, dict_pred = generate_biokg_targets(SPARQLendpointUrl_dict[kg],biokg_ds.by_pubmid_BGP, targets_count=100, offset=0, filter_year=2006,
                                                          dict_pred=biokg_ds.drug_dict_pred, class_dict=biokg_ds.drug_dict_pred_class)
      ground_truth_context_dict = generate_biokg_target_context(SPARQLendpointUrl_dict[kg],ground_truth_dict, dict_pred)
      save_targets_and_RC(kg,ground_truth_dict,dict_pred,ground_truth_context_dict)
    elif kg=="yago4-person":
      yago4_ds=yago4_GLOW_Bench(SPARQLendpointUrl_dict[kg.split("-")[0]])
      ground_truth_dict, dict_pred = generate_yago4_person_targets(SPARQLendpointUrl_dict[kg.split("-")[0]], targets_count=50,offset=0,filter_year=1995,
                                                          dict_pred=yago4_ds.person_dict_pred, class_dict=yago4_ds.person_dict_pred_class)
      ground_truth_context_dict = generate_yago4_person_target_context(SPARQLendpointUrl_dict[kg.split("-")[0]],ground_truth_dict, dict_pred)
      save_targets_and_RC(kg,ground_truth_dict,dict_pred,ground_truth_context_dict)
    elif kg=="yago4-creativwork":
      yago4_ds=yago4_GLOW_Bench(SPARQLendpointUrl_dict[kg.split("-")[0]])
      ground_truth_dict, dict_pred = generate_yago4_creativeWork_targets(SPARQLendpointUrl_dict[kg.split("-")[0]],yago4_ds.by_performer_BGP, targets_count=50,offset=0,filter_year=1996,
                                                          dict_pred=yago4_ds.creativeWork_dict_pred, class_dict=yago4_ds.creativeWork_dict_pred_class)
      ground_truth_context_dict = generate_yago4_creativeWork_target_context(SPARQLendpointUrl_dict[kg.split("-")[0]],ground_truth_dict, dict_pred)
      save_targets_and_RC(kg,ground_truth_dict,dict_pred,ground_truth_context_dict)
    elif kg == "crunchbase":
      crunchbase_ds = crunchbase_GLOW_Bench(SPARQLendpointUrl_dict[kg.split("-")[0]])
      ground_truth_dict, dict_pred = generate_crunchbase_targets(SPARQLendpointUrl_dict[kg.split("-")[0]],
                                                                         targets_count=40,offset=20,filter_year=1989,
                                                                         dict_pred=crunchbase_ds.person_dict_pred,
                                                                         class_dict=crunchbase_ds.person_dict_pred_class)
      ground_truth_context_dict = generate_crunchbase_target_context(SPARQLendpointUrl_dict[kg.split("-")[0]],
                                                                             ground_truth_dict, dict_pred)
    elif kg == "linkedIMDB":
      linkedIMDB_ds = linkedIMDB_GLOW_Bench(SPARQLendpointUrl_dict[kg.split("-")[0]])
      ground_truth_dict, dict_pred = generate_linkedIMDB_targets(SPARQLendpointUrl_dict[kg.split("-")[0]],
                                                                 targets_count=30,offset=0,filter_year=2006,
                                                                 dict_pred=linkedIMDB_ds.film_dict_pred,
                                                                 class_dict=linkedIMDB_ds.film_dict_pred_class)
      ground_truth_context_dict = generate_linkedIMDB_target_context(SPARQLendpointUrl_dict[kg.split("-")[0]],
                                                                     ground_truth_dict, dict_pred)
      save_targets_and_RC(kg, ground_truth_dict, dict_pred, ground_truth_context_dict)
    elif kg == "arxiv2023":
      AskGNN_ds = AskGNN_Glow_Bench(SPARQLendpointUrl_dict[kg.split("-")[0]])
      ground_truth_dict, dict_pred = generate_arxiv2023_targets(SPARQLendpointUrl_dict[kg.split("-")[0]],
                                                                 targets_count=50,offset=0,filter_year=2006,
                                                                 dict_pred=AskGNN_ds.arxiv2023_dict_pred,
                                                                 class_dict=AskGNN_ds.arxiv2023_dict_pred_class)
      ground_truth_context_dict = generate_arxiv2023_target_context(SPARQLendpointUrl_dict[kg.split("-")[0]],
                                                                     ground_truth_dict, dict_pred)
      save_targets_and_RC(kg, ground_truth_dict, dict_pred, ground_truth_context_dict)
    elif kg == "ogbnArxiv":
      AskGNN_ds = AskGNN_Glow_Bench(SPARQLendpointUrl_dict[kg.split("-")[0]])
      ground_truth_dict, dict_pred = generate_ogbnArxiv_targets(SPARQLendpointUrl_dict[kg.split("-")[0]],
                                                                 targets_count=50,offset=0,filter_year=2007,
                                                                 dict_pred=AskGNN_ds.ogbnrxiv_dict_pred,
                                                                 class_dict=AskGNN_ds.ogbnrxiv_dict_pred_class)
      ground_truth_context_dict = generate_ogbnArxiv_target_context(SPARQLendpointUrl_dict[kg.split("-")[0]],
                                                                     ground_truth_dict, dict_pred)
      save_targets_and_RC(kg, ground_truth_dict, dict_pred, ground_truth_context_dict)
    elif kg == "ogbnProduct":
        AskGNN_ds = AskGNN_Glow_Bench(SPARQLendpointUrl_dict[kg.split("-")[0]])
        ground_truth_dict, dict_pred = generate_ogbnArxiv_targets(SPARQLendpointUrl_dict[kg.split("-")[0]],
                                                                  targets_count=50, offset=0, filter_year=2007,
                                                                  dict_pred=AskGNN_ds.ogbnrxiv_dict_pred,
                                                                  class_dict=AskGNN_ds.ogbnrxiv_dict_pred_class)
        ground_truth_context_dict = generate_ogbnArxiv_target_context(SPARQLendpointUrl_dict[kg.split("-")[0]],
                                                                      ground_truth_dict, dict_pred)
        save_targets_and_RC(kg, ground_truth_dict, dict_pred, ground_truth_context_dict)

  return ground_truth_dict,dict_pred,ground_truth_context_dict


