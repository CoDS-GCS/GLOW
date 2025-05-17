from Utils.utils import executeSparqlQuery
from Utils.utils import SPARQLendpointUrl
from Utils.utils import generic_ignore_predicates
from Utils.utils import serialize_subgraph
import pandas as pd
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
protien_keyword_class_dict
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
protien_SPECIES_class_dict
###################################################################################
protien_FAMILY_class_lst=["http://www.biokg.com/FAMILY/IPR000276",
"http://www.biokg.com/FAMILY/IPR000725",
"http://www.biokg.com/FAMILY/IPR001128",
"http://www.biokg.com/FAMILY/IPR011701"]
protien_FAMILY_class_dict= dict(zip(protien_FAMILY_class_lst, [elem.split("/")[-1] for elem in protien_FAMILY_class_lst]))
protien_FAMILY_class_dict
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
drug_superclass_class_dict
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
drug_class_class_dict
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
drug_kingdom_class_dict



drug_dict_pred={"drug-superclass":{'predicate':"http://www.biokg.com/drug-property/SUPERCLASS","title":None},
                "drug-class":{'predicate':"http://www.biokg.com/drug-property/CLASS","title":None},
                "drug-kingdom":{'predicate':"http://www.biokg.com/drug-property/KINGDOM","title":None},
                "protein-related_keyword":{'predicate':"http://www.biokg.com/protein-property/RELATED_KEYWORD","title":None},
                "protein-SPECIES":{'predicate':"http://www.biokg.com/protein-property/SPECIES","title":None},
                "protein-FAMILY":{'predicate':"http://www.biokg.com/FAMILY","title":None}}
drug_dict_pred_class={"drug-superclass":{"classes":drug_superclass_class_dict,"mid":"24c3ff5f7399f2dff2bd2e6f26a216eec248323af695a6ecc67d01dab97158bf"},
                      "drug-class":{"classes":drug_class_class_dict,"mid":"b81ecc18c7c8700db2f92f5ead3096586c8a0d4c9cb9b68984dd8e9bf9b30d05"},
                      "drug-kingdom":{"classes":drug_kingdom_class_dict,"mid":"0b977c8df951ce4b1a117d5ddcc3e95be002d4435fb025cf7e7b0e18253bc8d9"},
                      "protein-related_keyword":{"classes":protien_keyword_class_dict,"mid":"b14fedb990ba1a57e5c8108c8c81b9eac65fa3a83af16770f2c3031a280f4c94"},
                      "protein-SPECIES":{"classes":protien_SPECIES_class_dict,"mid":"23cef3c023eb978de13f7f2b52524d799a6cdf4f0804ba22f65686b55bc7bda9"},
                      "protein-FAMILY":{"classes":protien_FAMILY_class_dict,"mid":"1d942651f976fe651780b8e6101379da933da0a5b9bd3e70023b5e0cbdf8ac1b"}}

drug_dict_pred_class



by_pubmid_BGP={"drug-class":"""?s ?p_by ?by.
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


filter_year=2003
def generate_biokg_targets(targets_count=5,offset=5,filter_year=1996,dict_pred=None,class_dict=None):
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
def generate_biokg_target_context(ground_truth_dict,dict_pred):
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

