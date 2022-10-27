
import os
import sys
import gzip
from zipfile import ZipFile
import wget
import requests
import shutil
import subprocess
import pandas as pd
from tqdm import tqdm
#from src import setup_datasets


def parse_drugbank_csv(csv, **kwargs):
    df = pd.read_csv(csv, sep=',', index_col=0)

    col_to_split = 'Drug IDs'
    new_col = 'Drug ID'
    df[col_to_split] = df[col_to_split].apply(lambda x: x.split('; '))
    orig_cols = [c for c in df.columns if c != col_to_split]
    # we want to put each Drug ID on its own row, with the rest of the row copied. 
    # this gets the job done
    df2 = pd.concat([df[col_to_split].apply(pd.Series), df], axis=1) \
        .drop(col_to_split, axis=1) \
        .melt(id_vars=orig_cols, value_name=new_col) \
        .drop("variable", axis=1) \
        .dropna(subset=[new_col])
    return df2


default_pathways_url = "https://webservice.wikipathways.org/listPathways?organism=Homo%20sapiens&format=json"
default_pathway_prots_url = "https://webservice.wikipathways.org/getXrefList?pwId=%s&code=S&format=json"


def download_wikipathways(
        pathways_file, gmt_file,
        pathway_prots_url=default_pathway_prots_url, **kwargs):
    """
    """
    # First download the list of pathways
    # should already be done
    #download_file(pathways_url, pathways_file)
    pathways = json.load(open(pathways_file))
    # Then for each pathway, download the list of prots. "code=S" means get the genes as uniprot IDs
    all_pathway_prots = {}
    descriptions = {} 
    pw_download_dir = pathways_file.split('.')[0]
    #pw_download_dir = os.path.dirname(pathways_file)
    #all_pathway_descriptions = {}
    for pathway in tqdm(pathways['pathways']):
        pw_id = pathway['id']
        pw_name = pathway['name']
        pathway_file = "%s/%s.json" % (pw_download_dir, pw_id)
        # TODO skip writing/reading to/from file
        download_file(pathway_prots_url % pw_id, pathway_file)
        # read in the json file and write a gmt file
        pw_json = json.load(open(pathway_file))
        pw_prots = pw_json['xrefs']
        all_pathway_prots[pw_name] = pw_prots 
        descriptions[pw_name] = pw_id
        # since there will be so many files, delete this one after downloading
        print("\t%d prots for '%s'. Deleting json file." % (len(pw_prots), pw_name))
        #os.remove(pathway_file)

    # now write as a gmt file
    #gmt_file = "%s/%s/%s" % (
    #    genesets_dir, settings['name'], settings['file_name'])
    #write_gmt_file(gmt_file, all_pathway_prots, descriptions) 
    return all_pathway_prots, descriptions


def write_gmt_file(gmt_file, genesets, descriptions={}):
    print("\twriting %s" % (gmt_file))
    # and re-write the file with the specified columns to keep
    open_command = gzip.open if '.gz' in gmt_file else open
    with open_command(gmt_file, 'wb' if '.gz' in gmt_file else 'w') as out:
        for name, prots in genesets.items():
            new_line = "%s\t%s\t%s\n" % (
                name, descriptions.get(name, ""), '\t'.join(prots))
            out.write(new_line.encode() if '.gz' in gmt_file else new_line)


def parse_gmt_file(gmt_file):
    """
    Parse a gmt file and return a dictionary of the geneset names mapped to the list of genes 
    """
    print("Reading %s" % (gmt_file))
    genesets = {}
    descriptions = {}
    open_command = gzip.open if '.gz' in gmt_file else open
    with open_command(gmt_file, 'r') as f:
        for line in f:
            line = line.decode() if '.gz' in gmt_file else line
            if line[0] == '#':
                continue
            line = line.rstrip().split('\t')
            name, description = line[:2]
            genes = line[2:]
            genesets[name] = genes
            descriptions[name] = description 
    print("\t%d gene sets" % (len(genesets)))
    return genesets, descriptions


def run_command(command, chdir=None):
    """
    Run the given command using subprocess. 
    *chdir*: Change to the specified directory before running the command, 
        then change back to the original directory
    """
    if chdir is not None:
        curr_dir = os.getcwd()
        os.chdir(chdir)
    print("Running: %s" % (command))
    subprocess.check_output(command, shell=True)

    if chdir is not None:
        os.chdir(curr_dir)


def gzip_file(f1, f2, remove_orig=True):
    print("\tgzipping %s" % (f1))
    with open(f1, 'rb') as f_in:
        with gzip.open(f2, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    # and remove the original
    if remove_orig:
        os.remove(f1)


def download_file(url, file_path):
    # make sure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print("Downloading to file '%s' from '%s'" % (file_path, url))
    try:
        wget.download(url, file_path)
    # TODO catch specific errors
    #except urllib.error:
    except Exception as e:
        print(e)
        print("WARNING: wget failed. Attempting to use the requests library to download the file")
        # wget didn't work for STRING, gave a 403 error. Using the requests library is working
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(file_path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
        print("\tdone")
    # TODO also add the date of download and url to a README file
    print("")
