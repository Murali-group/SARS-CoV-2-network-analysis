## BioGRID Download and Processing

Download from here: 
https://downloads.thebiogrid.org/File/BioGRID/Release-Archive/BIOGRID-4.4.197/BIOGRID-ORGANISM-4.4.197.tab3.zip

For some reason, I am not able to download this file from the command line. After downloading, unzip with:

```
unzip BIOGRID-ORGANISM-4.4.197.tab3.zip
```

Then, run these commands: 
```
cat BIOGRID-ORGANISM-Homo_sapiens-4.4.198.tab3.txt  | cut -f 4,5,12,13,16,17,18,19 > biogrid-9606.tab

cd ../
mkdir biogrid-y2h
cat biogrid/biogrid-9606.tab | grep "Two-hybrid" > biogrid-y2h/biogrid-9606-two-hybrid.tab 
```

You can then run the masterscript to map these networks to uniprot IDs
```
python src/masterscript.py --config config-files/biogrid.yaml
```
